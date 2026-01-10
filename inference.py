import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("--init_mode", type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument("--prompts_file", type=str, default="")
parser.add_argument("--model_path", type=str, default="/path/to/FLUX.1-dev")
parser.add_argument("--out_dir", type=str, default="results")
parser.add_argument("--use_interpolate", action='store_true')
parser.add_argument("--share_bg", action='store_true')
parser.add_argument("--save_mask", action='store_true')
parser.add_argument("--save_point_match", action='store_true')
parser.add_argument("--point_match_dir", type=str, default="")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--visualize_masks", action='store_true', help="Enable mask visualization with matplotlib")
parser.add_argument("--save_visualizations", action='store_true', help="Save mask visualizations to files")
parser.add_argument("--id_only", action='store_true', help="Generate only ID image for quick testing")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu_ids))
import torch
import numpy as np

from models.attention_processor_characonsist import (
    reset_attn_processor,
    set_text_len,
    reset_size,
    reset_id_bank,
)
from models.pipeline_characonsist import CharaConsistPipeline
from datetime import datetime


def init_model_mode_0():
    pipe = CharaConsistPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda:0")
    return pipe

def init_model_mode_1():
    pipe = CharaConsistPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

def init_model_mode_2():
    from diffusers import FluxTransformer2DModel
    from transformers import T5EncoderModel
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="balanced")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, device_map="balanced")
    pipe = CharaConsistPipeline.from_pretrained(
        args.model_path, 
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16, 
        device_map="balanced")
    return pipe

def init_model_mode_3():
    pipe = CharaConsistPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe


MODEL_INIT_FUNCS = {
    0: init_model_mode_0,
    1: init_model_mode_1,
    2: init_model_mode_2,
    3: init_model_mode_3
}

def get_text_tokens_length(pipe, p):
    text_mask = pipe.tokenizer_2(
        p,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).attention_mask
    return text_mask.sum().item() - 1

def modify_prompt_and_get_length(bg, fg, act, pipe):
    """修改后的函数，支持用 # 分隔的多个人物

    Args:
        bg: 背景描述
        fg: 前景描述，可以用 # 分隔多个人物，例如 "person1#person2"
        act: 动作描述
        pipe: pipeline对象

    Returns:
        prompt: 完整prompt
        bg_len: 背景token长度
        real_len: 总token长度
        num_objects: 人物数量 (新增)
        object_token_ranges: 每个人物对应的token范围列表 [(start, end), ...] (新增)
    """
    bg += " "
    act += " " if act else ""

    # 解析 fg_prompt，用 # 分隔不同人物
    fg_parts = [part.strip() for part in fg.split("#") if part.strip()]
    num_objects = len(fg_parts)

    # 构建完整的 prompt
    fg_combined = " ".join(fg_parts) + " "
    prompt = bg + fg_combined + act

    bg_len = get_text_tokens_length(pipe, bg)
    real_len = get_text_tokens_length(pipe, prompt)

    # 计算每个人物对应的 token 范围
    object_token_ranges = []
    if num_objects > 1:
        for i, fg_part in enumerate(fg_parts):
            # 计算到当前人物为止的 prompt 长度
            fg_so_far = " ".join(fg_parts[:i+1]) + " "
            prompt_so_far = bg + fg_so_far
            current_len = get_text_tokens_length(pipe, prompt_so_far)

            if i == 0:
                start_token = bg_len
            else:
                # 前一个人物的结束位置
                prev_fg_so_far = " ".join(fg_parts[:i]) + " "
                prev_prompt_so_far = bg + prev_fg_so_far
                start_token = get_text_tokens_length(pipe, prev_prompt_so_far)

            end_token = current_len
            object_token_ranges.append((start_token, end_token))
    else:
        # 单个人物的情况
        object_token_ranges = [(bg_len, real_len)]

    return prompt, bg_len, real_len, num_objects, object_token_ranges
            
def load_prompt_file(pipe, file_path):
    with open(file_path, "r") as f:
        all_lines = f.readlines()
    all_prompt_info, curr_prompts, curr_bg_len, curr_real_len, curr_num_objects, curr_object_ranges = [], [], [], [], [], []
    for line in all_lines:
        prompt = line.strip()
        if len(prompt) > 0:
            # 处理多对象格式：background#fg1#fg2#...#action
            parts = prompt.split("#")
            if len(parts) >= 3:
                bg = parts[0]
                act = parts[-1]
                fg = "#".join(parts[1:-1])  # 中间的都是fg描述
                prompt, bg_len, real_len, num_objects, object_token_ranges = modify_prompt_and_get_length(bg, fg, act, pipe)
                curr_prompts.append(prompt)
                curr_bg_len.append(bg_len)
                curr_real_len.append(real_len)
                curr_num_objects.append(num_objects)
                curr_object_ranges.append(object_token_ranges)
        else:
            all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len, curr_num_objects, curr_object_ranges))
            curr_prompts, curr_bg_len, curr_real_len, curr_num_objects, curr_object_ranges = [], [], [], [], []
    if len(curr_prompts) > 0:
        all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len, curr_num_objects, curr_object_ranges))
    return all_prompt_info

from PIL import Image
import matplotlib.pyplot as plt

def overlay_mask_on_image(image, mask, color, output_path=None):
    """
    在图像上叠加mask
    Args:
        image: PIL Image
        mask: numpy array, mask
        color: tuple, RGB color
        output_path: str, optional, if provided, save to file
    Returns:
        PIL Image: processed image
    """
    img_array = np.array(image).astype(np.float32) * 0.5
    mask_zero = np.zeros_like(img_array)

    mask_resized = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_resized.resize(image.size, Image.NEAREST)
    mask_resized = np.array(mask_resized)
    mask_resized = mask_resized[:, :, None]
    color = np.array(color, dtype=np.float32).reshape(1, 1, -1)
    mask_resized_color = mask_resized * color
    img_array = img_array + mask_resized_color * 0.5
    mask_zero = mask_zero + mask_resized_color
    out_img = np.concatenate([img_array, mask_zero], axis=1)
    out_img[out_img>255] = 255
    out_img = out_img.astype(np.uint8)
    result_image = Image.fromarray(out_img)
    if output_path is not None:
        result_image.save(output_path)
    return result_image


def visualize_object_masks(image, object_masks, overall_fg_mask=None, title="Object Masks", save_path=None):
    """
    可视化整体前景mask和每个对象的独立mask
    Args:
        image: PIL Image, 原始图像
        object_masks: list of torch.Tensor, 每个对象的mask列表
        overall_fg_mask: torch.Tensor, optional, 整体前景mask
        title: str, 图表标题
        save_path: str, optional, 保存路径
    """
    if object_masks is None or len(object_masks) == 0:
        print("No object masks available")
        return

    num_objects = len(object_masks)
    num_cols = num_objects + 1  # 原始图像
    if overall_fg_mask is not None:
        num_cols += 1  # 增加整体前景mask列

    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # 显示原图
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    col_idx = 1

    # 显示整体前景mask
    if overall_fg_mask is not None:
        # 处理mask维度
        if len(overall_fg_mask.shape) == 3:  # [B, H, W]
            fg_mask = overall_fg_mask[0].cpu().numpy()
        else:  # [H, W]
            fg_mask = overall_fg_mask.cpu().numpy() if hasattr(overall_fg_mask, 'cpu') else overall_fg_mask

        # 创建前景mask overlay (绿色)
        img_array = np.array(image).astype(np.float32) * 0.5
        mask_resized = Image.fromarray((fg_mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
        mask_resized = np.array(mask_resized)
        mask_resized = mask_resized[:, :, None] / 255.0

        # 使用绿色显示前景mask
        fg_color = np.array([0, 255, 0], dtype=np.float32).reshape(1, 1, -1)
        mask_resized_color = mask_resized * fg_color
        img_array = img_array + mask_resized_color * 0.5
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        axes[col_idx].imshow(img_array)
        axes[col_idx].set_title("Overall FG Mask")
        axes[col_idx].axis('off')
        col_idx += 1

    # 为每个对象分配不同颜色（跳过绿色，因为用于前景mask）
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # 显示每个对象的mask
    for obj_idx, obj_mask in enumerate(object_masks):
        # 处理mask维度
        if len(obj_mask.shape) == 3:  # [B, H, W]
            mask = obj_mask[0].cpu().numpy()
        else:  # [H, W]
            mask = obj_mask.cpu().numpy() if hasattr(obj_mask, 'cpu') else obj_mask

        # 创建带颜色的mask overlay
        img_array = np.array(image).astype(np.float32) * 0.5
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
        mask_resized = np.array(mask_resized)
        mask_resized = mask_resized[:, :, None] / 255.0

        color = np.array(colors[obj_idx % len(colors)], dtype=np.float32).reshape(1, 1, -1)
        mask_resized_color = mask_resized * color
        img_array = img_array + mask_resized_color * 0.5
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        axes[col_idx].imshow(img_array)
        axes[col_idx].set_title(f"Object {obj_idx + 1} Mask")
        axes[col_idx].axis('off')
        col_idx += 1

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    plt.show()


def save_point_match_data(out_dir, payload, filename_suffix=""):
    """Save point matching data for visualization"""
    if not args.save_point_match:
        return

    point_match_dir = args.point_match_dir if args.point_match_dir else os.path.join(out_dir, "point_match_data")
    os.makedirs(point_match_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"fg_only_cache{timestamp}{filename_suffix}.pt"
    filepath = os.path.join(point_match_dir, filename)

    torch.save(payload, filepath)
    print(f"Saved point match data to: {filepath}")

if __name__ == "__main__":
    # Model Init
    pipe = MODEL_INIT_FUNCS[args.init_mode]()
    reset_attn_processor(pipe, size=(args.height//16, args.width//16))
    # Load prompts
    all_prompt_info = load_prompt_file(pipe, args.prompts_file)

    pipe_kwargs = dict(
        height = args.height,
        width = args.width,
        use_interpolate = args.use_interpolate,
        share_bg = args.share_bg
    )

    # Collect all prompts for metadata
    all_bg_prompts = []
    all_fg_prompts = []
    all_act_prompts = []

    for prompt_ind, (prompts, bg_lens, real_lens, num_objects_list, object_ranges_list) in enumerate(all_prompt_info):
        out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}")
        os.makedirs(out_dir, exist_ok=True)
        if args.save_mask:
            mask_out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}", "mask")
            os.makedirs(mask_out_dir, exist_ok=True)
        id_prompt = prompts[0]
        frm_prompts = prompts[1:]

        # Parse prompts for metadata (assuming format: bg#fg#act)
        if "#" in id_prompt:
            bg_part, fg_part, act_part = id_prompt.split("#", 2)
            all_bg_prompts.append(bg_part.strip())
            all_fg_prompts.append(fg_part.strip())
            all_act_prompts.append(act_part.strip())
        else:
            # Fallback if format is different
            all_bg_prompts.append("")
            all_fg_prompts.append("")
            all_act_prompts.append(id_prompt)

        # ID Gen
        print("#" * 50)
        print("Generating ID image ...")
        num_objects, object_token_ranges = num_objects_list[0], object_ranges_list[0]
        set_text_len(pipe, bg_lens[0], real_lens[0], num_objects=num_objects, object_token_ranges=object_token_ranges)
        id_images, id_spatial_kwargs = pipe(
            id_prompt, is_id=True, generator = torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs)
        id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
        id_images[0].save(f"{out_dir}/id.jpg")

        # 保存mask（单对象/合并mask）
        if args.save_mask:
            overlay_mask_on_image(id_images[0], id_fg_mask[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_mask.jpg")

        # 检查是否有多个对象mask，进行多对象可视化
        id_object_masks = id_spatial_kwargs.get("curr_fg_masks", None)
        id_overall_fg_mask = id_spatial_kwargs.get("curr_fg_mask", None)

        # 调试信息
        print(f"Debug: id_object_masks is None: {id_object_masks is None}")
        if id_object_masks is not None:
            print(f"Debug: len(id_object_masks) = {len(id_object_masks)}")
        print(f"Debug: id_overall_fg_mask is None: {id_overall_fg_mask is None}")
        if id_overall_fg_mask is not None:
            print(f"Debug: overall_fg_mask shape: {id_overall_fg_mask.shape}")
            print(f"Debug: overall_fg_mask sum: {id_overall_fg_mask.sum().item() if hasattr(id_overall_fg_mask, 'sum') else 'N/A'}")

        if id_object_masks is not None and len(id_object_masks) > 1:
            print(f"Found {len(id_object_masks)} object masks for ID image")
            # 可视化多对象mask
            if args.visualize_masks:
                save_path = f"{mask_out_dir}/id_object_masks.png" if args.save_visualizations else None
                visualize_object_masks(
                    id_images[0],
                    id_object_masks,
                    overall_fg_mask=id_overall_fg_mask,
                    title="ID Image - Overall FG + Individual Object Masks",
                    save_path=save_path
                )

        # Initialize payload for this prompt set
        payload = {
            "id": {
                "image": np.array(id_images[0]),
                "mask": id_fg_mask[0].cpu().numpy(),
                "prompt": id_prompt,
                "bg_prompt": all_bg_prompts[-1],
                "act_prompt": all_act_prompts[-1]
            },
            "frames": [],
            "meta": {
                "bg_prompts": all_bg_prompts,
                "fg_prompt": all_fg_prompts[-1] if all_fg_prompts else "",
                "act_prompts": all_act_prompts,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "model_path": args.model_path,
                "use_interpolate": args.use_interpolate,
                "share_bg": args.share_bg
            }
        }

        # 如果有多个对象mask，添加到ID数据中
        if id_object_masks is not None and len(id_object_masks) > 1:
            payload["id"]["object_masks"] = [mask.cpu().numpy() for mask in id_object_masks]

        # Frame Gen
        if args.id_only:
            print("ID-only mode: Skipping frame generation")
        else:
            spatial_kwargs = dict(id_fg_mask = id_fg_mask, id_bg_mask = ~id_fg_mask)
            print("#" * 50)
            print("Generating frame images ...")
            for ind, prompt in enumerate(frm_prompts):
                frame_num_objects, frame_object_ranges = num_objects_list[1:][ind], object_ranges_list[1:][ind]
                set_text_len(pipe, bg_lens[1:][ind], real_lens[1:][ind], num_objects=frame_num_objects, object_token_ranges=frame_object_ranges)

                # Parse frame prompt
                if "#" in prompt:
                    bg_part, fg_part, act_part = prompt.split("#", 2)
                    frame_bg_prompt = bg_part.strip()
                    frame_act_prompt = act_part.strip()
                else:
                    frame_bg_prompt = ""
                    frame_act_prompt = prompt

                pre_images, spatial_kwargs = pipe(
                    prompt, is_pre_run=True, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs)
                pre_images[0].save(f"{out_dir}/{ind}_pre.jpg")
                images, spatial_kwargs = pipe(
                    prompt, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs)
                images[0].save(f"{out_dir}/{ind}.jpg")

                # 保存mask（单对象/合并mask）
                if args.save_mask:
                    overlay_mask_on_image(images[0], spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_mask.jpg")

                # 检查是否有多个对象mask，进行多对象可视化
                frame_object_masks = spatial_kwargs.get("curr_fg_masks", None)
                frame_overall_fg_mask = spatial_kwargs.get("curr_fg_mask", None)
                if frame_object_masks is not None and len(frame_object_masks) > 1:
                    print(f"Found {len(frame_object_masks)} object masks for frame {ind}")
                    # 可视化多对象mask
                    if args.visualize_masks:
                        save_path = f"{mask_out_dir}/{ind}_object_masks.png" if args.save_visualizations else None
                        visualize_object_masks(
                            images[0],
                            frame_object_masks,
                            overall_fg_mask=frame_overall_fg_mask,
                            title=f"Frame {ind} - Overall FG + Individual Object Masks",
                            save_path=save_path
                        )

                # Save frame data for visualization
                frame_data = {
                    "index": ind,
                    "image": np.array(images[0]),
                    "mask": spatial_kwargs["curr_fg_mask"][0].cpu().numpy(),
                    "prompt": prompt,
                    "bg_prompt": frame_bg_prompt,
                    "act_prompt": frame_act_prompt
                }

                # 如果有多个对象mask，添加到帧数据中
                if frame_object_masks is not None and len(frame_object_masks) > 1:
                    frame_data["object_masks"] = [mask.cpu().numpy() for mask in frame_object_masks]

                # Add point matching data if available
                if "argmax_indices" in spatial_kwargs:
                    frame_data["argmax_indices"] = spatial_kwargs["argmax_indices"][0].cpu().to(torch.int64).numpy()
                if "max_sim" in spatial_kwargs:
                    frame_data["max_sim"] = spatial_kwargs["max_sim"][0].cpu().to(torch.float32).numpy()

                payload["frames"].append(frame_data)

        # Save point match data for this prompt set
        save_point_match_data(args.out_dir, payload, f"_prompt_{prompt_ind}")

        reset_id_bank(pipe)