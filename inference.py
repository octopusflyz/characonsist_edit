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
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--seed", type=int, default=2025)
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
        num_objects: 人物数量
        object_token_ranges: 每个人物对应的token范围列表 [(start, end), ...]
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
    all_prompt_info, curr_prompts, curr_bg_len, curr_real_len, curr_object_info = [], [], [], [], []
    for line in all_lines:
        prompt = line.strip()
        if len(prompt) > 0:
            parts = prompt.split("#")
            if len(parts) < 3:
                raise ValueError(f"Prompt format error: expected at least bg#fg#act, got {len(parts)} parts")
            
            # 第一个 # 之前是 bg
            bg = parts[0]
            # 最后一个 # 之后是 act
            act = parts[-1]
            # 中间的所有部分（用 # 分隔）是 fg 中的多个人物
            fg_parts = parts[1:-1]  # 中间所有部分
            fg = "#".join(fg_parts)  # 重新组合，保持原始格式用于后续解析
            
            prompt, bg_len, real_len, num_objects, object_token_ranges = modify_prompt_and_get_length(
                bg, fg, act, pipe
            )
            curr_prompts.append(prompt)
            curr_bg_len.append(bg_len)
            curr_real_len.append(real_len)
            curr_object_info.append((num_objects, object_token_ranges))
        else:
            all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len, curr_object_info))
            curr_prompts, curr_bg_len, curr_real_len, curr_object_info = [], [], [], []
    if len(curr_prompts) > 0:
        all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len, curr_object_info))
    return all_prompt_info

from PIL import Image
def overlay_mask_on_image(image, mask, color, output_path):
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
    Image.fromarray(out_img).save(output_path)


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

    for prompt_ind, (prompts, bg_lens, real_lens, object_info_list) in enumerate(all_prompt_info):
        out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}")
        os.makedirs(out_dir, exist_ok=True)
        if args.save_mask:
            mask_out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}", "mask")
            os.makedirs(mask_out_dir, exist_ok=True)
        id_prompt = prompts[0]
        frm_prompts = prompts[1:]

        # ID Gen
        print("#" * 50)
        print("Generating ID image ...")
        num_objects, object_token_ranges = object_info_list[0]
        set_text_len(pipe, bg_lens[0], real_lens[0], num_objects=num_objects, object_token_ranges=object_token_ranges)
        id_images, id_spatial_kwargs = pipe(
            id_prompt, is_id=True, generator = torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs)
        id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
        id_images[0].save(f"{out_dir}/id.jpg")
        if args.save_mask:
            overlay_mask_on_image(id_images[0], id_fg_mask[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_mask.jpg")

        # Frame Gen
        spatial_kwargs = dict(id_fg_mask = id_fg_mask, id_bg_mask = ~id_fg_mask)
        # 如果是多对象模式，传递 id_object_masks
        if "id_object_masks" in id_spatial_kwargs:
            spatial_kwargs["id_object_masks"] = id_spatial_kwargs["id_object_masks"]
            print(f"Multi-object mode: {len(id_spatial_kwargs['id_object_masks'])} objects detected for point matching")
        print("#" * 50)
        print("Generating frame images ...")
        for ind, prompt in enumerate(frm_prompts):
            num_objects, object_token_ranges = object_info_list[1:][ind]
            set_text_len(pipe, bg_lens[1:][ind], real_lens[1:][ind], num_objects=num_objects, object_token_ranges=object_token_ranges)
            pre_images, spatial_kwargs = pipe(
                prompt, is_pre_run=True, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs) 
            pre_images[0].save(f"{out_dir}/{ind}_pre.jpg")       
            images, spatial_kwargs = pipe(
                prompt, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs)
            images[0].save(f"{out_dir}/{ind}.jpg")
            if args.save_mask:
                overlay_mask_on_image(images[0], spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_mask.jpg")
        reset_id_bank(pipe)