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
parser.add_argument("--save_cache", action='store_true', help="Save intermediate results to .pt file for visualization")
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
from visualize_gr import save_point_match_package
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
    bg += " "
    fg += " "
    prompt = bg + fg + act
    return prompt, get_text_tokens_length(pipe, bg), get_text_tokens_length(pipe, prompt)
            
def load_prompt_file(pipe, file_path):
    with open(file_path, "r") as f:
        all_lines = f.readlines()
    all_prompt_info, curr_prompts, curr_bg_len, curr_real_len, curr_bg_fg_act = [], [], [], [], []
    for line in all_lines:
        prompt = line.strip()
        if len(prompt) > 0:
            bg, fg, act = prompt.split("#")
            prompt, bg_len, real_len = modify_prompt_and_get_length(bg, fg, act, pipe)
            curr_prompts.append(prompt)
            curr_bg_len.append(bg_len)
            curr_real_len.append(real_len)
            curr_bg_fg_act.append((bg, fg, act))
        else:
            all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len, curr_bg_fg_act))
            curr_prompts, curr_bg_len, curr_real_len, curr_bg_fg_act = [], [], [], []
    if len(curr_prompts) > 0:
        all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len, curr_bg_fg_act))
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

    for prompt_ind, (prompts, bg_lens, real_lens, bg_fg_act_list) in enumerate(all_prompt_info):
        out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}")
        os.makedirs(out_dir, exist_ok=True)
        if args.save_mask:
            mask_out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}", "mask")
            os.makedirs(mask_out_dir, exist_ok=True)
        if args.save_cache:
            point_match_save_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}", "point_pkgs")
            os.makedirs(point_match_save_dir, exist_ok=True)
            cache_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"cache_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
            cache_payload = None
        
        # 提取 bg_prompts, fg_prompt, act_prompts
        bg_prompts = [item[0] for item in bg_fg_act_list]
        # 检查所有 prompt 是否共享同一个 fg（通常情况）
        fg_prompts = [item[1] for item in bg_fg_act_list]
        if len(set(fg_prompts)) == 1:
            fg_prompt = fg_prompts[0]  # 所有 prompt 共享同一个 fg
        else:
            fg_prompt = fg_prompts[0]  # 如果不相同，使用第一个（用于兼容性）
        act_prompts = [item[2] for item in bg_fg_act_list]
        
        id_prompt = prompts[0]
        frm_prompts = prompts[1:]

        # ID Gen
        print("#" * 50)
        print("Generating ID image ...")
        set_text_len(pipe, bg_lens[0], real_lens[0])
        id_images, id_spatial_kwargs = pipe(
            id_prompt, is_id=True, generator = torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs)
        id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
        id_images[0].save(f"{out_dir}/id.jpg")
        if args.save_mask:
            overlay_mask_on_image(id_images[0], id_fg_mask[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_mask.jpg")
        
        if args.save_cache:
            cache_payload = dict(
                meta=dict(
                    bg_prompts=bg_prompts,
                    fg_prompt=fg_prompt,
                    act_prompts=act_prompts,
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    init_mode=args.init_mode,
                ),
                id=dict(
                    prompt=id_prompt,
                    bg_prompt=bg_prompts[0],
                    act_prompt=act_prompts[0],
                    bg_len=bg_lens[0],
                    real_len=real_lens[0],
                    image=np.array(id_images[0]),
                    mask=id_fg_mask[0].cpu().numpy(),
                ),
                frames=[],
            )

        # Frame Gen
        spatial_kwargs = dict(id_fg_mask = id_fg_mask, id_bg_mask = ~id_fg_mask)
        print("#" * 50)
        print("Generating frame images ...")
        for ind, prompt in enumerate(frm_prompts):    
            set_text_len(pipe, bg_lens[1:][ind], real_lens[1:][ind])
            pre_images, spatial_kwargs = pipe(
                prompt, is_pre_run=True, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs) 
            pre_images[0].save(f"{out_dir}/{ind}_pre.jpg")       
            images, spatial_kwargs = pipe(
                prompt, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs)
            images[0].save(f"{out_dir}/{ind}.jpg")
            if args.save_mask:
                overlay_mask_on_image(images[0], spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_mask.jpg")
            
            if args.save_cache and cache_payload is not None:
                frame_entry = dict(
                    index=ind,
                    prompt=prompt,
                    bg_prompt=bg_prompts[1:][ind],
                    act_prompt=act_prompts[1:][ind],
                    bg_len=bg_lens[1:][ind],
                    real_len=real_lens[1:][ind],
                    image=np.array(images[0]),
                    mask=spatial_kwargs["curr_fg_mask"][0].cpu().numpy(),
                )
                if "argmax_indices" in spatial_kwargs:
                    frame_entry["argmax_indices"] = spatial_kwargs["argmax_indices"][0].cpu().numpy()
                if "max_sim" in spatial_kwargs:
                    frame_entry["max_sim"] = spatial_kwargs["max_sim"][0].float().cpu().numpy()
                cache_payload["frames"].append(frame_entry)

                if ("argmax_indices" in spatial_kwargs) and ("max_sim" in spatial_kwargs):
                    npz_path = os.path.join(point_match_save_dir, f"frame_{ind:02d}.npz")
                    save_point_match_package(
                        npz_path,
                        id_images[0],
                        images[0],
                        id_fg_mask[0],
                        spatial_kwargs["curr_fg_mask"][0],
                        spatial_kwargs["argmax_indices"][0],
                        spatial_kwargs["max_sim"][0].float(),
                    )
        
        if args.save_cache and cache_payload is not None:
            torch.save(cache_payload, cache_path)
            print(f"Intermediate data saved to {cache_path}")
        
        reset_id_bank(pipe)