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
# Style consistency parameters
parser.add_argument("--use_style_consistency", action='store_true', help="Enable style consistency loss")
parser.add_argument("--style_layers", type=int, nargs='+', default=[8, 16, 24], help="Layers to extract style features")
parser.add_argument("--style_weight", type=float, default=0.1, help="Style loss weight")
parser.add_argument("--style_max_timestep", type=int, default=30, help="Maximum timestep to apply style loss")
# Progressive Style guidance parameters
parser.add_argument("--use_cfg_guidance", action='store_true', help="Enable progressive style guidance")
parser.add_argument("--cfg_guidance_scale", type=float, default=0.3, help="Overall style guidance scale")
parser.add_argument("--cfg_guidance_start_step", type=int, default=1, help="Start step for style guidance (legacy)")
parser.add_argument("--cfg_guidance_end_step", type=int, default=30, help="End step for style guidance (legacy)")
parser.add_argument("--cfg_guidance_start_sigma", type=float, default=0.9, help="Start sigma for progressive style guidance")
parser.add_argument("--cfg_guidance_end_sigma", type=float, default=0.1, help="End sigma for progressive style guidance")
# Debug parameters
parser.add_argument("--debug_output_dir", type=str, default=None, help="Directory to save intermediate images every 10 steps")
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
    bg += " "
    fg += " "
    prompt = bg + fg + act
    return prompt, get_text_tokens_length(pipe, bg), get_text_tokens_length(pipe, prompt)
            
def load_prompt_file(pipe, file_path):
    with open(file_path, "r") as f:
        all_lines = f.readlines()
    all_prompt_info, curr_prompts, curr_bg_len, curr_real_len = [], [], [], []
    for line in all_lines:
        prompt = line.strip()
        if len(prompt) > 0:
            bg, fg, act = prompt.split("#")
            prompt, bg_len, real_len = modify_prompt_and_get_length(bg, fg, act, pipe)
            curr_prompts.append(prompt)
            curr_bg_len.append(bg_len)
            curr_real_len.append(real_len)
        else:
            all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len))
            curr_prompts, curr_bg_len, curr_real_len = [], [], []
    if len(curr_prompts) > 0:
        all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len))
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
    reset_attn_processor(
        pipe,
        size=(args.height//16, args.width//16),
        style_layers=args.style_layers if args.use_style_consistency else None,
        style_weight=args.style_weight if args.use_style_consistency else 0.0,
        style_max_timestep=args.style_max_timestep
    )
    # Load prompts
    all_prompt_info = load_prompt_file(pipe, args.prompts_file)

    pipe_kwargs = dict(
        height = args.height,
        width = args.width,
        use_interpolate = args.use_interpolate,
        share_bg = args.share_bg,
        use_style_guidance = args.use_cfg_guidance,  # Reusing cfg args for style guidance
        style_guidance_scale = args.cfg_guidance_scale,
        style_guidance_start_step = args.cfg_guidance_start_step,
        style_guidance_end_step = args.cfg_guidance_end_step,
        style_guidance_start_sigma = args.cfg_guidance_start_sigma,
        style_guidance_end_sigma = args.cfg_guidance_end_sigma,
        debug_output_dir = args.debug_output_dir
    )

    # DEBUG: Print Progressive Style Guidance parameters
    print(f"[STYLE_PARAMS] use_cfg_guidance: {args.use_cfg_guidance}")
    print(f"[STYLE_PARAMS] cfg_guidance_scale: {args.cfg_guidance_scale}")
    print(f"[STYLE_PARAMS] cfg_guidance_start_sigma: {args.cfg_guidance_start_sigma}")
    print(f"[STYLE_PARAMS] cfg_guidance_end_sigma: {args.cfg_guidance_end_sigma}")
    print(f"[STYLE_PARAMS] pipe_kwargs style params: use_style_guidance={pipe_kwargs.get('use_style_guidance', 'MISSING')}")

    # Collect all prompts for metadata
    all_bg_prompts = []
    all_fg_prompts = []
    all_act_prompts = []

    for prompt_ind, (prompts, bg_lens, real_lens) in enumerate(all_prompt_info):
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
        set_text_len(pipe, bg_lens[0], real_lens[0])
        id_images, id_spatial_kwargs = pipe(
            id_prompt, is_id=True, generator = torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs)
        id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
        id_images[0].save(f"{out_dir}/id.jpg")
        if args.save_mask:
            overlay_mask_on_image(id_images[0], id_fg_mask[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_mask.jpg")

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

        # Frame Gen
        spatial_kwargs = dict(id_fg_mask = id_fg_mask, id_bg_mask = ~id_fg_mask)
        print("#" * 50)
        print("Generating frame images ...")
        for ind, prompt in enumerate(frm_prompts):
            set_text_len(pipe, bg_lens[1:][ind], real_lens[1:][ind])

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
            if args.save_mask:
                overlay_mask_on_image(images[0], spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_mask.jpg")

            # Save frame data for visualization
            frame_data = {
                "index": ind,
                "image": np.array(images[0]),
                "mask": spatial_kwargs["curr_fg_mask"][0].cpu().numpy(),
                "prompt": prompt,
                "bg_prompt": frame_bg_prompt,
                "act_prompt": frame_act_prompt
            }

            # Add point matching data if available
            if "argmax_indices" in spatial_kwargs:
                frame_data["argmax_indices"] = spatial_kwargs["argmax_indices"][0].cpu().to(torch.int64).numpy()
            if "max_sim" in spatial_kwargs:
                frame_data["max_sim"] = spatial_kwargs["max_sim"][0].cpu().to(torch.float32).numpy()

            payload["frames"].append(frame_data)

        # Save point match data for this prompt set
        save_point_match_data(args.out_dir, payload, f"_prompt_{prompt_ind}")

        reset_id_bank(pipe)