import argparse
import os
import uuid
from typing import Any, Dict, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from point_and_mask.pipeline import MaskPointPipeline
from point_and_mask.attention_processor import reset_attn_processor, set_text_len


def init_model_mode_0(model_path: str, device: str):
    pipe = MaskPointPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


def init_model_mode_1(model_path: str):
    pipe = MaskPointPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def init_model_mode_2(model_path: str):
    pipe = MaskPointPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    return pipe


def init_model_mode_3(model_path: str):
    pipe = MaskPointPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe


MODEL_INIT_FUNCS = {
    0: init_model_mode_0,
    1: init_model_mode_1,
    2: init_model_mode_2,
    3: init_model_mode_3,
}


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return np.array(data)


def _image_to_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    return np.array(image)


def save_point_match_package(
    output_path: str,
    id_image: Union[Image.Image, np.ndarray],
    frm_image: Union[Image.Image, np.ndarray],
    id_mask: Union[np.ndarray, torch.Tensor],
    frm_mask: Union[np.ndarray, torch.Tensor],
    argmax_indices: Union[np.ndarray, torch.Tensor],
    max_sim: Union[np.ndarray, torch.Tensor],
):
    """Save everything needed for later visualization into a single npz file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        output_path,
        id_image=_image_to_array(id_image),
        frm_image=_image_to_array(frm_image),
        id_mask=_to_numpy(id_mask),
        frm_mask=_to_numpy(frm_mask),
        argmax_indices=_to_numpy(argmax_indices),
        max_sim=_to_numpy(max_sim),
    )


def get_text_tokens_length(pipe, text: str) -> int:
    text_mask = pipe.tokenizer_2(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).attention_mask
    return text_mask.sum().item() - 1


def modify_prompt_and_get_length(bg: str, fg: str, act: str, pipe) -> tuple[str, int, int]:
    bg = (bg or "").strip()
    fg = (fg or "").strip()
    act = (act or "").strip()
    prompt = f"{bg} {fg} {act}".strip()
    if not prompt:
        raise ValueError("Prompt parts cannot all be empty.")
    return prompt, get_text_tokens_length(pipe, bg), get_text_tokens_length(pipe, prompt)


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0, 120)) -> Image.Image:
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
    overlay = Image.new("RGBA", image.size, color)
    mask_rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_rgba.paste(overlay, mask=mask_img)
    return Image.alpha_composite(image.convert("RGBA"), mask_rgba).convert("RGB")


def draw_point(image: Image.Image, point: tuple[int, int], color=(0, 255, 0), radius: int = 12) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 0, 0), width=3, fill=color)
    return annotated


def format_prompt_display(payload: Dict[str, Any], is_id: bool = True, frame_idx_in_payload: int = None) -> str:
    """格式化 prompt 显示，将 bg、fg、act 分开显示
    
    Args:
        payload: .pt 文件的 payload
        is_id: 是否为 id image
        frame_idx_in_payload: 如果是 frame，这是 frame 在 payload["frames"] 中的索引
    """
    if payload is None or "meta" not in payload:
        return "*未保存*"
    
    meta = payload.get("meta", {})
    bg_prompts = meta.get("bg_prompts", [])
    fg_prompt = meta.get("fg_prompt", "")
    act_prompts = meta.get("act_prompts", [])
    
    # 尝试从 frame 的 bg_prompt 和 act_prompt 字段直接获取（如果保存了的话）
    if is_id:
        id_info = payload.get("id", {})
        bg_prompt = id_info.get("bg_prompt", "")
        act_prompt = id_info.get("act_prompt", "")
        # 如果没有单独保存，从 meta 中获取
        if not bg_prompt and bg_prompts and len(bg_prompts) > 0:
            bg_prompt = bg_prompts[0]
        if not act_prompt and act_prompts and len(act_prompts) > 0:
            act_prompt = act_prompts[0]
    else:
        frames = payload.get("frames", [])
        if frame_idx_in_payload is not None and frame_idx_in_payload < len(frames):
            frame = frames[frame_idx_in_payload]
            bg_prompt = frame.get("bg_prompt", "")
            act_prompt = frame.get("act_prompt", "")
            # 如果没有单独保存，尝试从 frame 的 index 推断
            if not bg_prompt or not act_prompt:
                frame_index = frame.get("index", frame_idx_in_payload)
                # frame 的 index 对应 prompts 中的 index+1（因为 id 是 0）
                prompt_idx = frame_index + 1
                if not bg_prompt and bg_prompts and len(bg_prompts) > prompt_idx:
                    bg_prompt = bg_prompts[prompt_idx]
                if not act_prompt and act_prompts and len(act_prompts) > prompt_idx:
                    act_prompt = act_prompts[prompt_idx]
        else:
            bg_prompt = ""
            act_prompt = ""
    
    # 格式化显示
    parts = []
    if bg_prompt:
        parts.append(f"**Background:**\n{bg_prompt}")
    if fg_prompt:
        parts.append(f"**Foreground:**\n{fg_prompt}")
    if act_prompt:
        parts.append(f"**Action:**\n{act_prompt}")
    
    if not parts:
        # 如果没有分离的 prompt，尝试使用完整 prompt
        if is_id:
            id_info = payload.get("id", {})
            full_prompt = id_info.get("prompt", "")
        else:
            frames = payload.get("frames", [])
            if frame_idx_in_payload is not None and frame_idx_in_payload < len(frames):
                full_prompt = frames[frame_idx_in_payload].get("prompt", "")
            else:
                full_prompt = ""
        
        if full_prompt:
            return f"**Full Prompt:**\n{full_prompt}"
        else:
            return "*未保存*"
    
    return "\n\n".join(parts)


class VisualizerApp:
    def __init__(self, args):
        self.args = args
        self.enable_generation = args.enable_generation
        self.pipe = None
        self.precomputed_bank: Dict[str, Dict[str, Any]] = {}
        self.current_data_key: Optional[str] = None  # 保存当前活动的 data_key
        self.loaded_payload: Optional[Dict[str, Any]] = None  # 保存加载的 .pt payload
        self.loaded_file_path: Optional[str] = None  # 保存加载的文件路径
        if self.enable_generation:
            init_fn = MODEL_INIT_FUNCS[args.init_mode]
            if args.init_mode == 0:
                self.pipe = init_fn(args.model_path, args.device)
            else:
                self.pipe = init_fn(args.model_path)
            reset_attn_processor(self.pipe)
        self.pipe_kwargs = dict(height=args.height, width=args.width, guidance_scale=args.guidance_scale)

    def _prepare_prompt(self, bg: str, fg: str, act: str):
        prompt, bg_len, real_len = modify_prompt_and_get_length(bg, fg, act, self.pipe)
        return prompt, bg_len, real_len

    def _run_pipe(self, prompt: str, bg_len: int, real_len: int, is_id: bool, seed: int):
        set_text_len(self.pipe, bg_len, real_len)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        images, spatial_kwargs = self.pipe(
            prompt,
            generator=generator,
            is_id=is_id,
            num_inference_steps=self.args.steps,
            **self.pipe_kwargs,
        )
        return images[0], spatial_kwargs

    def _prepare_state(
        self,
        id_image: Image.Image,
        frm_image: Image.Image,
        id_mask: np.ndarray,
        frm_mask: np.ndarray,
        argmax_indices: np.ndarray,
        max_sim: np.ndarray,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Dict[str, Any]]:
        print("[DEBUG] _prepare_state: Starting...")
        grid_h, grid_w = frm_mask.shape
        print(f"[DEBUG] _prepare_state: grid_size={grid_h}x{grid_w}, image_size={id_image.size}")
        print(f"[DEBUG] _prepare_state: argmax_indices shape={argmax_indices.shape}, max_sim shape={max_sim.shape}")
        
        id_np = np.array(id_image)
        frm_np = np.array(frm_image)
        id_mask_vis = overlay_mask(id_image, id_mask)
        frm_mask_vis = overlay_mask(frm_image, frm_mask, color=(0, 0, 255, 120))
        data_key = str(uuid.uuid4())
        print(f"[DEBUG] _prepare_state: Generated data_key={data_key}")
        
        self.precomputed_bank[data_key] = dict(
            id_image=id_np,
            frm_image=frm_np,
            id_mask_vis=np.array(id_mask_vis),
            frm_mask_vis=np.array(frm_mask_vis),
            argmax_indices=argmax_indices.flatten(),
            max_sim=max_sim.flatten(),
            grid_size=(grid_h, grid_w),
            image_size=id_image.size,
        )
        print(f"[DEBUG] _prepare_state: Saved to bank, bank size={len(self.precomputed_bank)}")
        print(f"[DEBUG] _prepare_state: argmax_indices flattened shape={self.precomputed_bank[data_key]['argmax_indices'].shape}")
        print(f"[DEBUG] _prepare_state: max_sim flattened shape={self.precomputed_bank[data_key]['max_sim'].shape}")
        
        self.current_data_key = data_key  # 保存当前活动的 data_key
        state = dict(data_key=data_key)
        print(f"[DEBUG] _prepare_state: Created state with data_key={data_key}, set current_data_key={data_key}")
        return id_mask_vis, frm_mask_vis, state

    def generate(self, bg_prompt, fg_prompt, act_prompt_1, act_prompt_2, seed):
        if not self.enable_generation or self.pipe is None:
            status = "Generation disabled. Please enable generation or load precomputed data."
            return None, None, None, None, status, {}
        prompt1, bg_len1, real_len1 = self._prepare_prompt(bg_prompt, fg_prompt, act_prompt_1)
        prompt2, bg_len2, real_len2 = self._prepare_prompt(bg_prompt, fg_prompt, act_prompt_2)

        id_image, id_kwargs = self._run_pipe(prompt1, bg_len1, real_len1, True, seed)
        frm_image, frm_kwargs = self._run_pipe(prompt2, bg_len2, real_len2, False, seed + 1)

        id_mask = id_kwargs["curr_fg_mask"][0].cpu().numpy()
        frm_mask = frm_kwargs["curr_fg_mask"][0].cpu().numpy()
        argmax_indices = frm_kwargs["argmax_indices"][0].cpu().numpy()
        max_sim = frm_kwargs["max_sim"][0].cpu().numpy()
        id_mask_vis, frm_mask_vis, state = self._prepare_state(id_image, frm_image, id_mask, frm_mask, argmax_indices, max_sim)

        status = "Generation finished. Click on the second image to see matched points."
        return id_image, frm_image, id_mask_vis, frm_mask_vis, status, state

    def load_precomputed(self, file_obj: Optional[Union[str, Dict[str, str]]]):
        if not file_obj:
            status = "Please upload a saved npz/pt file first."
            return None, None, None, None, status, {}, 0, [], "**Image 1 Prompt:** *未上传*", "**Image 2 Prompt:** *未上传*"

        if isinstance(file_obj, dict):
            file_path = file_obj.get("name")
        elif hasattr(file_obj, "name"):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)

        if not file_path or not os.path.exists(file_path):
            status = f"File not found: {file_path}"
            return None, None, None, None, status, {}, 0, [], "**Image 1 Prompt:** *文件未找到*", "**Image 2 Prompt:** *文件未找到*"

        _, ext = os.path.splitext(file_path)

        if ext.lower() == ".npz":
            data = np.load(file_path, allow_pickle=True)
            id_image = Image.fromarray(data["id_image"])
            frm_image = Image.fromarray(data["frm_image"])
            id_mask = data["id_mask"]
            frm_mask = data["frm_mask"]
            argmax_indices = data["argmax_indices"]
            max_sim = data["max_sim"]
            # .npz 文件不保存 payload
            self.loaded_payload = None
            self.loaded_file_path = file_path
        elif ext.lower() == ".pt":
            payload = torch.load(file_path, map_location="cpu")
            if "id" not in payload or "frames" not in payload or len(payload["frames"]) == 0:
                status = "The .pt file does not contain frame data."
                return None, None, None, None, status, {}, 0, [], "**Image 1 Prompt:** *错误*", "**Image 2 Prompt:** *错误*"
            
            # 保存 payload 以便后续切换 frame
            self.loaded_payload = payload
            self.loaded_file_path = file_path
            
            id_info = payload["id"]
            id_image = Image.fromarray(id_info["image"])
            id_mask = id_info.get("mask")

            # 找到所有可用的 frames（包含完整数据的）
            available_frames = []
            for idx, frame in enumerate(payload["frames"]):
                if (
                    frame.get("argmax_indices") is not None
                    and frame.get("max_sim") is not None
                    and frame.get("mask") is not None
                ):
                    available_frames.append(idx)
            
            if len(available_frames) == 0:
                status = "The .pt cache lacks necessary fields for visualization."
                return None, None, None, None, status, {}, 0, []
            
            # 默认选择第一个可用的 frame
            selected_frame_idx = available_frames[0]
            selected_frame = payload["frames"][selected_frame_idx]

            frm_image = Image.fromarray(selected_frame["image"])
            frm_mask = selected_frame.get("mask")
            argmax_indices = selected_frame.get("argmax_indices")
            max_sim = selected_frame.get("max_sim")

            if argmax_indices is None or max_sim is None or id_mask is None or frm_mask is None:
                status = "The .pt cache lacks necessary fields for visualization."
                return None, None, None, None, status, {}, 0, [], "**Image 1 Prompt:** *错误*", "**Image 2 Prompt:** *错误*"
            
            # 提取 prompt 信息（在 .pt 文件处理中）
            # 使用格式化函数来显示分离的 bg、fg、act
            id_prompt_text = format_prompt_display(payload, is_id=True)
            frm_prompt_text = format_prompt_display(payload, is_id=False, frame_idx_in_payload=selected_frame_idx)
        else:
            status = "Unsupported file type. Please upload .npz or .pt."
            return None, None, None, None, status, {}, 0, [], "**Image 1 Prompt:** *不支持的文件类型*", "**Image 2 Prompt:** *不支持的文件类型*"

        print("[DEBUG] load_precomputed: Preparing state...")
        print(f"[DEBUG] argmax_indices shape: {np.array(argmax_indices).shape}, dtype: {np.array(argmax_indices).dtype}")
        print(f"[DEBUG] max_sim shape: {np.array(max_sim).shape}, dtype: {np.array(max_sim).dtype}")
        
        id_mask_vis, frm_mask_vis, state = self._prepare_state(
            id_image,
            frm_image,
            np.array(id_mask),
            np.array(frm_mask),
            np.array(argmax_indices),
            np.array(max_sim),
        )
        
        print(f"[DEBUG] load_precomputed: state keys: {list(state.keys())}")
        print(f"[DEBUG] load_precomputed: state['data_key']: {state.get('data_key')}")
        
        # 对于 .pt 文件，返回 frame 信息
        if ext.lower() == ".pt" and self.loaded_payload:
            # 格式化 prompt 显示（已经包含标题）
            id_prompt_display_text = f"**Image 1 Prompt:**\n\n{id_prompt_text}"
            frm_prompt_display_text = f"**Image 2 Prompt:**\n\n{frm_prompt_text}"
            
            status = f"Loaded precomputed data from {file_path}. Frame {selected_frame_idx+1}/{len(available_frames)}. Click on Image 2 to inspect matches."
            return id_image, frm_image, id_mask_vis, frm_mask_vis, status, state, selected_frame_idx, available_frames, id_prompt_display_text, frm_prompt_display_text
        else:
            # .npz 文件返回（.npz 文件没有 prompt 信息）
            id_prompt_display_text = "**Image 1 Prompt:** *未保存*"
            frm_prompt_display_text = "**Image 2 Prompt:** *未保存*"
            status = f"Loaded precomputed data from {file_path}. Click on Image 2 to inspect matches."
            return id_image, frm_image, id_mask_vis, frm_mask_vis, status, state, 0, [], id_prompt_display_text, frm_prompt_display_text

    def switch_frame(self, frame_idx: int):
        """切换到指定的 frame"""
        if self.loaded_payload is None:
            status = "Please load a .pt file first."
            return None, None, None, None, status, {}, "**Image 1 Prompt:** *未加载*", "**Image 2 Prompt:** *未加载*"
        
        if "frames" not in self.loaded_payload or frame_idx >= len(self.loaded_payload["frames"]):
            status = f"Frame {frame_idx} not found."
            return None, None, None, None, status, {}, "**Image 1 Prompt:** *未找到*", "**Image 2 Prompt:** *未找到*"
        
        frame = self.loaded_payload["frames"][frame_idx]
        if (
            frame.get("argmax_indices") is None
            or frame.get("max_sim") is None
            or frame.get("mask") is None
        ):
            status = f"Frame {frame_idx} lacks necessary fields."
            return None, None, None, None, status, {}, "**Image 1 Prompt:** *数据不完整*", "**Image 2 Prompt:** *数据不完整*"
        
        id_info = self.loaded_payload["id"]
        id_image = Image.fromarray(id_info["image"])
        id_mask = id_info.get("mask")
        frm_image = Image.fromarray(frame["image"])
        frm_mask = frame.get("mask")
        argmax_indices = frame.get("argmax_indices")
        max_sim = frame.get("max_sim")
        
        print(f"[DEBUG] switch_frame: Switching to frame {frame_idx}")
        id_mask_vis, frm_mask_vis, state = self._prepare_state(
            id_image,
            frm_image,
            np.array(id_mask),
            np.array(frm_mask),
            np.array(argmax_indices),
            np.array(max_sim),
        )
        
        # 提取 prompt 信息（使用格式化函数）
        id_prompt_text = format_prompt_display(self.loaded_payload, is_id=True)
        frm_prompt_text = format_prompt_display(self.loaded_payload, is_id=False, frame_idx_in_payload=frame_idx)
        id_prompt_display_text = f"**Image 1 Prompt:**\n\n{id_prompt_text}"
        frm_prompt_display_text = f"**Image 2 Prompt:**\n\n{frm_prompt_text}"
        
        status = f"Switched to frame {frame_idx+1}. Click on Image 2 to inspect matches."
        return id_image, frm_image, id_mask_vis, frm_mask_vis, status, state, id_prompt_display_text, frm_prompt_display_text

    def handle_click(self, evt):
        print("=" * 60)
        print("[DEBUG] handle_click called")
        print(f"[DEBUG] evt type: {type(evt)}, evt value: {evt}")
        if hasattr(evt, "__dict__"):
            print(f"[DEBUG] evt.__dict__: {evt.__dict__}")
        if hasattr(evt, "__class__"):
            print(f"[DEBUG] evt.__class__: {evt.__class__}")
        if hasattr(evt, "__dir__"):
            print(f"[DEBUG] evt attributes: {[x for x in dir(evt) if not x.startswith('_')]}")
        
        # 直接使用 self.current_data_key，不依赖 state 参数
        data_key = self.current_data_key
        print(f"[DEBUG] Using current_data_key: {data_key}")
        print(f"[DEBUG] precomputed_bank keys: {list(self.precomputed_bank.keys())}")
        
        if data_key is None or data_key not in self.precomputed_bank:
            print(f"[DEBUG] current_data_key not found in bank!")
            status = "Please load data first."
            return None, None, None, None, status, {}
        
        data = self.precomputed_bank[data_key]
        print(f"[DEBUG] data keys: {list(data.keys())}")

        # 尝试从 evt 中提取点击坐标
        click_index = None
        if evt is None:
            print("[DEBUG] evt is None")
            click_index = None
        elif isinstance(evt, dict):
            print(f"[DEBUG] evt is dict, keys: {list(evt.keys())}")
            # 可能是 SelectData 对象被序列化成 dict
            if "index" in evt:
                click_index = evt.get("index")
            elif len(evt) == 2 and all(isinstance(k, (int, float)) for k in evt.keys()):
                # 可能是 {(x, y): value} 的形式
                click_index = tuple(evt.keys())[0] if len(evt) == 1 else None
            print(f"[DEBUG] extracted click_index from dict: {click_index}")
        elif isinstance(evt, (list, tuple)) and len(evt) == 2:
            print(f"[DEBUG] evt is list/tuple: {evt}")
            click_index = tuple(evt)
            print(f"[DEBUG] extracted click_index from tuple: {click_index}")
        elif hasattr(evt, "index"):
            print(f"[DEBUG] evt has .index attribute: {evt.index}")
            click_index = evt.index
        elif hasattr(evt, "value") and isinstance(evt.value, (list, tuple)) and len(evt.value) == 2:
            print(f"[DEBUG] evt has .value attribute with coordinates: {evt.value}")
            click_index = tuple(evt.value)
        else:
            print(f"[DEBUG] evt type not recognized: {type(evt)}")

        if click_index is None:
            print("[DEBUG] click_index is None, returning original images")
            status = "Please click on Image 2 to select a point."
            ret_state = dict(data_key=data_key) if data_key else {}
            ret = (
                Image.fromarray(data["id_image"]),
                Image.fromarray(data["frm_image"]),
                Image.fromarray(data["id_mask_vis"]),
                Image.fromarray(data["frm_mask_vis"]),
                status,
                ret_state,
            )
            print(f"[DEBUG] Returning {len(ret)} values, types: {[type(x) for x in ret]}")
            return ret

        print(f"[DEBUG] click_index: {click_index}")
        x, y = click_index  # (x, y)
        print(f"[DEBUG] x={x}, y={y}")
        
        width, height = data["image_size"]
        grid_h, grid_w = data["grid_size"]
        print(f"[DEBUG] image_size: {width}x{height}, grid_size: {grid_h}x{grid_w}")

        grid_x = min(max(int(x / width * grid_w), 0), grid_w - 1)
        grid_y = min(max(int(y / height * grid_h), 0), grid_h - 1)
        flat_idx = grid_y * grid_w + grid_x
        print(f"[DEBUG] grid_x={grid_x}, grid_y={grid_y}, flat_idx={flat_idx}")

        match_idx = data["argmax_indices"][flat_idx]
        sim = data["max_sim"][flat_idx]
        print(f"[DEBUG] match_idx={match_idx}, sim={sim}")
        
        match_y = match_idx // grid_w
        match_x = match_idx % grid_w

        id_px = int((match_x + 0.5) / grid_w * width)
        id_py = int((match_y + 0.5) / grid_h * height)

        frm_point = (int(x), int(y))
        id_point = (id_px, id_py)
        print(f"[DEBUG] frm_point={frm_point}, id_point={id_point}")

        id_base = Image.fromarray(data["id_image"])
        frm_base = Image.fromarray(data["frm_image"])
        id_annotated = draw_point(id_base, id_point, color=(255, 215, 0))
        frm_annotated = draw_point(frm_base, frm_point, color=(64, 224, 208))

        status = f"Clicked ({frm_point[0]}, {frm_point[1]}) → matched point ({id_point[0]}, {id_point[1]}), similarity={sim:.3f}"
        ret_state = dict(data_key=data_key) if data_key else {}
        ret = (
            id_annotated,
            frm_annotated,
            Image.fromarray(data["id_mask_vis"]),
            Image.fromarray(data["frm_mask_vis"]),
            status,
            ret_state,
        )
        print(f"[DEBUG] Returning {len(ret)} values, types: {[type(x) for x in ret]}")
        print("=" * 60)
        return ret


def build_interface(app: VisualizerApp):
    with gr.Blocks(title="CharaConsist Point Matching") as demo:
        gr.Markdown("# CharaConsist Point & Mask Visualizer")
        gr.Markdown(
            "输入背景、前景、两条动作描述，点击 Generate 生成两张图及其前景掩码。在第二张图上点击任意点，可显示与第一张图中匹配的点。"
        )
        state = gr.State({})

        if app.enable_generation:
            with gr.Row():
                bg_prompt = gr.Textbox(label="Background Prompt", lines=2)
                fg_prompt = gr.Textbox(label="Foreground Prompt", lines=2)
            with gr.Row():
                act_prompt_1 = gr.Textbox(label="Action Prompt (Image 1)", lines=2)
                act_prompt_2 = gr.Textbox(label="Action Prompt (Image 2)", lines=2)
            seed = gr.Number(value=2025, label="Seed", precision=0)
            generate_btn = gr.Button("Generate!")
        else:
            bg_prompt = fg_prompt = act_prompt_1 = act_prompt_2 = seed = None
            gr.Markdown("⚠️ 生成模式已关闭。请直接加载保存的 npz 数据进行可视化。")

        with gr.Row():
            precomputed_file = gr.File(label="Load saved point-matching data (.npz or .pt)", file_types=[".npz", ".pt"])
            load_btn = gr.Button("Load Saved Data")
        
        # Frame 选择器（仅对 .pt 文件有效）
        with gr.Row():
            frame_selector = gr.Dropdown(
                label="选择 Frame (仅 .pt 文件)",
                choices=[],
                value=None,
                interactive=True,
                visible=False
            )
            frame_info = gr.Markdown("")

        # Prompt 显示区域
        with gr.Row():
            id_prompt_display = gr.Markdown("**Image 1 Prompt:** *未加载*", label="Image 1 Prompt")
            frm_prompt_display = gr.Markdown("**Image 2 Prompt:** *未加载*", label="Image 2 Prompt")

        with gr.Row():
            id_image = gr.Image(label="Image 1", type="pil")
            frm_image = gr.Image(label="Image 2 (click to match)", type="pil")
        with gr.Row():
            id_mask = gr.Image(label="Image 1 Mask Overlay", type="pil")
            frm_mask = gr.Image(label="Image 2 Mask Overlay", type="pil")

        # 隐藏的文本框用于存储点击坐标
        click_coords = gr.Textbox(visible=False, value="")
        
        # 添加测试按钮（用于调试）
        with gr.Row():
            test_x = gr.Number(label="Test X", value=512)
            test_y = gr.Number(label="Test Y", value=512)
            test_btn = gr.Button("Test Click (for debugging)")

        status = gr.Markdown()

        if app.enable_generation:
            generate_btn.click(
                fn=app.generate,
                inputs=[bg_prompt, fg_prompt, act_prompt_1, act_prompt_2, seed],
                outputs=[id_image, frm_image, id_mask, frm_mask, status, state],
            )

        def load_and_update_frame_selector(file_obj):
            """加载数据并更新 frame 选择器"""
            result = app.load_precomputed(file_obj)
            if len(result) == 10:  # .pt 文件返回 10 个值（增加了 2 个 prompt）
                id_img, frm_img, id_msk, frm_msk, stat, st, curr_idx, available, id_prompt, frm_prompt = result
                # 更新 frame 选择器
                choices = [f"Frame {idx+1}" for idx in available]
                frame_selector_update = gr.Dropdown.update(
                    choices=choices,
                    value=f"Frame {curr_idx+1}" if choices else None,
                    visible=len(choices) > 1,
                )
                frame_info_update = f"已加载 {len(available)} 个 frames" if available else ""
                return id_img, frm_img, id_msk, frm_msk, stat, st, frame_selector_update, frame_info_update, id_prompt, frm_prompt
            else:  # .npz 文件返回 10 个值（增加了 2 个 prompt）
                id_img, frm_img, id_msk, frm_msk, stat, st, _, _, id_prompt, frm_prompt = result
                frame_selector_update = gr.Dropdown.update(choices=[], value=None, visible=False)
                return id_img, frm_img, id_msk, frm_msk, stat, st, frame_selector_update, "", id_prompt, frm_prompt
        
        load_btn.click(
            fn=load_and_update_frame_selector,
            inputs=[precomputed_file],
            outputs=[id_image, frm_image, id_mask, frm_mask, status, state, frame_selector, frame_info, id_prompt_display, frm_prompt_display],
        )
        
        # Frame 切换功能
        def switch_frame_handler(frame_name):
            """处理 frame 切换"""
            if not frame_name or frame_name == "":
                return None, None, None, None, "Please select a frame.", {}, "**Image 1 Prompt:** *未选择*", "**Image 2 Prompt:** *未选择*"
            try:
                frame_idx = int(frame_name.split()[-1]) - 1  # "Frame 1" -> 0
                return app.switch_frame(frame_idx)
            except Exception as e:
                print(f"[DEBUG] Error switching frame: {e}")
                return None, None, None, None, f"Error: {e}", {}, "**Image 1 Prompt:** *错误*", "**Image 2 Prompt:** *错误*"
        
        frame_selector.change(
            fn=switch_frame_handler,
            inputs=[frame_selector],
            outputs=[id_image, frm_image, id_mask, frm_mask, status, state, id_prompt_display, frm_prompt_display],
        )

        # 当点击坐标更新时，触发处理函数
        def handle_click_from_coords(coords_str):
            """从坐标字符串解析并处理点击"""
            print(f"[DEBUG] handle_click_from_coords called with: {coords_str}")
            if not coords_str or coords_str == "":
                return app.handle_click(None)
            
            try:
                x, y = map(int, coords_str.split(','))
                print(f"[DEBUG] Parsed coordinates: x={x}, y={y}")
                # 创建一个简单的对象来模拟 SelectData
                class FakeSelectData:
                    def __init__(self, x, y):
                        self.index = (x, y)
                evt = FakeSelectData(x, y)
                return app.handle_click(evt)
            except Exception as e:
                print(f"[DEBUG] Error parsing coordinates: {e}")
                return app.handle_click(None)
        
        click_coords.change(
            fn=handle_click_from_coords,
            inputs=[click_coords],
            outputs=[id_image, frm_image, id_mask, frm_mask, status, state],
        )
        
        # 测试按钮：手动输入坐标进行测试
        def test_click(x, y):
            """测试点击功能"""
            print(f"[DEBUG] test_click called with x={x}, y={y}")
            class FakeSelectData:
                def __init__(self, x, y):
                    self.index = (int(x), int(y))
            evt = FakeSelectData(x, y)
            return app.handle_click(evt)
        
        test_btn.click(
            fn=test_click,
            inputs=[test_x, test_y],
            outputs=[id_image, frm_image, id_mask, frm_mask, status, state],
        )
        
        # # 使用事件委托方式捕获点击事件
        # demo.load(
        #     fn=None,
        #     js="""
        #     () => {
        #         console.log('[DEBUG] Page loaded, setting up click handler');
                
        #         // 使用事件委托，在 document 上监听所有点击
        #         document.addEventListener('click', function(e) {
        #             const target = e.target;
        #             if (target.tagName !== 'IMG') return;
                    
        #             // 检查是否是 Image 2
        #             let isImage2 = false;
        #             const parent = target.closest('[class*="gradio"]');
        #             if (parent) {
        #                 const labels = parent.querySelectorAll('label, span');
        #                 for (let label of labels) {
        #                     if (label.textContent && label.textContent.includes('Image 2')) {
        #                         isImage2 = true;
        #                         break;
        #                     }
        #                 }
        #             }
        #             if (!isImage2 && target.alt && target.alt.includes('Image 2')) {
        #                 isImage2 = true;
        #             }
                    
        #             if (isImage2) {
        #                 console.log('[DEBUG] Image 2 clicked!');
        #                 e.preventDefault();
        #                 e.stopPropagation();
                        
        #                 const rect = target.getBoundingClientRect();
        #                 const scaleX = target.naturalWidth / rect.width;
        #                 const scaleY = target.naturalHeight / rect.height;
        #                 const x = Math.round((e.clientX - rect.left) * scaleX);
        #                 const y = Math.round((e.clientY - rect.top) * scaleY);
        #                 const coords = x + ',' + y;
        #                 console.log('[DEBUG] Coordinates:', coords);
                        
        #                 // 查找隐藏的文本框
        #                 const inputs = Array.from(document.querySelectorAll('input[type="text"]'));
        #                 let hiddenInput = null;
        #                 for (let input of inputs) {
        #                     const style = window.getComputedStyle(input);
        #                     if (style.display === 'none' || input.offsetParent === null || input.hidden) {
        #                         hiddenInput = input;
        #                         break;
        #                     }
        #                 }
                        
        #                 if (hiddenInput) {
        #                     hiddenInput.value = coords;
        #                     hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
        #                     console.log('[DEBUG] Updated click_coords');
        #                 } else {
        #                     console.error('[DEBUG] Could not find hidden input!');
        #                 }
        #             }
        #         }, true);
                
        #         // 设置所有图片的光标样式
        #         setTimeout(() => {
        #             const images = document.querySelectorAll('img');
        #             images.forEach(img => {
        #                 const parent = img.closest('[class*="gradio"]');
        #                 if (parent) {
        #                     const labels = parent.querySelectorAll('label, span');
        #                     for (let label of labels) {
        #                         if (label.textContent && label.textContent.includes('Image 2')) {
        #                             img.style.cursor = 'crosshair';
        #                             break;
        #                         }
        #                     }
        #                 }
        #             });
        #         }, 1000);
        #     }
        #     """,
        # )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio visualizer for CharaConsist mask & point matching.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to FLUX model (required if generation enabled).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init_mode", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--share", action="store_true", help="Share Gradio app publicly.")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name (default: 127.0.0.1 for local only).")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port (default: 7860).")
    parser.add_argument(
        "--disable_generation",
        action="store_true",
        help="Skip model loading and only support loading precomputed data.",
    )
    args = parser.parse_args()

    args.enable_generation = not args.disable_generation

    if args.enable_generation and args.model_path is None:
        raise ValueError("--model_path is required unless --disable_generation is set.")

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))

    app = VisualizerApp(args)
    demo = build_interface(app)
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()

