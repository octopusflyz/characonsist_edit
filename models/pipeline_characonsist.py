# Copied and modified from diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import copy

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from .attention_processor_characonsist import get_curr_fg_mask, get_cross_sim


def get_interpolate_weight(weight, start_step, decay_step, end_step):
    steps = np.arange(0, end_step - decay_step)
    decay_weights = weight * 0.5 * (1 + np.cos(np.pi * steps / (end_step - decay_step)))
    decay_weights = decay_weights.tolist()
    constant_weights = [weight] * (decay_step - start_step)
    weight_list = constant_weights + decay_weights
    weight_dict = dict()
    for ind, interpolate_step in enumerate(range(start_step, end_step)):
        weight_dict[interpolate_step] = weight_list[ind]
    return weight_dict

def get_style_guidance_weight(weight, start_step, end_step):
    """Get style guidance weight schedule with smooth transitions"""
    if weight <= 0:
        return {}
    steps = np.arange(start_step, end_step)

    # Create a smooth bell-shaped curve for more natural style influence
    # This avoids abrupt changes that cause hard transitions
    progress = (steps - start_step) / max(1, end_step - start_step - 1)

    # Gaussian-like curve: smooth rise and fall
    # Peak influence in the middle, gentle transitions at start and end
    sigma = 0.25  # Controls the width of the bell curve
    weights = weight * np.exp(-((progress - 0.5) ** 2) / (2 * sigma ** 2))

    # Ensure non-negative and smooth
    weights = np.maximum(weights, 0)

    weight_dict = dict(zip(steps, weights.tolist()))
    print(f"[STYLE_WEIGHT] Created smooth style transfer schedule: {len(weight_dict)} steps, peak influence: {weight:.4f}")
    return weight_dict


def compute_global_style_statistics(latents, fg_mask=None):
    """
    Compute global style statistics (mean, std) for FLUX latents with foreground masking

    Args:
        latents: FLUX latent tensor of shape (batch, channels, height, width)
        fg_mask: Optional foreground mask of shape (batch, 1, height, width)

    Returns:
        mean, std: Global statistics for each channel
    """
    if fg_mask is not None:
        # Apply foreground mask - reshape to match latent dimensions
        if fg_mask.dim() == 3:  # (batch, height, width)
            fg_mask = fg_mask.unsqueeze(1)  # Add channel dimension -> (batch, 1, height, width)
        fg_mask = F.interpolate(fg_mask.float(), size=latents.shape[-2:], mode='nearest')
        masked_latents = latents * fg_mask

        # Compute statistics only on foreground regions
        valid_pixels = fg_mask.sum(dim=[1, 2, 3], keepdim=True) + 1e-8
        mean = masked_latents.sum(dim=[2, 3], keepdim=True) / valid_pixels
        var = ((masked_latents - mean) ** 2 * fg_mask).sum(dim=[2, 3], keepdim=True) / valid_pixels
        std = torch.sqrt(var + 1e-8)
    else:
        # Global statistics across entire latent
        mean = latents.mean(dim=[2, 3], keepdim=True)
        std = latents.std(dim=[2, 3], keepdim=True) + 1e-8

    return mean, std


def compute_local_texture_grams(latents, fg_mask=None, patch_size=8):
    """
    Compute local texture statistics using Gram matrices on patches

    Args:
        latents: FLUX latent tensor of shape (batch, channels, height, width)
        fg_mask: Optional foreground mask
        patch_size: Size of patches for local texture analysis

    Returns:
        gram_matrices: Local Gram matrices for texture matching
    """
    batch, channels, height, width = latents.shape

    # Ensure patch size doesn't exceed spatial dimensions
    patch_size = min(patch_size, height, width)

    # Extract patches using unfold
    patches = F.unfold(latents, kernel_size=patch_size, stride=patch_size//2)
    patches = patches.view(batch, channels, patch_size*patch_size, -1)
    patches = patches.permute(0, 3, 1, 2)  # (batch, num_patches, channels, patch_size^2)

    if fg_mask is not None:
        # Apply foreground mask to patches - ensure correct shape
        if fg_mask.dim() == 3:  # (batch, height, width)
            fg_mask = fg_mask.unsqueeze(1)  # Add channel dimension -> (batch, 1, height, width)
        fg_mask = F.interpolate(fg_mask.float(), size=latents.shape[-2:], mode='nearest')
        fg_mask_patches = F.unfold(fg_mask, kernel_size=patch_size, stride=patch_size//2)
        fg_mask_patches = fg_mask_patches.view(batch, 1, patch_size*patch_size, -1)
        fg_mask_patches = fg_mask_patches.permute(0, 3, 1, 2)
        patches = patches * fg_mask_patches

    # Compute Gram matrices for each patch
    grams = []
    num_patches = patches.shape[1]

    for patch_idx in range(min(num_patches, 64)):  # Limit to avoid memory issues
        patch = patches[:, patch_idx]  # (batch, channels, patch_size^2)
        if fg_mask is not None:
            # Only compute Gram if patch has foreground content
            fg_pixels = fg_mask_patches[:, patch_idx].sum(dim=[1, 2])
            if fg_pixels.mean() > 0.1:  # At least 10% foreground pixels
                gram = torch.bmm(patch, patch.transpose(1, 2)) / (channels * patch_size * patch_size)
                grams.append(gram)
        else:
            gram = torch.bmm(patch, patch.transpose(1, 2)) / (channels * patch_size * patch_size)
            grams.append(gram)

    if len(grams) == 0:
        return None

    return torch.stack(grams, dim=1)  # (batch, num_valid_patches, channels, channels)


def apply_progressive_style_guidance(latents, id_latents, fg_mask, current_sigma, guidance_scale=1.0):
    """
    Apply progressive style guidance based on sigma (noise level) in FLUX

    Args:
        latents: Current frame latents to be styled
        id_latents: ID latents providing style reference
        fg_mask: Foreground mask to constrain style application
        current_sigma: Current noise level (0-1, higher = more noise)
        guidance_scale: Overall strength of style guidance

    Returns:
        styled_latents: Latents with progressive style guidance applied
    """
    sigma_normalized = current_sigma / 1.0  # Normalize by initial sigma (typically 1.0)

    # Determine guidance weights based on sigma
    # Early (sigma > 0.8): focus on global statistics (color/brightness)
    # Mid (0.3 < sigma < 0.8): add local texture constraints
    # Late (sigma < 0.3): reduce constraints to focus on details

    global_weight = min(1.0, 2.0 * (1 - sigma_normalized)) * guidance_scale  # Increase as noise decreases
    texture_weight = 0.0

    # Local texture matching only in mid-range sigma
    if 0.2 < sigma_normalized < 0.8:
        texture_weight = guidance_scale * (1 - abs(sigma_normalized - 0.5) / 0.3)

    # Late stage: reduce all constraints to focus on details
    if sigma_normalized < 0.2:
        global_weight *= 0.3
        texture_weight *= 0.1

    print(f"[STYLE_GUIDANCE] sigma={current_sigma:.3f}, global_w={global_weight:.3f}, texture_w={texture_weight:.3f}")

    styled_latents = latents.clone()

    # 1. Global statistics matching (Adaptive Instance Normalization)
    if global_weight > 0.01:
        try:
            current_mean, current_std = compute_global_style_statistics(latents, fg_mask)
            target_mean, target_std = compute_global_style_statistics(id_latents, fg_mask)

            # AdaIN: normalize by current statistics, then scale by target statistics
            normalized = (latents - current_mean) / current_std
            global_styled = normalized * target_std + target_mean

            # Smooth interpolation to avoid abrupt changes
            styled_latents = (1 - global_weight) * styled_latents + global_weight * global_styled

        except Exception as e:
            print(f"[STYLE_WARNING] Global statistics guidance failed: {e}")

    # 2. Local texture matching using Gram matrix loss
    if texture_weight > 0.01:
        try:
            print(f"[STYLE_DEBUG] Attempting texture guidance with weight {texture_weight:.3f}")

            current_grams = compute_local_texture_grams(latents, fg_mask)
            target_grams = compute_local_texture_grams(id_latents, fg_mask)

            if current_grams is not None and target_grams is not None:
                # Compute texture loss and gradient-based guidance
                texture_loss = F.mse_loss(current_grams, target_grams)

                if texture_loss.item() > 1e-6:
                    # Use gradient to guide latents toward target texture
                    # Note: We need to ensure latents requires_grad for gradient computation
                    latents_for_grad = latents.detach().clone().requires_grad_(True)
                    current_grams_grad = compute_local_texture_grams(latents_for_grad, fg_mask)

                    if current_grams_grad is not None and len(current_grams_grad) > 0:
                        texture_loss_for_grad = F.mse_loss(current_grams_grad, target_grams)
                        grad = torch.autograd.grad(texture_loss_for_grad, latents_for_grad, retain_graph=False)[0]
                        styled_latents = styled_latents - texture_weight * grad.detach()

                        print(f"[TEXTURE_GUIDANCE] Applied texture guidance, loss={texture_loss.item():.6f}")
                    else:
                        print(f"[STYLE_WARNING] Could not compute gradient for texture guidance")
                else:
                    print(f"[STYLE_DEBUG] Texture loss too small: {texture_loss.item():.6f}")
            else:
                print(f"[STYLE_DEBUG] Texture grams not available: current={current_grams is not None}, target={target_grams is not None}")

        except Exception as e:
            print(f"[STYLE_WARNING] Texture guidance failed: {e}")
            import traceback
            traceback.print_exc()

    return styled_latents

def get_shared_fg_mask(id_fg_mask, curr_fg_mask, curr2id_argmax_indices, curr2id_valid_mask):
    curr_valid_mask = curr2id_valid_mask.flatten()
    id_fg_mask = id_fg_mask.flatten().to(curr2id_argmax_indices.device)
    curr_fg_mask = curr_fg_mask.flatten().to(curr2id_argmax_indices.device)
    rearrange_id_fg_mask = id_fg_mask[curr2id_argmax_indices[0]]
    share_fg_mask = curr_fg_mask & rearrange_id_fg_mask & curr_valid_mask
    id_share_fg_indices = curr2id_argmax_indices[0][share_fg_mask]
    curr_share_fg_indices = torch.nonzero(share_fg_mask).squeeze()
    return id_share_fg_indices, curr_share_fg_indices


class CharaConsistPipeline(FluxPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # extra args
        spatial_kwargs: dict = dict(),
        is_id: bool = False,
        is_pre_run: bool = False,
        use_interpolate: bool = True,
        share_bg: bool = True,
        update_bg: bool = False,
        attn_start_step: int = 1,
        attn_end_step: int = 41,
        interpolate_start_step: int = 1,
        interpolate_decay_step: int = 11,
        interpolate_end_step: int = 31,
        interpolate_weight: float = 0.8,
        sim_thr = 0.5,
        save_mask_point_step: int = 10,
        # Style guidance args
        use_style_guidance: bool = False,
        style_guidance_scale: float = 1.0,
        style_guidance_start_step: int = 1,
        style_guidance_end_step: int = 30,
        style_guidance_start_sigma: float = 0.9,
        style_guidance_end_sigma: float = 0.1,
        # Debug args
        debug_output_dir: str = None
    ):
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        import numpy as np
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        interpolate_weight_dict = get_interpolate_weight(
            interpolate_weight, interpolate_start_step, interpolate_decay_step, interpolate_end_step)

        # Style guidance setup - preserve state across calls
        cfg_key = (use_style_guidance, style_guidance_scale, style_guidance_start_step, style_guidance_end_step, style_guidance_start_sigma, style_guidance_end_sigma)
        current_cfg_key = getattr(self, '_cfg_initialized', None)
        if current_cfg_key != cfg_key:
            print(f"[STYLE_INIT] Style guidance params changed from {current_cfg_key} to {cfg_key}")
            self._use_style_guidance = use_style_guidance
            self._style_guidance_scale = style_guidance_scale
            self._style_guidance_weight_dict = get_style_guidance_weight(
                style_guidance_scale, style_guidance_start_step, style_guidance_end_step)
            self._start_sigma = style_guidance_start_sigma
            self._end_sigma = style_guidance_end_sigma
            self._cfg_initialized = cfg_key
            print(f"[STYLE_INIT] Created progressive style guidance: scale={style_guidance_scale}, sigma_range=({style_guidance_start_sigma:.1f}, {style_guidance_end_sigma:.1f})")
        else:
            print(f"[STYLE_DEBUG] Style guidance params unchanged ({cfg_key}), preserving state")

        # Initialize _id_latents only once per pipeline instance
        if not hasattr(self, '_id_latents'):
            self._id_latents = None
            print(f"[STYLE_DEBUG] Initialized ID latents for style transfer")
        
        # Debug setup - always update to handle changing debug_output_dir
        old_debug_dir = getattr(self, '_debug_output_dir', None)
        if old_debug_dir != debug_output_dir:
            print(f"[DEBUG_SETUP] Debug output dir changed: {old_debug_dir} -> {debug_output_dir}")
            self._debug_output_dir = debug_output_dir
            if debug_output_dir:
                import os
                os.makedirs(debug_output_dir, exist_ok=True)
                print(f"[DEBUG_SETUP] Created debug directory for style transfer analysis: {debug_output_dir}")

        def get_consist_kwargs(i):
            if is_id:
                save_attn_weight = i == save_mask_point_step
                save_attn_kv = (i < attn_end_step) and (i >= attn_start_step)
                update_attn_kv = update_bg & (i < attn_end_step) and (i >= attn_start_step)
                save_attn_out_for_sim = i == save_mask_point_step
                save_attn_out_for_interpolate = use_interpolate and (i < interpolate_end_step) and (i >= interpolate_start_step)
                save_cross_sim = False
                fg_inter_img_attn = False
                bg_inter_img_attn = False
                attn_out_interpolate = False
            elif is_pre_run:
                save_attn_weight = i <= save_mask_point_step
                save_attn_kv = False
                update_attn_kv = False
                save_attn_out_for_sim = False
                save_attn_out_for_interpolate = False
                save_cross_sim = i == save_mask_point_step
                fg_inter_img_attn = False
                bg_inter_img_attn = (i < attn_end_step) and (i >= attn_start_step) and share_bg and (not update_bg)
                attn_out_interpolate = False
            else:
                save_attn_weight = i <= save_mask_point_step
                save_attn_kv = False
                update_attn_kv = update_bg & (i < attn_end_step) and (i >= attn_start_step)
                save_attn_out_for_sim = False
                save_attn_out_for_interpolate = False
                save_cross_sim = i == save_mask_point_step
                fg_inter_img_attn = (i < attn_end_step) and (i >= attn_start_step)
                bg_inter_img_attn = (i < attn_end_step) and (i >= attn_start_step) and share_bg and (not update_bg)
                attn_out_interpolate = use_interpolate & (i < interpolate_end_step) and (i >= interpolate_start_step)
            return dict(
                timestep_ind=i,
                save_attn_weight = save_attn_weight,
                save_attn_kv = save_attn_kv,
                update_attn_kv=update_attn_kv,
                save_attn_out_for_sim = save_attn_out_for_sim,
                save_attn_out_for_interpolate = save_attn_out_for_interpolate,
                save_cross_sim = save_cross_sim,
                fg_inter_img_attn = fg_inter_img_attn,
                bg_inter_img_attn = bg_inter_img_attn,
                attn_out_interpolate = attn_out_interpolate,
                interpolate_weight_dict=interpolate_weight_dict,
                spatial_kwargs=spatial_kwargs)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Debug: Save denoising progress images for steps 40-50
                if hasattr(self, '_debug_output_dir') and self._debug_output_dir and (i + 1) >= 40 and (i + 1) <= 50:
                    try:
                        import os
                        os.makedirs(self._debug_output_dir, exist_ok=True)

                        # Try to decode latents to actual images showing denoising progress
                        try:
                            # For FLUX, we need to properly unpack and scale latents
                            latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)

                            # Apply proper VAE scaling (FLUX specific)
                            latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                            # Decode to image
                            decoded = self.vae.decode(latents_unpacked, return_dict=False)[0]
                            image = self.image_processor.postprocess(decoded, output_type="pil")[0]

                            print(f"[DEBUG_SAVE] Step {i+1}: successfully decoded latent to image")

                        except Exception as decode_error:
                            print(f"[DEBUG_VAE_FAILED] Step {i+1} VAE decode failed ({str(decode_error)}), creating placeholder")
                            # Create a placeholder image showing denoising progress
                            from PIL import Image
                            import numpy as np

                            # Use timestep progress as visual indicator
                            progress = (i + 1) / 50.0  # Assuming 50 steps total
                            brightness = int(255 * (1.0 - progress))  # Dark to light as denoising progresses

                            # Add some texture based on latent values
                            latent_texture = float(latents.std())
                            noise_level = min(50, int(latent_texture * 10))

                            img_array = np.full((height, width, 3), brightness, dtype=np.uint8)
                            # Add some noise to show it's not just a flat color
                            noise = np.random.randint(-noise_level, noise_level+1, (height, width, 3))
                            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

                            image = Image.fromarray(img_array)
                            print(f"[DEBUG_PLACEHOLDER] Step {i+1}: created progress indicator (brightness={brightness}, noise={noise_level})")

                        # Save the image
                        step_num = i + 1
                        cfg_suffix = "_cfg" if (not is_id and not is_pre_run and hasattr(self, '_use_style_guidance') and self._use_style_guidance) else ""
                        gen_type = 'id' if is_id else ('frame_pre' if is_pre_run else 'frame')
                        filename = f"{self._debug_output_dir}/step_{step_num:03d}_{gen_type}{cfg_suffix}.jpg"

                        image.save(filename)
                        print(f"[DEBUG_SAVED] Step {step_num}: denoising progress image -> {filename}")

                    except Exception as e:
                        print(f"[DEBUG_ERROR] Step {i+1} complete failure: {str(e)}")
                        # Minimal fallback
                        try:
                            from PIL import Image
                            import numpy as np
                            error_img = Image.new('RGB', (height, width), (255, 0, 0))  # Red error image
                            error_filename = f"{self._debug_output_dir}/step_{i+1:03d}_error.jpg"
                            error_img.save(error_filename)
                            print(f"[DEBUG_ERROR_IMG] Saved error indicator: {error_filename}")
                        except:
                            pass
                if self.interrupt:
                    continue

                self._joint_attention_kwargs = get_consist_kwargs(i)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.joint_attention_kwargs["save_attn_weight"]:
                    curr_fg_mask = get_curr_fg_mask(self)
                    spatial_kwargs["curr_fg_mask"] = curr_fg_mask
                    if update_bg:
                        spatial_kwargs["id_bg_mask"] = copy.deepcopy(~curr_fg_mask)
                        
                if self.joint_attention_kwargs["save_cross_sim"]:
                    avg_cross_sim = get_cross_sim(self)
                    max_sim, argmax_indices = torch.max(avg_cross_sim, dim=-1)
                    id_fg_inds, curr_fg_inds = get_shared_fg_mask(
                        spatial_kwargs["id_fg_mask"], 
                        spatial_kwargs["curr_fg_mask"], 
                        argmax_indices,
                        max_sim>sim_thr
                    )
                    spatial_kwargs.update(
                        id_fg_inds = id_fg_inds,
                        curr_fg_inds = curr_fg_inds,
                        max_sim=max_sim,
                        argmax_indices=argmax_indices, 
                    )
                
                if is_pre_run and (i == save_mask_point_step):
                    latents = (latents - self.scheduler.sigmas[i] * noise_pred)
                    break
                
                # Apply CFG-style guidance for frame generation
                # DEBUG: Check all conditions at multiple points
                if not is_id and (i == 10 or i == 40):  # Check at steps 10 and 40
                    print(f"[CFG_DEBUG] Step {i}: is_pre_run={is_pre_run}, use_guidance={self._use_style_guidance}, has_id_latents={self._id_latents is not None}, in_dict={i in self._style_guidance_weight_dict}")
                    if self._id_latents is not None:
                        print(f"[CFG_DEBUG] ID latents shape: {self._id_latents.shape}")
                    else:
                        print(f"[CFG_DEBUG] ID latents is None!")

                # Apply CFG-style guidance for frame generation
                # DEBUG: Check all conditions at multiple points
                if not is_id and (i == 10 or i == 40):  # Check at steps 10 and 40
                    print(f"[CFG_DEBUG] Step {i}: is_pre_run={is_pre_run}, use_guidance={self._use_style_guidance}, has_id_latents={self._id_latents is not None}, in_dict={i in self._style_guidance_weight_dict}")
                    if self._id_latents is not None:
                        print(f"[CFG_DEBUG] ID latents shape: {self._id_latents.shape}")
                    else:
                        print(f"[CFG_DEBUG] ID latents is None!")

                if not is_id and not is_pre_run and self._use_style_guidance and self._id_latents is not None:
                    # 获取当前sigma值 (FLUX使用sigma调度)
                    current_sigma = self.scheduler.sigmas[i].item()

                    # 检查是否在有效的sigma范围内
                    if hasattr(self, '_start_sigma') and hasattr(self, '_end_sigma'):
                        if not (self._end_sigma <= current_sigma <= self._start_sigma):
                            continue

                    print(f"[STYLE_ACTIVE] Step {i}: sigma={current_sigma:.3f}")

                    # 获取前景掩码
                    fg_mask = spatial_kwargs.get("curr_fg_mask", None)
                    if fg_mask is not None:
                        fg_mask = fg_mask.to(latents.device)

                    # 应用渐进式风格引导
                    original_latents = latents.clone()
                    styled_latents = apply_progressive_style_guidance(
                        latents, self._id_latents, fg_mask, current_sigma,
                        guidance_scale=self._style_guidance_scale
                    )

                    # 温和的混合策略 - 避免突变
                    alpha = min(0.4, self._style_guidance_scale * (1 - current_sigma))
                    latents = (1 - alpha) * original_latents + alpha * styled_latents

                    # 确保latents保持正确的属性和维度
                    latents = latents.detach()  # 移除梯度计算图，避免干扰transformer
                    latents = latents.to(original_latents.dtype)  # 确保数据类型一致

                    guidance_effect = (latents - original_latents).norm().item()
                    print(f"[STYLE_DETAIL] sigma={current_sigma:.3f}, effect={guidance_effect:.4f}, alpha={alpha:.3f}")

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        # Save ID latents for style guidance (save at CFG start step for better reference)
        if is_id and self._use_style_guidance:
            # Find the timestep corresponding to CFG start step
            cfg_start_step = getattr(self, '_cfg_start_step', 35)  # Default fallback
            if hasattr(self, '_style_guidance_weight_dict') and self._style_guidance_weight_dict:
                cfg_start_step = min(self._style_guidance_weight_dict.keys())

            # For now, save the final latent (could be improved to save at specific step)
            self._id_latents = latents.detach().clone()
            print(f"[CFG] Saved ID latents for style guidance, shape: {self._id_latents.shape}, CFG starts at step {cfg_start_step}")

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, spatial_kwargs)

        return FluxPipelineOutput(images=image)


