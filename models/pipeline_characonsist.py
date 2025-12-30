# Copied and modified from diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
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
        save_mask_point_step: int = 10
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

    @torch.no_grad()
    def from_reference_image(
        self,
        reference_image: Union["PIL.Image.Image", torch.Tensor],
        prompt: str,
        bg_len: Optional[int] = None,
        real_len: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_interpolate: bool = True,
        share_bg: bool = True,
        attn_start_step: int = 1,
        attn_end_step: int = 41,
        interpolate_start_step: int = 1,
        interpolate_decay_step: int = 11,
        interpolate_end_step: int = 31,
        interpolate_weight: float = 0.8,
        sim_thr: float = 0.5,
        save_mask_point_step: int = 10,
        **kwargs
    ):
        """
        从参考图像初始化ID特征
        
        Args:
            reference_image: PIL Image 或 torch.Tensor，参考图像
            prompt: 描述参考图像的完整prompt
            bg_len: 背景部分的token长度（如果已知）
            real_len: 完整prompt的token长度（如果已知）
            height, width: 图像尺寸（如果不提供，将使用参考图像的尺寸）
            num_inference_steps: 推理步数（可以设置较小值，因为我们主要是获取特征）
            **kwargs: 其他参数传递给 __call__ 方法
        
        Returns:
            id_images: 生成的图像（应该与参考图像相似）
            id_spatial_kwargs: 包含 id_fg_mask, id_attn_bank 等中间特征
        """
        from PIL import Image
        
        device = self._execution_device
        
        # 1. 处理参考图像
        if isinstance(reference_image, Image.Image):
            # 获取图像尺寸
            if height is None:
                height = reference_image.height
            if width is None:
                width = reference_image.width
            
            # 预处理图像
            processed_image = self.image_processor.preprocess(reference_image)
        else:
            # 如果已经是tensor，直接使用
            processed_image = reference_image
            if height is None or width is None:
                raise ValueError("When reference_image is a tensor, height and width must be provided")
        
        # 确保图像在正确的device和dtype上
        # 检查VAE的实际dtype（可能是bfloat16）
        if not isinstance(processed_image, torch.Tensor):
            processed_image = torch.tensor(processed_image)
        # 获取VAE的dtype，确保输入与VAE的dtype匹配
        vae_dtype = next(self.vae.parameters()).dtype
        processed_image = processed_image.to(device=device, dtype=vae_dtype)
        
        # 2. 先获取prompt_embeds以确定dtype，然后编码图像
        prompt_embeds, pooled_prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        
        # 使用VAE编码图像到latent space
        with torch.no_grad():
            # VAE编码：得到 [B, C, H, W] 格式的latent
            encoded_latents = self.vae.encode(processed_image).latent_dist.sample()
            # FLUX的VAE需要应用scaling_factor
            encoded_latents = encoded_latents * self.vae.config.scaling_factor
            # 确保latent在正确的device和dtype上
            encoded_latents = encoded_latents.to(device=device, dtype=prompt_embeds.dtype)
        
        # 3. 使用prepare_latents生成正确的格式（不传入latents，让它生成格式）
        num_channels_latents = self.transformer.config.in_channels // 4
        dummy_latents, latent_image_ids = self.prepare_latents(
            1,  # batch_size
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,  # 不传入latents，让它生成正确的pack格式
        )
        
        # 4. 将编码后的latent pack成与dummy_latents相同的格式
        # dummy_latents是[B, seq_len, C]格式，我们需要将encoded_latents也pack成这个格式
        B_dummy, seq_len, C_dummy = dummy_latents.shape
        B, C_vae, H_enc, W_enc = encoded_latents.shape
        
        # 问题：VAE输出是16通道，但transformer期望64通道
        # 需要将16通道扩展到64通道
        # 方法：使用简单的重复和线性组合
        if C_vae != C_dummy:
            # 计算扩展倍数
            expansion_factor = C_dummy // C_vae
            remainder = C_dummy % C_vae
            
            # 方法1：重复通道并添加一些变化
            # 将16通道重复4次得到64通道
            expanded_channels = []
            for i in range(expansion_factor):
                expanded_channels.append(encoded_latents)
            if remainder > 0:
                # 如果有余数，添加部分通道
                expanded_channels.append(encoded_latents[:, :remainder])
            
            # 拼接得到64通道
            encoded_latents = torch.cat(expanded_channels, dim=1)
            C_vae = C_dummy
        
        # 计算期望的patch尺寸：seq_len = H_patches * W_patches
        # 根据height和width的比例来推断
        aspect_ratio = width / height
        H_patches = int(np.sqrt(seq_len / aspect_ratio))
        W_patches = int(seq_len / H_patches)
        
        # 如果VAE输出的尺寸不匹配，需要resize
        if H_enc != H_patches or W_enc != W_patches:
            encoded_latents = torch.nn.functional.interpolate(
                encoded_latents, size=(H_patches, W_patches), mode='bilinear', align_corners=False
            )
        
        # Reshape: [B, C, H, W] -> [B, H*W, C] = [B, seq_len, C]
        latents = encoded_latents.permute(0, 2, 3, 1).reshape(B, H_patches * W_patches, C_vae)
        
        # 确保dtype和device匹配
        latents = latents.to(device=device, dtype=prompt_embeds.dtype)
        
        # 最终验证：确保形状完全匹配
        assert latents.shape == dummy_latents.shape, f"Shape mismatch: {latents.shape} vs {dummy_latents.shape}"
        
        # 3. 获取文本编码长度（如果未提供）
        if bg_len is None or real_len is None:
            # 尝试从prompt中解析（这里假设prompt格式为 "bg#fg#act"）
            # 如果格式不同，用户需要手动提供bg_len和real_len
            if "#" in prompt:
                parts = prompt.split("#")
                if len(parts) >= 2:
                    bg_part = parts[0]
                    fg_part = "#".join(parts[1:-1]) if len(parts) > 2 else parts[1]
                    bg_len = self._get_text_tokens_length(bg_part)
                    real_len = self._get_text_tokens_length(prompt)
                else:
                    # 如果无法解析，使用完整prompt
                    real_len = self._get_text_tokens_length(prompt)
                    bg_len = 0  # 默认值，可能需要用户手动设置
            else:
                real_len = self._get_text_tokens_length(prompt)
                bg_len = 0  # 默认值
        
        # 4. 执行一次forward pass，保存所有中间特征
        # 关键：设置is_id=True，让系统保存id_attn_bank
        id_images, id_spatial_kwargs = self(
            prompt=prompt,
            height=height,
            width=width,
            latents=latents,  # 使用编码后的latent
            is_id=True,  # 关键：保存所有中间特征
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            use_interpolate=use_interpolate,
            share_bg=share_bg,
            attn_start_step=attn_start_step,
            attn_end_step=attn_end_step,
            interpolate_start_step=interpolate_start_step,
            interpolate_decay_step=interpolate_decay_step,
            interpolate_end_step=interpolate_end_step,
            interpolate_weight=interpolate_weight,
            sim_thr=sim_thr,
            save_mask_point_step=save_mask_point_step,
            **kwargs
        )
        
        return id_images, id_spatial_kwargs
    
    def _get_text_tokens_length(self, text: str) -> int:
        """辅助方法：获取文本的token长度"""
        text_mask = self.tokenizer_2(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).attention_mask
        return text_mask.sum().item() - 1


