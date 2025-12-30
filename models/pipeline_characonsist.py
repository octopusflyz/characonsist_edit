# Copied and modified from diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import copy

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from .attention_processor_characonsist import get_curr_fg_mask, get_cross_sim, get_multi_object_fg_masks


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


def get_object_aware_point_matching(
    id_object_masks,  # List of masks for each object in ID image
    curr_object_masks,  # List of masks for each object in current image
    cross_sim,  # Cross similarity matrix [H*W, H*W]
    sim_thr=0.5
):
    """对象感知的点匹配，确保只在同一对象内匹配"""
    num_objects = len(id_object_masks)
    if num_objects != len(curr_object_masks):
        # 如果对象数量不匹配，回退到全局匹配
        max_sim, argmax_indices = torch.max(cross_sim, dim=-1)
        return argmax_indices, max_sim
    
    # 获取mask的形状
    if len(id_object_masks[0].shape) == 3:  # [B, H, W]
        h, w = id_object_masks[0].shape[-2:]
    else:  # [H, W]
        h, w = id_object_masks[0].shape
    total_pixels = h * w
    
    # 初始化全局结果
    all_argmax_indices = torch.zeros(total_pixels, dtype=torch.long, device=cross_sim.device)
    all_max_sim = torch.zeros(total_pixels, dtype=torch.float, device=cross_sim.device)
    
    for obj_idx in range(num_objects):
        # 处理mask维度
        id_mask = id_object_masks[obj_idx]
        curr_mask = curr_object_masks[obj_idx]
        if len(id_mask.shape) == 3:  # [B, H, W]
            id_mask = id_mask[0]
            curr_mask = curr_mask[0]
        
        id_mask_flat = id_mask.flatten()
        curr_mask_flat = curr_mask.flatten()
        
        # 只在当前对象的区域内计算相似度
        # 使用 as_tuple=True 然后取第一个元素，确保得到 1D 张量
        id_indices_tuple = torch.nonzero(id_mask_flat, as_tuple=True)
        curr_indices_tuple = torch.nonzero(curr_mask_flat, as_tuple=True)
        
        # 取第一个（也是唯一的）维度
        id_indices = id_indices_tuple[0] if len(id_indices_tuple) > 0 else torch.tensor([], dtype=torch.long, device=id_mask_flat.device)
        curr_indices = curr_indices_tuple[0] if len(curr_indices_tuple) > 0 else torch.tensor([], dtype=torch.long, device=curr_mask_flat.device)
        
        # 确保是 1D 张量
        id_indices = id_indices.flatten()
        curr_indices = curr_indices.flatten()
        
        if len(id_indices) == 0 or len(curr_indices) == 0:
            continue
        
        # 使用小批量逐行处理，平衡内存和速度
        # 每次只处理少量行（如10-50行），避免创建大型中间张量
        if len(curr_indices) > 0 and len(id_indices) > 0:
            # 使用很小的批次大小（10-50行），避免内存峰值
            # 这样既不会创建太大的中间张量，也不会太慢
            small_batch_size = 20  # 每次处理20行，内存占用约 20 * H*W * 4 bytes
            
            obj_max_sim_list = []
            obj_argmax_local_list = []
            curr_indices_list = []
            
            # 边界检查：确保所有索引都在有效范围内
            cross_sim_size = cross_sim.shape[0]
            cross_sim_cols = cross_sim.shape[1]
            curr_indices = torch.clamp(curr_indices, 0, cross_sim_size - 1)
            id_indices = torch.clamp(id_indices, 0, cross_sim_cols - 1)
            
            for i in range(0, len(curr_indices), small_batch_size):
                batch_curr_indices = curr_indices[i:i+small_batch_size]
                
                # 使用高级索引直接提取需要的行和列，避免创建完整的中间张量
                # cross_sim[batch_curr_indices, :][:, id_indices] 会创建 [batch_size, len(id_indices)]
                # 但我们可以使用更高效的方式：先提取行，再索引列
                expected_batch_size = batch_curr_indices.shape[0]
                
                # 确保 batch_curr_indices 是 1D 的
                batch_curr_indices = batch_curr_indices.flatten()
                if batch_curr_indices.shape[0] != expected_batch_size:
                    raise ValueError(f"batch_curr_indices shape mismatch: {batch_curr_indices.shape} != [{expected_batch_size}]")
                
                # 确保 id_indices 是 1D 的
                id_indices_flat = id_indices.flatten()
                
                # 验证 id_indices_flat 的形状
                if id_indices_flat.dim() != 1:
                    raise ValueError(f"id_indices_flat should be 1D, got {id_indices_flat.dim()}D with shape {id_indices_flat.shape}")
                
                batch_rows = cross_sim[batch_curr_indices]  # [batch_size, H*W]
                
                # 验证 batch_rows 的形状
                if batch_rows.shape[0] != expected_batch_size:
                    raise ValueError(f"batch_rows.shape[0]={batch_rows.shape[0]} != expected_batch_size={expected_batch_size}")
                if batch_rows.dim() != 2:
                    raise ValueError(f"batch_rows should be 2D, got {batch_rows.dim()}D with shape {batch_rows.shape}")
                
                # 索引列：使用 index_select 确保结果是 2D [batch_size, len(id_indices)]
                # 这样可以避免高级索引可能产生的意外广播
                batch_obj_cross_sim = torch.index_select(batch_rows, dim=1, index=id_indices_flat)  # [batch_size, len(id_indices)]
                
                # 验证形状
                if batch_obj_cross_sim.shape[0] != expected_batch_size:
                    raise ValueError(f"batch_obj_cross_sim.shape[0]={batch_obj_cross_sim.shape[0]} != expected_batch_size={expected_batch_size}")
                if batch_obj_cross_sim.dim() != 2:
                    raise ValueError(f"batch_obj_cross_sim should be 2D, got {batch_obj_cross_sim.dim()}D with shape {batch_obj_cross_sim.shape}")
                
                # 沿着最后一个维度求最大值
                batch_obj_max_sim, batch_obj_argmax_local = torch.max(batch_obj_cross_sim, dim=-1)
                
                # 验证返回值的形状：应该是 [batch_size]
                if batch_obj_max_sim.dim() != 1 or batch_obj_max_sim.shape[0] != expected_batch_size:
                    raise ValueError(f"After max: batch_obj_max_sim.shape={batch_obj_max_sim.shape}, expected [{expected_batch_size}]")
                if batch_obj_argmax_local.dim() != 1 or batch_obj_argmax_local.shape[0] != expected_batch_size:
                    raise ValueError(f"After max: batch_obj_argmax_local.shape={batch_obj_argmax_local.shape}, expected [{expected_batch_size}]")
                
                # 确保是1D的（虽然应该已经是1D的了）
                batch_obj_max_sim = batch_obj_max_sim.flatten()
                batch_obj_argmax_local = batch_obj_argmax_local.flatten()
                batch_curr_indices = batch_curr_indices.flatten()
                
                # 最终验证：所有张量的长度应该相同
                if batch_obj_max_sim.shape[0] != expected_batch_size or batch_obj_argmax_local.shape[0] != expected_batch_size or batch_curr_indices.shape[0] != expected_batch_size:
                    raise ValueError(f"Final shape check failed: batch_obj_max_sim.shape={batch_obj_max_sim.shape}, batch_obj_argmax_local.shape={batch_obj_argmax_local.shape}, batch_curr_indices.shape={batch_curr_indices.shape}, expected_batch_size={expected_batch_size}")
                
                obj_max_sim_list.append(batch_obj_max_sim)
                obj_argmax_local_list.append(batch_obj_argmax_local)
                curr_indices_list.append(batch_curr_indices)
                
                # 立即释放中间张量
                del batch_rows, batch_obj_cross_sim
                # 每处理几批就清理一次缓存
                if (i // small_batch_size) % 10 == 0 and cross_sim.is_cuda:
                    torch.cuda.empty_cache()
            
            # 合并结果，确保所有张量都是1D的，并且形状正确
            # 首先检查每个列表中的元素形状
            for i, (msim, argmax, cidx) in enumerate(zip(obj_max_sim_list, obj_argmax_local_list, curr_indices_list)):
                if msim.shape[0] != argmax.shape[0] or msim.shape[0] != cidx.shape[0]:
                    raise ValueError(f"Batch {i} shape mismatch: msim.shape={msim.shape}, argmax.shape={argmax.shape}, cidx.shape={cidx.shape}")
            
            obj_max_sim = torch.cat([x.flatten() for x in obj_max_sim_list], dim=0)
            obj_argmax_local = torch.cat([x.flatten() for x in obj_argmax_local_list], dim=0)
            all_curr_indices = torch.cat([x.flatten() for x in curr_indices_list], dim=0)
            
            # 验证形状：obj_argmax_local 应该和 all_curr_indices 长度相同
            if obj_argmax_local.shape[0] != all_curr_indices.shape[0]:
                raise ValueError(f"Shape mismatch after cat: obj_argmax_local.shape={obj_argmax_local.shape}, all_curr_indices.shape={all_curr_indices.shape}, expected same length")
            
            # 确保所有张量都是1D的
            obj_max_sim = obj_max_sim.flatten()
            obj_argmax_local = obj_argmax_local.flatten()
            all_curr_indices = all_curr_indices.flatten()
            
            # 边界检查：确保 obj_argmax_local 不会越界
            if len(id_indices) > 0:
                # 确保 id_indices 是 1D 的
                id_indices_1d = id_indices.flatten() if id_indices.dim() > 1 else id_indices
                if id_indices_1d.dim() > 1:
                    id_indices_1d = id_indices_1d.squeeze()
                
                # 验证 obj_argmax_local 的长度应该等于 all_curr_indices 的长度
                if obj_argmax_local.shape[0] != all_curr_indices.shape[0]:
                    raise ValueError(f"Before indexing: obj_argmax_local.shape={obj_argmax_local.shape}, all_curr_indices.shape={all_curr_indices.shape}, id_indices_1d.shape={id_indices_1d.shape}")
                
                max_idx = len(id_indices_1d) - 1
                if max_idx < 0:
                    global_argmax = torch.zeros_like(all_curr_indices)
                else:
                    # 确保 obj_argmax_local 在有效范围内
                    obj_argmax_local = torch.clamp(obj_argmax_local, 0, max_idx)
                    # 创建全局索引映射
                    global_argmax = id_indices_1d[obj_argmax_local]
                    # 确保 global_argmax 是 1D 的
                    global_argmax = global_argmax.flatten()
                    
                    # 最终验证：global_argmax 应该和 all_curr_indices 长度相同
                    if global_argmax.shape[0] != all_curr_indices.shape[0]:
                        raise ValueError(f"After indexing: global_argmax.shape={global_argmax.shape}, all_curr_indices.shape={all_curr_indices.shape}, obj_argmax_local.shape={obj_argmax_local.shape}, id_indices_1d.shape={id_indices_1d.shape}")
            else:
                # 如果没有有效的 id_indices，使用 0 作为默认值
                global_argmax = torch.zeros_like(all_curr_indices)
            
            # 边界检查：确保 all_curr_indices 不会越界
            all_curr_indices = torch.clamp(all_curr_indices, 0, total_pixels - 1)
            
            # 更新结果
            all_argmax_indices[all_curr_indices] = global_argmax
            all_max_sim[all_curr_indices] = obj_max_sim
            
            # 清理
            del obj_max_sim_list, obj_argmax_local_list, curr_indices_list
            del obj_max_sim, obj_argmax_local, all_curr_indices, global_argmax
            if cross_sim.is_cuda:
                torch.cuda.empty_cache()
    
    # 处理未匹配的区域（不在任何对象mask内）
    curr_mask_stack = torch.stack([m.flatten() if len(m.shape) == 2 else m[0].flatten() for m in curr_object_masks], dim=0)
    unmatched_mask = ~torch.any(curr_mask_stack, dim=0)
    if unmatched_mask.any():
        # 对于未匹配区域，使用全局匹配
        unmatched_indices_tuple = torch.nonzero(unmatched_mask, as_tuple=True)
        unmatched_indices = unmatched_indices_tuple[0] if len(unmatched_indices_tuple) > 0 else torch.tensor([], dtype=torch.long, device=unmatched_mask.device)
        unmatched_indices = unmatched_indices.flatten()
        if len(unmatched_indices) > 0:
            # 使用小批量处理，与上面保持一致
            small_batch_size = 20
            unmatched_max_sim_list = []
            unmatched_argmax_list = []
            unmatched_indices_list = []
            
            for i in range(0, len(unmatched_indices), small_batch_size):
                batch_indices = unmatched_indices[i:i+small_batch_size]
                # 边界检查：确保 batch_indices 不会超出 cross_sim 的范围
                cross_sim_size = cross_sim.shape[0]
                batch_indices = torch.clamp(batch_indices, 0, cross_sim_size - 1)
                batch_rows = cross_sim[batch_indices]  # [batch_size, H*W]
                batch_max_sim, batch_argmax = torch.max(batch_rows, dim=-1)
                # 边界检查：确保 batch_argmax 不会超出范围
                batch_argmax = torch.clamp(batch_argmax, 0, cross_sim.shape[1] - 1)
                unmatched_max_sim_list.append(batch_max_sim)
                unmatched_argmax_list.append(batch_argmax)
                unmatched_indices_list.append(batch_indices)
                del batch_rows
                if (i // small_batch_size) % 10 == 0 and cross_sim.is_cuda:
                    torch.cuda.empty_cache()
            
            unmatched_max_sim = torch.cat(unmatched_max_sim_list, dim=0)
            unmatched_argmax = torch.cat(unmatched_argmax_list, dim=0)
            all_unmatched_indices = torch.cat(unmatched_indices_list, dim=0)
            
            # 边界检查：确保 unmatched_argmax 和 all_unmatched_indices 不会越界
            unmatched_argmax = torch.clamp(unmatched_argmax, 0, total_pixels - 1)
            all_unmatched_indices = torch.clamp(all_unmatched_indices, 0, total_pixels - 1)
            
            all_argmax_indices[all_unmatched_indices] = unmatched_argmax
            all_max_sim[all_unmatched_indices] = unmatched_max_sim
            
            del unmatched_max_sim_list, unmatched_argmax_list, unmatched_indices_list
            del unmatched_max_sim, unmatched_argmax, all_unmatched_indices
            if cross_sim.is_cuda:
                torch.cuda.empty_cache()
    
    # 重塑回原始形状
    all_argmax_indices = all_argmax_indices.reshape(h, w)
    all_max_sim = all_max_sim.reshape(h, w)
    
    return all_argmax_indices, all_max_sim


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
                    # 检查是否是多对象模式
                    num_objects = 1
                    for name in self.transformer.attn_processors:
                        processor = self.transformer.attn_processors[name]
                        if hasattr(processor, 'num_objects') and processor.num_objects > 1:
                            num_objects = processor.num_objects
                            break
                    
                    if num_objects > 1:
                        # 多对象模式：获取每个对象的独立mask
                        curr_object_masks = get_multi_object_fg_masks(self)
                        # 合并所有对象的mask作为总的fg_mask
                        curr_fg_mask = curr_object_masks[0]
                        for obj_mask in curr_object_masks[1:]:
                            curr_fg_mask = curr_fg_mask | obj_mask
                        # 确保curr_fg_mask是[1, H, W]形状，与单对象模式保持一致
                        if len(curr_fg_mask.shape) == 2:
                            curr_fg_mask = curr_fg_mask.unsqueeze(0)
                        spatial_kwargs["curr_fg_mask"] = curr_fg_mask
                        spatial_kwargs["curr_object_masks"] = curr_object_masks
                        
                        # 如果是 ID 图像，也保存 id_object_masks
                        if is_id:
                            spatial_kwargs["id_object_masks"] = curr_object_masks
                    else:
                        # 单对象模式，保持原有逻辑
                        curr_fg_mask = get_curr_fg_mask(self)
                        spatial_kwargs["curr_fg_mask"] = curr_fg_mask
                    
                    if update_bg:
                        spatial_kwargs["id_bg_mask"] = copy.deepcopy(~curr_fg_mask)
                        
                if self.joint_attention_kwargs["save_cross_sim"]:
                    avg_cross_sim = get_cross_sim(self)
                    
                    # 检查是否是多对象模式
                    num_objects = 1
                    id_object_masks = None
                    curr_object_masks = None
                    
                    if "id_object_masks" in spatial_kwargs:
                        id_object_masks = spatial_kwargs["id_object_masks"]
                        num_objects = len(id_object_masks)
                    if "curr_object_masks" in spatial_kwargs:
                        curr_object_masks = spatial_kwargs["curr_object_masks"]
                        if num_objects == 1:
                            num_objects = len(curr_object_masks)
                    
                    # 使用对象约束的点匹配
                    if num_objects > 1 and id_object_masks is not None and curr_object_masks is not None:
                        argmax_indices, max_sim = get_object_aware_point_matching(
                            id_object_masks, curr_object_masks, avg_cross_sim, sim_thr
                        )
                    else:
                        # 单对象模式，使用原有逻辑
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


