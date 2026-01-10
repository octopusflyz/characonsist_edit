# Copied and modified from diffusers/blob/main/src/diffusers/models/attention_processor.py

import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gc
from typing import Optional
from einops import rearrange
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from tqdm import tqdm


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # new args
        timestep_ind: Optional[int] = None,
        save_attn_weight: bool = False,
        save_attn_kv: bool = False,
        update_attn_kv: bool = False,
        save_attn_out_for_sim: bool = False,
        save_attn_out_for_interpolate: bool = False,
        save_cross_sim: bool = False,
        fg_inter_img_attn: bool = False,
        bg_inter_img_attn: bool = False,
        attn_out_interpolate: bool = False,
        interpolate_weight_dict: dict = dict(),
        spatial_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class CharaConsistAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        size=(64, 64),
        text_seq_len=512,
        # Attention Share
        fg_share_flag=False,
        bg_share_flag=False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.text_seq_len = text_seq_len
        self.visual_seq_len = size[0] * size[1]
        # Hyper Params
        self.fg_share_flag = fg_share_flag
        self.bg_share_flag = bg_share_flag
        # Sample Info
        self.bg_len = 0
        self.real_len = 0
        self.size = size
        # Multi-object support
        self.num_objects = 1
        self.object_token_ranges = None
        self.fg_lengths = None
        # Save id info
        self.id_attn_bank = dict()
        self.attn_weights = dict()
        self.cross_sims = None
            
    def get_curr_attn_weights(self, q, k):
        attn_weights = torch.matmul(q[:, :, self.text_seq_len:], k[:, :, :self.real_len].transpose(-2, -1)).to(torch.float16)
        attn_weights = attn_weights.softmax(-1).sum(1)
        attn_weights = rearrange(attn_weights, "b (h w) s -> b s h w", h=self.size[0], w=self.size[1])
        bg_attn_weights = attn_weights[:, :self.bg_len].mean(1)

        # 支持多对象：为每个对象单独计算fg注意力权重
        if self.num_objects > 1 and self.object_token_ranges is not None:
            object_fg_attn_weights = []
            for start_token, end_token in self.object_token_ranges:
                obj_fg_attn = attn_weights[:, start_token:end_token].mean(1)
                object_fg_attn_weights.append(obj_fg_attn)
            # 为了向后兼容，仍然返回平均的fg_attn_weights
            fg_attn_weights = sum(object_fg_attn_weights) / len(object_fg_attn_weights)
        else:
            # 单对象情况，保持原有逻辑
            fg_attn_weights = attn_weights[:, self.bg_len:].mean(1)

        return bg_attn_weights, fg_attn_weights

    def get_curr_attn_weights_multi_fg(self, q, k, fg_lengths):
        """计算多个前景主体的注意力权重"""
        attn_weights = torch.matmul(q[:, :, self.text_seq_len:], k[:, :, :self.real_len].transpose(-2, -1)).to(torch.float16)
        attn_weights = attn_weights.softmax(-1).sum(1)
        attn_weights = rearrange(attn_weights, "b (h w) s -> b s h w", h=self.size[0], w=self.size[1])

        bg_attn_weights = attn_weights[:, :self.bg_len].mean(1)

        fg_attn_weights = []
        start_idx = self.bg_len
        for fg_len in fg_lengths:
            fg_attn_weights.append(attn_weights[:, start_idx:start_idx+fg_len].mean(1))
            start_idx += fg_len

        return bg_attn_weights, fg_attn_weights

    def get_curr_cross_sim(self, curr_hidden_states, timestep_ind):
        id_hidden_states = self.id_attn_bank[timestep_ind]["attn_out"].to(curr_hidden_states.device, non_blocking=True)
        sim = torch.matmul(
            F.normalize(curr_hidden_states, dim=-1),
            F.normalize(id_hidden_states, dim=-1).transpose(-2, -1))
        return sim
    
    def get_expand_attn_mask(self, id_fg_mask, curr_fg_mask, bg_share_flag, fg_share_flag, device):
        id_len = len(id_fg_mask)
        _id_fg_mask = id_fg_mask.view(1, 1, 1, -1)
        _curr_fg_mask = curr_fg_mask.view(1, 1, -1, 1)
        expand_mask = _id_fg_mask != _curr_fg_mask
        if not bg_share_flag:
            expand_mask[:, :, :, ~id_fg_mask] = True
        elif not fg_share_flag:
            expand_mask[:, :, :, id_fg_mask] = True
        t2i_expand_mask = torch.ones((1, 1, self.text_seq_len, id_len), device=device, dtype=bool)
        return torch.cat((t2i_expand_mask, expand_mask), dim=-2)
    
    def get_expanded_key_value(
            self,
            image_rotary_emb,
            bg_share_flag,
            fg_share_flag,
            timestep_ind,
            device, 
            id_fg_mask=None,
            id_bg_mask=None,
            curr_fg_mask=None,
            id_fg_inds=None,
            id_bg_inds=None,
            curr_fg_inds=None,
            **kwargs):
        
        saved_key, saved_value = None, None
        id_fg_mask = id_fg_mask.flatten().to(device, non_blocking=True)
        curr_fg_mask = curr_fg_mask.flatten().to(device, non_blocking=True)

        ori_saved_key = self.id_attn_bank[timestep_ind]["key"].to(device, non_blocking=True)
        ori_saved_value = self.id_attn_bank[timestep_ind]["value"].to(device, non_blocking=True)

        if bg_share_flag:
            assert id_bg_mask is not None
            id_bg_mask = id_bg_mask.flatten().to(device, non_blocking=True)
            shared_bg = (id_bg_mask & (~curr_fg_mask))
            if "updated_key" in self.id_attn_bank[timestep_ind]:
                saved_key = self.id_attn_bank[timestep_ind]["updated_key"].to(device, non_blocking=True)[:, :, shared_bg]
                saved_value = self.id_attn_bank[timestep_ind]["updated_value"].to(device, non_blocking=True)[:, :, shared_bg]
            else:
                saved_key = ori_saved_key[:, :, shared_bg]
                saved_value = ori_saved_value[:, :, shared_bg]
            image_rotary_emb_bg = (
                image_rotary_emb[0][shared_bg],
                image_rotary_emb[1][shared_bg],
            )
            saved_key = apply_rotary_emb(saved_key, image_rotary_emb_bg)
            id_fg_mask = torch.zeros([saved_key.shape[2]], device=device, dtype=torch.bool)
        
        if fg_share_flag:
            saved_key_fg = ori_saved_key[:, :, id_fg_inds]
            saved_value_fg = ori_saved_value[:, :, id_fg_inds]
            image_rotary_emb_fg = (
                image_rotary_emb[0][curr_fg_inds],
                image_rotary_emb[1][curr_fg_inds],
            )
            saved_key_fg = apply_rotary_emb(saved_key_fg, image_rotary_emb_fg)
            if saved_key is not None:
                saved_key = torch.cat([saved_key, saved_key_fg], dim=2)
                saved_value = torch.cat([saved_value, saved_value_fg], dim=2)
                id_fg_mask = torch.zeros([saved_key.shape[2]], device=device, dtype=torch.bool)
                id_fg_mask[-saved_key_fg.shape[2]:] = True
            else:
                saved_key = saved_key_fg
                saved_value = saved_value_fg
                id_fg_mask = torch.ones([saved_key.shape[2]], device=device, dtype=torch.bool)
        
        expand_mask= self.get_expand_attn_mask(
            id_fg_mask, curr_fg_mask, bg_share_flag, fg_share_flag, device=device)
        attention_mask = torch.zeros(
            (1, 1, self.text_seq_len + self.visual_seq_len, self.text_seq_len + self.visual_seq_len + len(id_fg_mask)), 
            device=device, dtype=torch.bfloat16)
        attention_mask[:, :, :, self.text_seq_len + self.visual_seq_len:].masked_fill_(expand_mask, float('-inf'))
        return saved_key, saved_value, attention_mask
    
    
    def ada_tome(self, hidden_states, timestep_ind, alpha, id_fg_inds=None, curr_fg_inds=None, max_sim=None, argmax_indices=None, **kwargs):
        id_hidden_states = self.id_attn_bank[timestep_ind]["attn_out"].to(hidden_states.device, non_blocking=True)
        vision_hidden_states = hidden_states[:, self.text_seq_len:, :]

        if argmax_indices is not None and curr_fg_inds is not None:
            # 使用语义匹配：curr_fg_inds中的像素对应argmax_indices中指定的ID图像位置
            matched_id_positions = argmax_indices.flatten()[curr_fg_inds]  # 获取语义对应的ID图像位置
            matched_id_hidden_states = id_hidden_states[:, matched_id_positions]
        else:
            # 降级到空间匹配（向后兼容）
            matched_id_hidden_states = id_hidden_states[:, id_fg_inds] if id_fg_inds is not None else id_hidden_states[:, curr_fg_inds]

        matched_curr_hidden_states = vision_hidden_states[:, curr_fg_inds]

        # 使用与输入张量相同的dtype，避免类型不匹配
        input_dtype = vision_hidden_states.dtype
        alpha_tensor = torch.ones_like(curr_fg_inds, dtype=input_dtype, device=curr_fg_inds.device)
        alpha_tensor = alpha_tensor * alpha

        if max_sim is not None:
            sim_weight = max_sim.flatten()[curr_fg_inds]
            # 使用sigmoid限制相似度权重在合理范围内
            sim_weight = torch.sigmoid(sim_weight).to(input_dtype)
            alpha_tensor = alpha_tensor * sim_weight

        alpha_tensor = alpha_tensor.view(1, -1, 1)

        new_matched_curr_hidden_states = (1 - alpha_tensor) * matched_curr_hidden_states + alpha_tensor * matched_id_hidden_states
        vision_hidden_states[:, curr_fg_inds] = new_matched_curr_hidden_states
        out_hidden_states = torch.cat((hidden_states[:, :self.text_seq_len, :], vision_hidden_states), dim=-2)
        return out_hidden_states


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # new args
        timestep_ind: Optional[int] = None,
        save_attn_weight: bool = False,
        save_attn_kv: bool = False,
        update_attn_kv: bool = False,
        save_attn_out_for_sim: bool = False,
        save_attn_out_for_interpolate: bool = False,
        save_cross_sim: bool = False,
        fg_inter_img_attn: bool = False,
        bg_inter_img_attn: bool = False,
        attn_out_interpolate: bool = False,
        interpolate_weight_dict: dict = dict(),
        spatial_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        if save_attn_kv:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["key"] = key[:, :, self.text_seq_len:, :].cpu()
            self.id_attn_bank[timestep_ind]["value"] = value[:, :, self.text_seq_len:, :].cpu()
        
        if update_attn_kv:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["updated_key"] = key[:, :, self.text_seq_len:, :].cpu()
            self.id_attn_bank[timestep_ind]["updated_value"] = value[:, :, self.text_seq_len:, :].cpu()

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if save_attn_weight:
            bg_attn, fg_attn = self.get_curr_attn_weights(query, key)
            # 支持多对象：保存每个对象的注意力权重
            if self.num_objects > 1 and self.object_token_ranges is not None:
                attn_weights_temp = torch.matmul(query[:, :, self.text_seq_len:], key[:, :, :self.real_len].transpose(-2, -1)).to(torch.float16)
                attn_weights_temp = attn_weights_temp.softmax(-1).sum(1)
                attn_weights_temp = rearrange(attn_weights_temp, "b (h w) s -> b s h w", h=self.size[0], w=self.size[1])
                object_fg_attns = []
                for start_token, end_token in self.object_token_ranges:
                    obj_fg_attn = attn_weights_temp[:, start_token:end_token].mean(1)
                    object_fg_attns.append(obj_fg_attn.to("cuda:0", non_blocking=True))
                self.attn_weights = dict(bg=bg_attn.to("cuda:0", non_blocking=True), objects=object_fg_attns)
            else:
                # 单对象情况，保持原有格式
                self.attn_weights = dict(bg=bg_attn.to("cuda:0", non_blocking=True), fg=fg_attn.to("cuda:0", non_blocking=True))
        
        fg_share_flag = self.fg_share_flag and fg_inter_img_attn
        bg_share_flag = self.bg_share_flag and bg_inter_img_attn and (not update_attn_kv)

        # 在多对象模式下，暂时跳过扩展key-value逻辑，因为它是为单对象设计的
        if (fg_share_flag or bg_share_flag) and self.num_objects <= 1:
            saved_key, saved_value, attention_mask = self.get_expanded_key_value(
                (image_rotary_emb[0][self.text_seq_len:], image_rotary_emb[1][self.text_seq_len:]),
                bg_share_flag,
                fg_share_flag,
                timestep_ind,
                query.device,
                **spatial_kwargs)
            key = torch.cat([key, saved_key], dim=2)
            value = torch.cat([value, saved_value], dim=2)
            attention_mask = attention_mask
        else:
            attention_mask = None

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if save_attn_out_for_sim:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["attn_out"] = hidden_states[:, self.text_seq_len:, :].cpu()
        elif save_attn_out_for_interpolate and self.fg_share_flag:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["attn_out"] = hidden_states[:, self.text_seq_len:, :].cpu()

        if save_cross_sim:
            self.cross_sims = self.get_curr_cross_sim(hidden_states[:, self.text_seq_len:, :], timestep_ind).to("cuda:0", non_blocking=True)

        if attn_out_interpolate:
            if fg_share_flag:
                hidden_states = self.ada_tome(hidden_states, timestep_ind, interpolate_weight_dict[timestep_ind], **spatial_kwargs)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def reset_attn_processor(pipe, size, fg_share_freq=2, bg_share_freq=1):
    new_attn_processors = dict()
    transformer = pipe.transformer
    ori_attn_processors = transformer.attn_processors
    reset_num = 0
    for name in ori_attn_processors:
        if 'single' in name:
            block_ind = int(name.split(".")[1])
            new_attn_processors[name] = CharaConsistAttnProcessor2_0(
                size=size,
                fg_share_flag=(block_ind % fg_share_freq == 0),
                bg_share_flag=(block_ind % bg_share_freq == 0),
            )
            print(f"reset attn processor of layer {name}")
            reset_num += 1
        else:
            new_attn_processors[name] = FluxAttnProcessor2_0()
    transformer.set_attn_processor(new_attn_processors)
    print(f"{reset_num} layers have been reset")


def set_text_len(pipe, bg_len, real_len, num_objects=None, object_token_ranges=None, fg_lengths=None):
    """设置文本长度，支持多对象

    Args:
        pipe: pipeline对象
        bg_len: 背景token长度
        real_len: 总token长度
        num_objects: 对象数量（可选）
        object_token_ranges: 每个对象对应的token范围列表（可选）
        fg_lengths: 每个前景对象的token长度列表（可选）
    """
    attn_processors = pipe.transformer.attn_processors
    reset_num = 0
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.bg_len = bg_len
            processor.real_len = real_len
            if num_objects is not None:
                processor.num_objects = num_objects
            if object_token_ranges is not None:
                processor.object_token_ranges = object_token_ranges
            if fg_lengths is not None:
                processor.fg_lengths = fg_lengths
            reset_num += 1
    print(f"{reset_num} layers' background and real text length have been reset to {bg_len} and {real_len}.")
    if num_objects and num_objects > 1:
        print(f"Multi-object mode: {num_objects} objects detected.")

def remove_small_holes_and_points(mask_tensor):
    n, h, w = mask_tensor.shape
    results = []
    for i in range(n):
        mask_np = mask_tensor[i].cpu().numpy().astype(np.uint8)

        # 自适应内核大小：基于图像尺寸的2%，最小3
        kernel_size = max(3, int(min(h, w) * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 开操作：去除小噪声同时保持主要结构
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=1)

        # 闭操作：填充小洞同时保持边界
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 连通组件分析：保留面积最大的连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
        if num_labels > 1:  # 如果有多个连通区域
            # 保留面积最大的组件（跳过背景标签0）
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask_np = (labels == largest_label).astype(np.uint8)

        results.append(torch.tensor(mask_np))

    results = torch.stack(results, dim=0).to(mask_tensor.device, dtype=mask_tensor.dtype)
    return results

def get_curr_fg_mask(pipe):
    """获取前景mask，支持多对象"""
    attn_processors = pipe.transformer.attn_processors
    all_attn_weights = dict(bg=[], fg=[], objects=[])
    num_objects = 1

    # 检查是否是多对象模式
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            if hasattr(processor, 'num_objects') and processor.num_objects > 1:
                num_objects = processor.num_objects
            break

    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            saved_attns = processor.attn_weights
            for k in saved_attns:
                if k not in all_attn_weights:
                    all_attn_weights[k] = []
                all_attn_weights[k].append(saved_attns[k])
            processor.attn_weights = dict()

    bg_attns = sum(all_attn_weights["bg"]) / len(all_attn_weights["bg"])

    # 多对象模式：合并所有对象的mask
    if num_objects > 1 and "objects" in all_attn_weights and len(all_attn_weights["objects"]) > 0:
        # 获取第一个processor的object_token_ranges
        object_token_ranges = None
        for name in attn_processors:
            processor = attn_processors[name]
            if isinstance(processor, CharaConsistAttnProcessor2_0) and hasattr(processor, 'object_token_ranges'):
                object_token_ranges = processor.object_token_ranges
                break

        # 为每个对象计算mask，然后合并（使用与get_multi_object_fg_masks相同的逻辑）
        # 收集所有对象的attention权重
        # all_attn_weights["objects"] 是一个列表，每个元素是一个processor的所有对象权重
        # 我们需要对每个对象位置聚合所有processor的权重
        all_obj_attns = []
        for obj_idx in range(num_objects):
            # 从每个processor中获取第obj_idx个对象的权重
            obj_attn_list = []
            for processor_weights in all_attn_weights["objects"]:
                if len(processor_weights) > obj_idx:
                    obj_attn_list.append(processor_weights[obj_idx])

            if len(obj_attn_list) > 0:
                obj_attn = sum(obj_attn_list) / len(obj_attn_list)
                # 确保obj_attn是2维的 [H, W]
                if len(obj_attn.shape) == 3:
                    obj_attn = obj_attn.squeeze(0)
                elif len(obj_attn.shape) != 2:
                    obj_attn = obj_attn.view(-1, obj_attn.shape[-2], obj_attn.shape[-1])[0]
                all_obj_attns.append(obj_attn)
            else:
                all_obj_attns.append(torch.zeros_like(bg_attns))

        # 使用对象间的相对比较来提取mask，然后合并
        if len(all_obj_attns) > 0:
            stacked_obj_attns = torch.stack(all_obj_attns, dim=0)  # [num_objects, H, W]
            max_obj_attn, max_obj_idx = torch.max(stacked_obj_attns, dim=0)  # [H, W]

            # 为每个对象生成mask并合并
            combined_mask = None
            for obj_idx in range(num_objects):
                is_max_obj = (max_obj_idx == obj_idx)
                obj_gt_bg = (all_obj_attns[obj_idx] > bg_attns)
                obj_mask = is_max_obj & obj_gt_bg
                # 确保obj_mask是2维的 [H, W]，然后添加batch维度
                if len(obj_mask.shape) == 3:
                    obj_mask = obj_mask.squeeze(0)  # 如果是 [1, H, W]，去掉第一个维度
                elif len(obj_mask.shape) != 2:
                    # 如果不是2维，尝试reshape
                    obj_mask = obj_mask.view(-1, obj_mask.shape[-2], obj_mask.shape[-1])[0]
                obj_mask = remove_small_holes_and_points(obj_mask.unsqueeze(0))[0]

                if combined_mask is None:
                    combined_mask = obj_mask
                else:
                    combined_mask = combined_mask | obj_mask

            if combined_mask is not None:
                return combined_mask
        else:
            # 回退到单对象模式
            fg_attns = sum(all_attn_weights["fg"]) / len(all_attn_weights["fg"])
            fg_confidence = fg_attns / (bg_attns + fg_attns + 1e-6)
            fg_confidence_float = fg_confidence.float()
            threshold = torch.quantile(fg_confidence_float, 0.55)
            mask = fg_confidence > threshold
            return remove_small_holes_and_points(mask)
    else:
        # 单对象模式，保持原有逻辑
        fg_attns = sum(all_attn_weights["fg"]) / len(all_attn_weights["fg"])
        fg_confidence = fg_attns / (bg_attns + fg_attns + 1e-6)
        fg_confidence_float = fg_confidence.float()
        threshold = torch.quantile(fg_confidence_float, 0.55)
        mask = fg_confidence > threshold
        return remove_small_holes_and_points(mask)

def get_multi_object_fg_masks(pipe):
    """获取每个对象的独立fg mask（多对象模式）"""
    attn_processors = pipe.transformer.attn_processors
    all_attn_weights = dict(bg=[], objects=[])
    num_objects = 1
    object_token_ranges = None

    # 获取对象信息
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            if hasattr(processor, 'num_objects') and processor.num_objects > 1:
                num_objects = processor.num_objects
                object_token_ranges = getattr(processor, 'object_token_ranges', None)
            break

    print(f"Debug get_multi_object_fg_masks: num_objects={num_objects}, object_token_ranges={object_token_ranges}")

    if num_objects <= 1:
        # 单对象模式，返回单个mask
        return [get_curr_fg_mask(pipe)]

    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            saved_attns = processor.attn_weights
            for k in saved_attns:
                if k not in all_attn_weights:
                    all_attn_weights[k] = []
                all_attn_weights[k].append(saved_attns[k])
            processor.attn_weights = dict()

    print(f"Debug: all_attn_weights keys: {list(all_attn_weights.keys())}")
    if "objects" in all_attn_weights:
        print(f"Debug: objects list length: {len(all_attn_weights['objects'])}")
        if len(all_attn_weights['objects']) > 0:
            print(f"Debug: first processor objects: {len(all_attn_weights['objects'][0])}")

    bg_attns = sum(all_attn_weights["bg"]) / len(all_attn_weights["bg"])
    # 确保bg_attns是2维的 [H, W]
    if len(bg_attns.shape) == 3:
        bg_attns = bg_attns.squeeze(0)  # 如果是 [1, H, W]，去掉第一个维度
    elif len(bg_attns.shape) != 2:
        bg_attns = bg_attns.view(-1, bg_attns.shape[-2], bg_attns.shape[-1])[0]

    # 两阶段多对象mask生成
    object_masks = []

    # 第一阶段：确定整体前景区域（所有对象attention vs 背景）
    # 计算所有对象的attention之和
    if "objects" in all_attn_weights and len(all_attn_weights["objects"]) > 0:
        combined_obj_attn = None
        for obj_idx in range(num_objects):
            # 从每个processor中获取第obj_idx个对象的权重
            obj_attn_list = []
            for processor_weights in all_attn_weights["objects"]:
                if len(processor_weights) > obj_idx:
                    obj_attn_list.append(processor_weights[obj_idx])

            if len(obj_attn_list) > 0:
                obj_attn = sum(obj_attn_list) / len(obj_attn_list)
                # 确保obj_attn是2维的 [H, W]
                if len(obj_attn.shape) == 3:
                    obj_attn = obj_attn.squeeze(0)
                elif len(obj_attn.shape) != 2:
                    obj_attn = obj_attn.view(-1, obj_attn.shape[-2], obj_attn.shape[-1])[0]

                if combined_obj_attn is None:
                    combined_obj_attn = obj_attn
                else:
                    combined_obj_attn = combined_obj_attn + obj_attn

            # 使用置信度方法确定整体前景区域
            if combined_obj_attn is not None:
                fg_confidence = combined_obj_attn / (bg_attns + combined_obj_attn + 1e-6)
                fg_confidence_float = fg_confidence.float()
                overall_threshold = torch.quantile(fg_confidence_float, 0.55)
                overall_fg_mask = fg_confidence > overall_threshold
                # 确保mask有正确的维度 [1, H, W] 用于remove_small_holes_and_points
                if len(overall_fg_mask.shape) == 2:
                    overall_fg_mask = overall_fg_mask.unsqueeze(0)
                overall_fg_mask = remove_small_holes_and_points(overall_fg_mask)
                # 移除batch维度，返回 [H, W]
                overall_fg_mask = overall_fg_mask.squeeze(0)

            # 第二阶段：在前景区域内，竞争性地分配给不同对象
            # 收集所有对象的attention权重
            all_obj_attns = []
            for obj_idx in range(num_objects):
                obj_attn_list = [obj_attns[obj_idx] for obj_attns in all_attn_weights["objects"] if len(obj_attns) > obj_idx]
                if len(obj_attn_list) > 0:
                    obj_attn = sum(obj_attn_list) / len(obj_attn_list)
                    # 确保obj_attn是2维的 [H, W]
                    if len(obj_attn.shape) == 3:
                        obj_attn = obj_attn.squeeze(0)
                    elif len(obj_attn.shape) != 2:
                        obj_attn = obj_attn.view(-1, obj_attn.shape[-2], obj_attn.shape[-1])[0]
                    all_obj_attns.append(obj_attn)
                else:
                    # 如果没有找到该对象的注意力权重，创建一个零tensor
                    all_obj_attns.append(torch.zeros_like(bg_attns))

            # 在前景区域内进行对象竞争
            if len(all_obj_attns) > 0:
                # 将所有对象的attention堆叠起来 [num_objects, H, W]
                stacked_obj_attns = torch.stack(all_obj_attns, dim=0)  # [num_objects, H, W]

                # 找到每个像素对应的最大attention的对象索引
                max_obj_weights, max_obj_indices = torch.max(stacked_obj_attns, dim=0)

                # 为每个对象生成mask：只在前景区域内，该对象赢得竞争
                for obj_idx in range(num_objects):
                    # 该对象在该像素赢得竞争
                    obj_wins = (max_obj_indices == obj_idx)
                    # 只在整体前景区域内
                    obj_mask = obj_wins & overall_fg_mask
                    # 确保obj_mask是2维的 [H, W]，然后添加batch维度
                    if len(obj_mask.shape) == 3:
                        obj_mask = obj_mask.squeeze(0)  # 如果是 [1, H, W]，去掉第一个维度
                    elif len(obj_mask.shape) != 2:
                        # 如果不是2维，尝试reshape
                        obj_mask = obj_mask.view(-1, obj_mask.shape[-2], obj_mask.shape[-1])[0]
                    # 应用形态学操作
                    obj_mask_processed = remove_small_holes_and_points(obj_mask.unsqueeze(0))
                    object_masks.append(obj_mask_processed[0])

    return object_masks if len(object_masks) > 0 else [get_curr_fg_mask(pipe)]

def get_cross_sim(pipe):
    attn_processors = pipe.transformer.attn_processors
    all_cross_sims = []
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            all_cross_sims.append(processor.cross_sims)
            processor.cross_sims = None
    return sum(all_cross_sims) / len(all_cross_sims)

def reset_id_bank(pipe):
    attn_processors = pipe.transformer.attn_processors
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.id_attn_bank = dict()
    gc.collect()

def reset_size(pipe, h, w):
    attn_processors = pipe.transformer.attn_processors
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.size = (h//16, w//16)
            processor.visual_seq_len = processor.size[0] * processor.size[1]

def set_fg_lengths(pipe, fg_lengths):
    """设置多对象前景长度"""
    attn_processors = pipe.transformer.attn_processors
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.fg_lengths = fg_lengths

def get_curr_fg_masks_two_stage(pipe, fg_lengths):
    """两阶段前景mask生成"""
    attn_processors = pipe.transformer.attn_processors
    all_attn_weights = dict(bg=[], fg=[])

    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            saved_attns = processor.attn_weights
            for k in saved_attns:
                if k == "bg":
                    all_attn_weights[k].append(saved_attns[k])
                elif k == "fg" and isinstance(saved_attns[k], list):
                    # 多对象情况
                    if not all_attn_weights[k]:
                        all_attn_weights[k] = [[] for _ in range(len(saved_attns[k]))]
                    for i, fg_attn in enumerate(saved_attns[k]):
                        all_attn_weights[k][i].append(fg_attn)
                else:
                    # 单对象兼容
                    if not all_attn_weights[k]:
                        all_attn_weights[k] = []
                    all_attn_weights[k].append(saved_attns[k])

    # 第一阶段：计算整体前景区域（使用置信度方式）
    device = all_attn_weights["bg"][0].device
    bg_attns = sum(attn.to(device) for attn in all_attn_weights["bg"]) / len(all_attn_weights["bg"])

    # 计算整体前景权重（用于第一阶段）
    if isinstance(all_attn_weights["fg"][0], list):
        # 多对象：合并所有对象的权重
        combined_fg_weights = []
        for i in range(len(fg_lengths)):
            fg_attns_list = [attn[i].to(device) for attn in all_attn_weights["fg"]]
            combined_fg_weights.append(sum(fg_attns_list) / len(fg_attns_list))
        overall_fg_attns = sum(combined_fg_weights) / len(combined_fg_weights)
    else:
        overall_fg_attns = sum(attn.to(device) for attn in all_attn_weights["fg"]) / len(all_attn_weights["fg"])

    # 使用置信度确定整体前景区域
    fg_confidence = overall_fg_attns / (bg_attns + overall_fg_attns + 1e-6)
    fg_confidence_float = fg_confidence.float()
    overall_threshold = torch.quantile(fg_confidence_float, 0.55)
    overall_fg_mask = fg_confidence > overall_threshold
    overall_fg_mask = remove_small_holes_and_points(overall_fg_mask)

    # 第二阶段：在前景区域内直接比较各对象权重，赢者通吃
    fg_masks = []
    if "objects" in all_attn_weights and len(all_attn_weights["objects"]) > 0:
        # 收集所有对象的注意力权重
        all_obj_attns = []
        for obj_idx in range(num_objects):
            obj_attn_list = [obj_attns[obj_idx] for obj_attns in all_attn_weights["objects"] if len(obj_attns) > obj_idx]
            if len(obj_attn_list) > 0:
                obj_attn = sum(obj_attn_list) / len(obj_attn_list)
                # 确保obj_attn是2维的 [H, W]
                if len(obj_attn.shape) == 3:
                    obj_attn = obj_attn.squeeze(0)
                elif len(obj_attn.shape) != 2:
                    obj_attn = obj_attn.view(-1, obj_attn.shape[-2], obj_attn.shape[-1])[0]
                all_obj_attns.append(obj_attn)

        # 在前景区域内进行对象竞争
        if len(all_obj_attns) > 0:
            # 将所有对象权重堆叠起来 [num_objects, H, W]
            stacked_obj_attns = torch.stack(all_obj_attns, dim=0)

            # 找到每个像素对应的最大attention的对象索引
            max_obj_weights, max_obj_indices = torch.max(stacked_obj_attns, dim=0)

            # 为每个对象生成mask：只在前景区域内，该对象赢得竞争
            for obj_idx in range(num_objects):
                # 该对象在该像素赢得竞争
                obj_wins = (max_obj_indices == obj_idx)
                # 只在整体前景区域内
                obj_mask = obj_wins & overall_fg_mask
                # 应用形态学操作
                fg_masks.append(remove_small_holes_and_points(obj_mask))
    else:
        # 单对象情况
        fg_masks = [overall_fg_mask]

    # 清空attn_weights
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.attn_weights = dict()

    return fg_masks
