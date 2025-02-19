# Custom ViT from T5
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import copy
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging as hf_logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5Config,
    T5ForConditionalGeneration,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5Model,
    T5PreTrainedModel,
    T5Stack,
)

logger = logging.getLogger("debug")


class ViTARCEmbedding(nn.Module):
    """
    A module for mixing input embeddings and positional embeddings according to different strategies.
    Instead of reading from a config, it now directly takes a `mixer_strategy` argument.

    Supported strategies:
      - 'hardcoded_normalization'
      - 'learnable_scaling'
      - 'weighted_sum'
      - 'weighted_sum_no_norm'
      - 'learnable_scaling_vec'
      - 'weighted_sum_vec'
      - 'weighted_sum_no_norm_vec'
      - 'positional_attention'
      - 'layer_norm'
      - 'default'

    Example usage:
        embedding_module = ViTARCEmbedding(embed_dim=512, mixer_strategy='weighted_sum')
        output_embeds = embedding_module(inputs_embeds, position_embeds)
    """
    def __init__(self, embed_dim: int, mixer_strategy: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.mixer_strategy = mixer_strategy

        # For 'learnable_scaling_vec', 'weighted_sum_vec', 'weighted_sum_no_norm_vec'
        # we need vector-based parameters (1, embed_dim).
        if self.mixer_strategy in ['learnable_scaling_vec',
                                   'weighted_sum_vec',
                                   'weighted_sum_no_norm_vec']:
            self.position_scale = nn.Parameter(torch.ones(1, embed_dim))
            self.input_weight = nn.Parameter(torch.ones(1, embed_dim))
            self.position_weight = nn.Parameter(torch.ones(1, embed_dim))

        # For 'learnable_scaling', 'weighted_sum', 'weighted_sum_no_norm'
        # we need scalar-based parameters (1,).
        if self.mixer_strategy in ['learnable_scaling',
                                   'weighted_sum',
                                   'weighted_sum_no_norm']:
            self.position_scale = nn.Parameter(torch.ones(1))
            self.input_weight = nn.Parameter(torch.ones(1))
            self.position_weight = nn.Parameter(torch.ones(1))

        # For 'positional_attention'
        if self.mixer_strategy == 'positional_attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)

        # For 'layer_norm'
        if self.mixer_strategy == 'layer_norm':
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs_embeds: torch.Tensor, position_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
            position_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]

        Returns:
            output_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
        """
        strategy = self.mixer_strategy

        if strategy == 'hardcoded_normalization':
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = inputs_embeds_norm + position_embeds_norm

        elif strategy in ['learnable_scaling', 'learnable_scaling_vec']:
            scaled_position_embeds = self.position_scale * position_embeds
            output_embeds = inputs_embeds + scaled_position_embeds

        elif strategy in ['weighted_sum', 'weighted_sum_vec']:
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = (self.input_weight * inputs_embeds_norm) + (self.position_weight * position_embeds_norm)

        elif strategy in ['weighted_sum_no_norm', 'weighted_sum_no_norm_vec']:
            output_embeds = (self.input_weight * inputs_embeds) + (self.position_weight * position_embeds)

        elif strategy == 'positional_attention':
            # Expand position_embeds to match the batch dimension of inputs_embeds
            position_embeds_expanded = position_embeds.expand(inputs_embeds.shape[0], -1, -1)

            # Reshape to [seq_len, batch_size, embed_dim] for MultiheadAttention
            inputs_embeds_reshaped = inputs_embeds.transpose(0, 1)
            position_embeds_reshaped = position_embeds_expanded.transpose(0, 1)

            attn_output, _ = self.attention(
                inputs_embeds_reshaped,
                position_embeds_reshaped,
                position_embeds_reshaped
            )
            output_embeds = inputs_embeds_reshaped + attn_output
            output_embeds = output_embeds.transpose(0, 1)  # back to [batch_size, seq_len, embed_dim]

        elif strategy == 'layer_norm':
            combined_embeds = inputs_embeds + position_embeds
            output_embeds = self.layer_norm(combined_embeds)

        elif strategy == 'default':
            output_embeds = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported mixer_strategy: {strategy}")

        return output_embeds


# SinusoidalAPE 
class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(16384).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())



class CustomT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, attn_type="self"):
        super().__init__(config)

         # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Four-diag-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")

        self.d_head = config.d_kv
        self.attn_type = attn_type        
        self.has_relative_attention_bias = has_relative_attention_bias

        if self.has_relative_attention_bias:
            # Apply 2D RPE in 1st layers
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            device = self.relative_attention_bias.weight.device

            if self.rpe_type == "Four-diag-slope-Alibi":
                # Two slopes are sufficient here, since we manipudate the distance matrix with pre-added per-diag-direction ratios.
                self.slopes_l = torch.Tensor(self.get_slopes(self.n_heads, start_exponent=1)).to(device)*-1
                self.slopes_r = torch.Tensor(self.get_slopes(self.n_heads, start_exponent=0.5)).to(device)*-1
            elif self.rpe_type in ["NoRPE"]:
                #self.relative_attention_bias = None  # No positional encoding bias
                pass
            else:
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
                    

    # https://github.com/ofirpress/attention_with_linear_biases/issues/5
    def get_slopes(self, n, start_exponent=1):
        def get_geometric_slopes(n, start_exponent):
            start = 2 ** (-start_exponent)  # Starting value 2^(-start_exponent)
            ratio = 2 ** -1  # Halving each step
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_geometric_slopes(n, start_exponent)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (get_geometric_slopes(closest_power_of_2, start_exponent) +
                    get_slopes(2 * closest_power_of_2, start_exponent)[0::2][:n - closest_power_of_2])    

    def compute_bias(self, query_length, key_length, device=None, relative_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device

        if self.rpe_type in ["NoRPE"]:
            # Zeros
            return torch.zeros((1, self.n_heads, query_length, key_length), device=device)        
        elif self.rpe_type in ["Four-diag-slope-Alibi"]:            
            relative_position = relative_position.to(device)

            if self.rpe_abs:
                relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
            else:
                relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

            self.slopes_l = self.slopes_l.to(device)
            self.slopes_r = self.slopes_r.to(device)

            # relative_position is pre-mult with factor 2**0.25 for top-right, down-right
            alibi_left = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
            alibi_right = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position

            values = torch.triu(alibi_right) + torch.tril(alibi_left)

            # Slice the relevant part of the bias before reshaping
            values = values[:, :query_length, :key_length]  # Slicing the tensor before reshaping
            values = values.view(1, self.n_heads, query_length, key_length)  # shape (1, num_heads, query_length, key_length)            

            return values            
        else:
            # Zeros
            return torch.zeros((1, self.n_heads, query_length, key_length), device=device)    

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        relative_position=None,        
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]


        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]


        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )

        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        attention_output_dict = {}

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9


        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                if self.rpe_type in ["Four-diag-slope-Alibi"]:
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=relative_position)
                else:                    
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=None)
            
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:                
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
                            

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        
        scores += position_bias_masked

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class CustomT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = CustomT5Attention(config, has_relative_attention_bias=has_relative_attention_bias, attn_type="self")
        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        relative_position=None,        
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,            
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            relative_position=relative_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class CustomT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)        
        self.EncDecAttention = CustomT5Attention(config, has_relative_attention_bias=False, attn_type="cross")
        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        relative_position=None,        
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            relative_position=relative_position,            
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs



class CustomT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)        
        # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Four-diag-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")

        self.layer = nn.ModuleList()
        self.layer.append(CustomT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(CustomT5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,        
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        relative_position=None,        
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            relative_position=relative_position,            
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,                
                relative_position=relative_position,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)                

        # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Four-diag-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")
        
        self.block = nn.ModuleList(
            [CustomT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )

        self.APE_mixer = ViTARCEmbedding(config.d_model, self.ape_mixer_strategy)
        self.config = config

        if self.ape_type == "LearnedAPE":
            # 2D LearnedAPE is the same as LearnedAPE
            # 2D LearnedAPE + OPE is not implemented, but you can extend base on the code easily
            self.wpe = nn.Embedding(2048, config.d_model)
            self.wpe.weight.data.normal_(
                    mean=0.0, std=config.initializer_factor * 1.0
            )

        elif self.ape_type == "SinusoidalAPE":
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)            

        elif self.ape_type == "SinusoidalAPE2D":            
            # 2D APE for encoder and cross attn            
            if config.use_OPE:
                # If with OPE, half enc is reserved for obj_idx
                self.wpe_obj_enc = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64
                self.wpe_x_enc = FixedAbsolutePositionalEmbedding(config.d_model/4) # 128/4 -> 32
                self.wpe_y_enc = FixedAbsolutePositionalEmbedding(config.d_model/4) # 128/4 -> 32

            # Decoder is the same old 2D
            self.wpe_x = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64
            self.wpe_y = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64

            # 1D APE for decoder/ non-2d positions
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)

        if self.rpe_type == "Four-diag-slope-Alibi":
            # Calculate relative positions for the 2D grid
            # Four different slopes for each diag direction, top-left, top-right, down-left, down-right
            grid_height = self.config.rows # 33
            grid_width = self.config.cols # 34
            large_dist = self.config.cols + 10 # 44

            # Calculate the x and y difference matrices
            relative_position_2d = self.calculate_2d_relative_positions(grid_height, grid_width)

            # Create distance matrices for x_diff and y_diff including <s> and </s> tokens
            total_length = grid_height * grid_width + 2  # +2 for <s> and </s>
            distance_matrix = torch.full((total_length, total_length), fill_value=large_dist, dtype=torch.float)  # 100 as a large distance

            # Assign the 2D relative positions to the correct part of the matrix
            distance_matrix[1:1 + grid_height * grid_width, 1:1 + grid_height * grid_width] = relative_position_2d

            # Optionally handle <s> and </s> relative positions
            distance_matrix[0, :] = large_dist  # <s> is far from everything
            distance_matrix[:, 0] = large_dist
            distance_matrix[-1, :] = large_dist+1  # </s> is far from everything
            distance_matrix[:, -1] = large_dist+1

            self.distance_matrix_2D = distance_matrix            

    def calculate_2d_relative_positions(self, grid_height, grid_width):
        # Define direction-specific factors
        top_right_factor = 2 ** 0.25
        down_right_factor = 2 ** 0.25

        # Create grid coordinates
        x_coords, y_coords = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.long),
            torch.arange(grid_width, dtype=torch.long),
            indexing='ij'
        )

        # Flatten the 2D grid coordinates
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()

        # Initialize the relative position matrix
        num_positions = grid_height * grid_width
        relative_position = torch.zeros((num_positions, num_positions), dtype=torch.float)

        # Calculate Manhattan distance between each pair of points
        for i in range(num_positions):
            for j in range(num_positions):
                x_diff = x_flat[i] - x_flat[j]
                y_diff = y_flat[i] - y_flat[j]
                manhattan_distance = float(abs(x_diff) + abs(y_diff))  # Convert to float

                # Adjust the distance based on the direction
                if x_diff < 0 and y_diff < 0:  # Top-right
                    manhattan_distance *= top_right_factor
                elif x_diff > 0 and y_diff < 0:  # Down-right
                    manhattan_distance *= down_right_factor

                relative_position[i, j] = manhattan_distance

        return relative_position


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        position_ids=None,
        return_dict=None,
        relative_position=None,
        object_idx=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.rpe_type == "Four-diag-slope-Alibi":
            relative_position = self.distance_matrix_2D

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length        

        if self.ape_type in ["SinusoidalAPE2D"]:
            # 1) Prepare or shape position_ids
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                # Usually from self-attn: past_key_values[0] => (k, v) with shape [batch, n_heads, seq_len, dim_per_head]
                # so we take the -2 dimension for length
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            # 2) Build the core 2D embeddings
            #    - rows/cols stored in config => e.g. ARC shapes
            rows = getattr(self.config, "rows", 33)
            cols = getattr(self.config, "cols", 34)
            grid_len = rows * cols

            # Flatten position_ids if needed
            flat_position_ids = position_ids.view(-1)

            # X-coordinates => repeated row times, Y-coordinates => repeat_interleave col times
            position_ids_x = torch.arange(cols, device=device).repeat(rows)
            position_ids_y = torch.arange(rows, device=device).repeat_interleave(cols)

            # Repeat for the batch
            batch_size = position_ids.shape[0]
            position_ids_x = position_ids_x.repeat(batch_size, 1)
            position_ids_y = position_ids_y.repeat(batch_size, 1)

            # Build your 2D embeddings
            # object_idx-based embeddings only in encoder
            if self.use_OPE and (not self.is_decoder):
                # 2D + object embeddings
                # e.g. self.wpe_obj_enc, self.wpe_x_enc, self.wpe_y_enc
                object_embeds = self.wpe_obj_enc(object_idx[:, 1:-1])  # shape [batch, some_len, embed_dim/2], for example
                position_embeds_x = self.wpe_x_enc(position_ids_x)
                position_embeds_y = self.wpe_y_enc(position_ids_y)

                # Expand X, Y if needed
                position_embeds_x = position_embeds_x.expand(object_embeds.size(0), -1, -1)
                position_embeds_y = position_embeds_y.expand(object_embeds.size(0), -1, -1)

                # Concatenate them
                position_embeds_2d = torch.cat((object_embeds, position_embeds_x, position_embeds_y), dim=-1)

            else:
                # Normal 2D scenario => x,y
                position_embeds_x = self.wpe_x(position_ids_x)  # shape [batch, grid_len, embed_dim/2]
                position_embeds_y = self.wpe_y(position_ids_y)  # shape [batch, grid_len, embed_dim/2]

                # Expand X, Y if needed
                position_embeds_x = position_embeds_x.expand(batch_size, -1, -1)
                position_embeds_y = position_embeds_y.expand(batch_size, -1, -1)

                position_embeds_2d = torch.cat((position_embeds_x, position_embeds_y), dim=-1)  # [batch, grid_len, embed_dim]

            # Also build 1D embeddings (some fallback for tokens outside the 2D region)
            position_embeds_1d = self.wpe(position_ids)  # shape [batch, seq_len, embed_dim]
            position_embeds_1d = position_embeds_1d.expand(position_embeds_2d.size(0), -1, -1)  # Expand along the batch size

            # 3) Insert 2D portion into 1D
            # We store final in 'position_embeds'
            # We'll typically place 2D from index=1 up to grid_len - 1, etc.
            p_seq_len = position_ids.shape[-1]
            position_embeds = position_embeds_1d.clone()
            # print("batch_size",batch_size)
            # print("position_embeds.shape", position_embeds.shape)
            # print("position_embeds_2d.shape", position_embeds_2d.shape)
            

            # A) If is_decoder => we handle offsets differently
            if self.is_decoder:
                # For the decoder, we often have the first token as <pad> or <s>.
                # We'll place the 2D portion from index [1 : grid_len-1] if enough length
                if p_seq_len >= grid_len - 1:
                    position_embeds[:, 1 : grid_len - 1] = position_embeds_2d[:, : grid_len - 2]
                elif p_seq_len == 1:
                    # Possibly only the first token <s> or <pad>, do nothing or partial
                    pos_index = flat_position_ids[0]
                    if pos_index == 0:
                        # e.g. first token is <s>, no 2D portion
                        pass
                    elif 1 <= pos_index < (grid_len - 2):
                        # place that single token from position_embeds_2d
                        # e.g. position_embeds[:, 0] = position_embeds_2d[:, pos_index-1]
                        position_embeds[:, 0] = position_embeds_2d[:, pos_index - 1]
                    else:
                        # pos_index beyond 2D range => fallback
                        pass
                else:
                    # partial coverage: we have p_seq_len > 1 but less than grid_len-1
                    position_embeds[:, 1 : p_seq_len] = position_embeds_2d[:, : (p_seq_len - 1)]

            else:
                # B) If not is_decoder => simpler approach for an encoder or partial usage
                # We might do something similar: fill [1 : grid_len-1]
                if p_seq_len >= grid_len:
                    position_embeds[:, 1 : grid_len] = position_embeds_2d[:, : (grid_len - 1)]
                else:
                    # partial coverage
                    position_embeds[:, 1 : p_seq_len] = position_embeds_2d[:, : (p_seq_len - 1)]

            # 4) Finally mix them into inputs_embeds using APE_mixer
            inputs_embeds = self.APE_mixer(inputs_embeds, position_embeds)
        

        if self.ape_type in [
            "SinusoidalAPE",
            "LearnedAPE",
        ]:
            # 1D APE cases
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
            position_embeds = self.wpe(position_ids)   
            inputs_embeds = self.APE_mixer(inputs_embeds, position_embeds)         
            #inputs_embeds += position_embeds

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None        
        encoder_decoder_position_bias = None
        

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)                
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)                
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,                    
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,                    
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    relative_position=relative_position,  # Pass the relative_position to the layer
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights),
            #                                  (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]            
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]                

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)


            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class ViTARCForConditionalGeneration(T5ForConditionalGeneration):
    """
    Specialized T5-based model for the ViTARC project, extending T5ForConditionalGeneration.

    This model can read the following fields from the T5Config (if present):
      - ape_type (str): e.g. 'SinusoidalAPE', 'SinusoidalAPE2D', 'LearnedAPE', or 'none'. Defaults to 'SinusoidalAPE2D'.
      - rpe_type (str): e.g. 'Four-diag-slope-Alibi'. Defaults to 'Four-diag-slope-Alibi'.
      - rpe_abs (bool): default True or False if not present.
      - use_OPE (bool): default True.
      - ape_mixer (str): indicates the approach to mixing embeddings, e.g. 'learnable_scaling', 'weighted_sum', etc.
                         (not used in this snippet, just carried in config).
    
    """
    def __init__(self, config: T5Config):        
        """
        Extracts custom positional-encoding fields from config if available:
          ape_type, rpe_type, rpe_abs, use_OPE, ape_mixer.
        """
        # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Four-diag-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")

        super().__init__(config)
        self.model_dim = config.d_model        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CustomT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,               
        object_idx: Optional[torch.FloatTensor] = None,                
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,                
                object_idx=object_idx,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
