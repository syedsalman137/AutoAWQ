import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.phi3.modeling_phi3  import Phi3DecoderLayer, Phi3ForCausalLM
from awq.modules.fused.norm import FasterTransformerRMSNorm

class Phi3AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Phi3DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: Phi3ForCausalLM):
        fuser = Phi3Fuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: Phi3ForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: Phi3DecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: Phi3ForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: Phi3DecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.qkv_proj,
                ],
                inp=input_feat["self_attn.qkv_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        v_proj_shape = list(module.self_attn.qkv_proj.weight.shape)
        v_proj_shape[0] //= 3
        if v_proj_shape == list(module.self_attn.o_proj.weight.shape):
            layers.append(
                dict(
                    prev_op=module.self_attn.qkv_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_up_proj],
                inp=input_feat["mlp.gate_up_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.gate_up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers


class Phi3Fuser:
    def __init__(self, model: Phi3ForCausalLM):
        self.model = model

        self.Phi3_blocks: List[Tuple[str, Phi3DecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Phi3DecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: Phi3DecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device

            q_proj = nn.Linear(module.self_attn.qkv_proj.in_features, module.self_attn.qkv_proj.out_features  // 3)
            q_proj.weight = module.self_attn.qkv_proj.weight.chunk(3, dim=-2)[0]
            k_proj = nn.Linear(module.self_attn.qkv_proj.in_features, module.self_attn.qkv_proj.out_features  // 3)
            k_proj.weight = module.self_attn.qkv_proj.weight.chunk(3, dim=-2)[1]
            v_proj = nn.Linear(module.self_attn.qkv_proj.in_features, module.self_attn.qkv_proj.out_features  // 3)
            v_proj.weight = module.self_attn.qkv_proj.weight.chunk(3, dim=-2)[2]

            qkv = fuse_qkv(
                module,
                q_proj,
                k_proj,
                v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon,
            )
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                )
            )

        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
