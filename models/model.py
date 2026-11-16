import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import hashlib


class MLP(nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if batch_norm and i < len(sizes) - 2:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
            elif last_layer_act == "ReLU":
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MemoryEfficientMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, chunk_size=256):
        super().__init__()
        # assert d_model % nhead == 0, 
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.chunk_size = chunk_size
        

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        return x.view(-1, batch_size, self.nhead, self.d_k).permute(1, 2, 0, 3)

    def forward(self, query, key, value, key_padding_mask=None):
        # (seq, batch, d_model)
        batch_size = query.size(1)
        

        q = self.split_heads(self.q_proj(query), batch_size)  
        k = self.split_heads(self.k_proj(key), batch_size)
        v = self.split_heads(self.v_proj(value), batch_size)
        outputs = []
        for i in range(0, query.size(0), self.chunk_size):
            q_chunk = q[:, :, i:i+self.chunk_size, :]         
            k_chunk = k[:, :, i:i+self.chunk_size, :]         
            v_chunk = v[:, :, i:i+self.chunk_size, :]        
            

            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            chunk_output = torch.matmul(attn_probs, v_chunk)
            outputs.append(chunk_output)

        output = torch.cat(outputs, dim=2)                    
        output = output.permute(2, 0, 1, 3).reshape(-1, batch_size, self.nhead * self.d_k)
        output = self.out_proj(output)                       
        return output, None  

class ImprovedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = MemoryEfficientMultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src_permuted = src.permute(1, 0, 2)  # (seq, batch, dim)
        attn_output, _ = self.self_attn(src_permuted, src_permuted, src_permuted)
        attn_output = attn_output.permute(1, 0, 2)  # (batch, seq, dim)
        
 
        src = self.norm1(src + self.dropout(attn_output))
        
     
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src


class ResidualMoEFusion(nn.Module):
    def __init__(
        self,
        kg_input_dim=128,
        scgpt_input_dim=512,
        hidden_dim=64,
        output_dim=64,
        dropout=0.1,
        num_experts=3,
        router_mode="dynamic",
        ablation_mode="full",
    ):
        super().__init__()
        if num_experts != 3:
            raise ValueError("ResidualMoEFusion first version expects exactly 3 experts.")
        if router_mode not in ("dynamic", "fixed_uniform"):
            raise ValueError(f"Unsupported router mode: {router_mode}")
        if ablation_mode not in ("full", "base_only", "no_interaction", "experts_only"):
            raise ValueError(f"Unsupported MoE ablation mode: {ablation_mode}")
        self.num_experts = num_experts
        self.router_mode = router_mode
        self.ablation_mode = ablation_mode

        self.kg_projector = nn.Sequential(
            nn.Linear(kg_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.scgpt_projector = nn.Sequential(
            nn.Linear(scgpt_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.base_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.kg_expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.scgpt_expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.interaction_expert = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.inference_router_mode = "dynamic"
        router_last = self.router[-1]
        nn.init.normal_(router_last.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(router_last.bias)
        self._apply_ablation_freezing()

    def set_inference_router_mode(self, mode):
        if mode not in ("dynamic", "fixed_uniform"):
            raise ValueError(f"Unsupported inference router mode: {mode}")
        self.inference_router_mode = mode

    def _set_module_requires_grad(self, module, requires_grad):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _apply_ablation_freezing(self):
        if self.ablation_mode == "base_only":
            self._set_module_requires_grad(self.kg_expert, False)
            self._set_module_requires_grad(self.scgpt_expert, False)
            self._set_module_requires_grad(self.interaction_expert, False)
            self._set_module_requires_grad(self.router, False)
        elif self.ablation_mode == "no_interaction":
            self._set_module_requires_grad(self.interaction_expert, False)
        elif self.ablation_mode == "experts_only":
            self._set_module_requires_grad(self.base_mlp, False)

    def _ablation_mode_id(self):
        return {
            "full": 0,
            "base_only": 1,
            "no_interaction": 2,
            "experts_only": 3,
        }[self.ablation_mode]

    def _active_expert_mask_tensor(self, device):
        if self.ablation_mode == "base_only":
            values = [False, False, False]
        elif self.ablation_mode == "no_interaction":
            values = [True, True, False]
        else:
            values = [True, True, True]
        return torch.tensor(values, dtype=torch.bool, device=device)

    def _active_expert_names_tensor(self, device):
        names = torch.full((self.num_experts,), -1, dtype=torch.long, device=device)
        mask = self._active_expert_mask_tensor(device)
        names[mask] = torch.arange(self.num_experts, device=device)[mask]
        return names

    def _empty_router_aux(self, h_base):
        device = h_base.device
        empty_weights = torch.empty(0, self.num_experts, dtype=h_base.dtype, device=device)
        zero = torch.zeros((), dtype=h_base.dtype, device=device)
        inactive_expert = torch.zeros_like(h_base).detach()
        return {
            "router_logits": empty_weights,
            "raw_router_weights": empty_weights,
            "router_weights": empty_weights,
            "balance_loss": zero,
            "router_mode_fixed_uniform": torch.tensor(
                self.router_mode == "fixed_uniform",
                device=device,
            ),
            "inference_router_mode_fixed_uniform": torch.tensor(False, device=device),
            "moe_ablation_mode": torch.tensor(self._ablation_mode_id(), device=device),
            "active_expert_mask": self._active_expert_mask_tensor(device),
            "active_expert_names": self._active_expert_names_tensor(device),
            "base_enabled": torch.tensor(True, device=device),
            "router_enabled": torch.tensor(False, device=device),
            "expert_outputs": {
                "kg": inactive_expert,
                "scgpt": inactive_expert,
                "interaction": inactive_expert,
            },
        }

    def _compute_router_weights(self, router_logits, raw_router_weights):
        use_inference_fixed_uniform_router = False
        active_expert_mask = self._active_expert_mask_tensor(router_logits.device)

        if self.ablation_mode == "no_interaction":
            active_logits = router_logits[..., active_expert_mask]
            if self.router_mode == "fixed_uniform":
                active_weights = torch.full_like(active_logits, 1.0 / int(active_expert_mask.sum().item()))
            elif (not self.training) and self.inference_router_mode == "fixed_uniform":
                active_weights = torch.full_like(active_logits, 1.0 / int(active_expert_mask.sum().item()))
                use_inference_fixed_uniform_router = True
            else:
                active_weights = torch.softmax(active_logits, dim=-1)
            router_weights = torch.zeros_like(router_logits)
            router_weights[..., active_expert_mask] = active_weights
        elif self.router_mode == "fixed_uniform":
            router_weights = torch.full_like(router_logits, 1.0 / self.num_experts)
        elif (not self.training) and self.inference_router_mode == "fixed_uniform":
            router_weights = torch.full_like(router_logits, 1.0 / self.num_experts)
            use_inference_fixed_uniform_router = True
        else:
            router_weights = raw_router_weights

        return router_weights, use_inference_fixed_uniform_router, active_expert_mask

    def _compute_balance_loss(self, router_weights, active_expert_mask):
        active_weights = router_weights[..., active_expert_mask]
        if active_weights.numel() == 0:
            return torch.zeros((), dtype=router_weights.dtype, device=router_weights.device)
        reduce_dims = tuple(range(active_weights.dim() - 1))
        mean_router_weight = active_weights.mean(dim=reduce_dims) if reduce_dims else active_weights
        target = torch.full_like(mean_router_weight, 1.0 / active_weights.shape[-1])
        return torch.mean((mean_router_weight - target) ** 2)

    def forward(self, kg_embedding, scgpt_embedding, return_aux=False):
        z_kg = self.kg_projector(kg_embedding)
        z_sc = self.scgpt_projector(scgpt_embedding)
        base_input = torch.cat([z_kg, z_sc], dim=-1)

        if self.ablation_mode == "base_only":
            h_base = self.base_mlp(base_input)
            output = self.output_norm(h_base)
            if not return_aux:
                return output
            return output, self._empty_router_aux(h_base)

        if self.ablation_mode == "experts_only":
            h_base = None
        else:
            h_base = self.base_mlp(base_input)

        h_kg = self.kg_expert(z_kg)
        h_sc = self.scgpt_expert(z_sc)

        if self.ablation_mode == "no_interaction":
            h_interaction = torch.zeros_like(h_kg).detach()
        else:
            interaction_input = torch.cat([z_kg, z_sc, z_kg * z_sc, torch.abs(z_kg - z_sc)], dim=-1)
            h_interaction = self.interaction_expert(interaction_input)

        router_logits = self.router(base_input)
        raw_router_weights = torch.softmax(router_logits, dim=-1)
        router_weights, use_inference_fixed_uniform_router, active_expert_mask = self._compute_router_weights(
            router_logits,
            raw_router_weights,
        )
        expert_stack = torch.stack([h_kg, h_sc, h_interaction], dim=-2)
        expert_output = torch.sum(router_weights.unsqueeze(-1) * expert_stack, dim=-2)
        if self.ablation_mode == "experts_only":
            output = self.output_norm(expert_output)
            base_enabled = False
        else:
            output = self.output_norm(h_base + expert_output)
            base_enabled = True

        if not return_aux:
            return output

        balance_loss = self._compute_balance_loss(router_weights, active_expert_mask)
        aux = {
            "router_logits": router_logits,
            "raw_router_weights": raw_router_weights,
            "router_weights": router_weights,
            "balance_loss": balance_loss,
            "router_mode_fixed_uniform": torch.tensor(
                self.router_mode == "fixed_uniform",
                device=router_weights.device,
            ),
            "inference_router_mode_fixed_uniform": torch.tensor(
                use_inference_fixed_uniform_router,
                device=router_weights.device,
            ),
            "moe_ablation_mode": torch.tensor(
                self._ablation_mode_id(),
                device=router_weights.device,
            ),
            "active_expert_mask": active_expert_mask,
            "active_expert_names": self._active_expert_names_tensor(router_weights.device),
            "base_enabled": torch.tensor(base_enabled, device=router_weights.device),
            "router_enabled": torch.tensor(True, device=router_weights.device),
            "expert_outputs": {
                "kg": h_kg,
                "scgpt": h_sc,
                "interaction": h_interaction,
            },
        }
        return output, aux


class scPert_Model(nn.Module):
    def __init__(self, args,embedding_path):
        super(scPert_Model, self).__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        self.hidden_size = args['hidden_size']
        self.embedding_size = args['embedding_size']
        self.device = args['device']
        self.use_moe_fusion = bool(args.get('use_moe_fusion', False))
        self.moe_router_mode = args.get('moe_router_mode', 'dynamic')
        if self.moe_router_mode not in ("dynamic", "fixed_uniform"):
            raise ValueError(f"Unsupported MoE router mode: {self.moe_router_mode}")
        self.moe_ablation_mode = args.get('moe_ablation_mode', 'full')
        if self.moe_ablation_mode not in ("full", "base_only", "no_interaction", "experts_only"):
            raise ValueError(f"Unsupported MoE ablation mode: {self.moe_ablation_mode}")
        self.moe_balance_weight = float(args.get('moe_balance_weight', 0.01))
        if self.moe_router_mode == "fixed_uniform" or self.moe_ablation_mode == "base_only":
            self.moe_balance_weight = 0.0
        self.moe_return_router_stats = bool(args.get('moe_return_router_stats', True))
        self._last_moe_aux = None

        # Load gene embeddings 
        gene_data = np.load("./embeddings/gene_2_kge_comgcn_final_common.npz")
        self.loaded_gene_names = gene_data['gene_names']
        self.register_buffer(
            "loaded_embeddings",
            torch.tensor(gene_data['embeddings'], dtype=torch.float32)
        )
        self.gene_to_index = {gene: idx for idx, gene in enumerate(self.loaded_gene_names)}

        # Load pert embeddings
        pert_data = np.load("./embeddings/gene_embeddingss_full_common.npz")
        self.pert_gene_names = pert_data['gene_names']
        self.register_buffer(
            "pert_embeddings",
            torch.tensor(pert_data['embeddings'], dtype=torch.float32)
        )
        self.pert_to_index = {gene: idx for idx, gene in enumerate(self.pert_gene_names)}
        self.register_buffer(
            "gene_emb",
            torch.tensor(np.load(embedding_path), dtype=torch.float32)
        )


        self.gene_interaction_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.gene_dense = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        

        self.pert_dense = nn.Sequential(
            nn.Linear(640, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        moe_hidden_dim = int(args.get('moe_hidden_dim') or self.hidden_size)
        moe_dropout = float(args.get('moe_dropout', 0.1))
        if self.use_moe_fusion:
            self.moe_fusion = ResidualMoEFusion(
                kg_input_dim=128,
                scgpt_input_dim=512,
                hidden_dim=moe_hidden_dim,
                output_dim=self.hidden_size,
                dropout=moe_dropout,
                num_experts=3,
                router_mode=self.moe_router_mode,
                ablation_mode=self.moe_ablation_mode,
            )
        else:
            self.moe_fusion = None
        

        self.pert_fuse = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )


        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerLayer(self.hidden_size, nhead=8, dropout=0.1)
            for _ in range(8) 
        ])
        

        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.LayerNorm(self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        self.output_layer = nn.Linear(self.hidden_size, 1)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)


        self.random_embedding_dim = 640
        

        self.emb_scale = nn.Parameter(torch.ones(1))
        

        self.pert_emb = nn.Embedding(self.num_perts, self.hidden_size)
        self.control_emb = nn.Parameter(torch.randn(1, self.hidden_size) / np.sqrt(self.hidden_size))
        self.pos_emb = nn.Embedding(self.num_genes, self.hidden_size)    
        self.to(self.device)

    def _fuse_pert_embedding(self, kg_embedding, scgpt_embedding, moe_aux_list=None):
        if not self.use_moe_fusion:
            combined_pert_emb = torch.cat([kg_embedding, scgpt_embedding])
            return self.pert_dense(combined_pert_emb)

        fused, aux = self.moe_fusion(
            kg_embedding,
            scgpt_embedding,
            return_aux=True,
        )
        if moe_aux_list is not None:
            moe_aux_list.append(aux)
        return fused

    def _merge_moe_aux(self, aux_list):
        device = self._runtime_device()
        hidden_size = self.hidden_size
        if not aux_list:
            zero = torch.zeros((), device=device)
            empty_weights = torch.empty(0, self.moe_fusion.num_experts, device=device)
            empty_expert = torch.empty(0, hidden_size, device=device)
            return {
                "router_logits": empty_weights,
                "raw_router_weights": empty_weights,
                "router_weights": empty_weights,
                "balance_loss": zero,
                "router_mode_fixed_uniform": torch.tensor(
                    self.moe_router_mode == "fixed_uniform",
                    device=device,
                ),
                "inference_router_mode_fixed_uniform": torch.tensor(False, device=device),
                "moe_ablation_mode": torch.tensor(
                    {"full": 0, "base_only": 1, "no_interaction": 2, "experts_only": 3}[self.moe_ablation_mode],
                    device=device,
                ),
                "active_expert_mask": torch.tensor(
                    {
                        "full": [True, True, True],
                        "base_only": [False, False, False],
                        "no_interaction": [True, True, False],
                        "experts_only": [True, True, True],
                    }[self.moe_ablation_mode],
                    dtype=torch.bool,
                    device=device,
                ),
                "active_expert_names": torch.tensor(
                    {
                        "full": [0, 1, 2],
                        "base_only": [-1, -1, -1],
                        "no_interaction": [0, 1, -1],
                        "experts_only": [0, 1, 2],
                    }[self.moe_ablation_mode],
                    dtype=torch.long,
                    device=device,
                ),
                "base_enabled": torch.tensor(self.moe_ablation_mode != "experts_only", device=device),
                "router_enabled": torch.tensor(self.moe_ablation_mode != "base_only", device=device),
                "expert_outputs": {
                    "kg": empty_expert,
                    "scgpt": empty_expert,
                    "interaction": empty_expert,
                },
                "router_entropy": zero,
            }
        router_enabled = torch.stack([
            aux["router_enabled"].to(device=device)
            for aux in aux_list
        ])
        if not router_enabled.bool().any():
            zero = torch.stack([aux["balance_loss"] for aux in aux_list]).mean()
            empty_weights = torch.empty(0, self.moe_fusion.num_experts, device=device)
            empty_expert = torch.empty(0, hidden_size, device=device)
            return {
                "router_logits": empty_weights,
                "raw_router_weights": empty_weights,
                "router_weights": empty_weights,
                "balance_loss": zero,
                "router_mode_fixed_uniform": torch.stack([
                    aux["router_mode_fixed_uniform"].to(device=device)
                    for aux in aux_list
                ]).any(),
                "inference_router_mode_fixed_uniform": torch.stack([
                    aux["inference_router_mode_fixed_uniform"].to(device=device)
                    for aux in aux_list
                ]).any(),
                "moe_ablation_mode": aux_list[0]["moe_ablation_mode"].to(device=device),
                "active_expert_mask": aux_list[0]["active_expert_mask"].to(device=device),
                "active_expert_names": aux_list[0]["active_expert_names"].to(device=device),
                "base_enabled": torch.stack([
                    aux["base_enabled"].to(device=device)
                    for aux in aux_list
                ]).any(),
                "router_enabled": torch.tensor(False, device=device),
                "expert_outputs": {
                    "kg": empty_expert,
                    "scgpt": empty_expert,
                    "interaction": empty_expert,
                },
                "router_entropy": torch.zeros((), device=device),
            }
        router_logits = torch.stack([aux["router_logits"] for aux in aux_list], dim=0)
        raw_router_weights = torch.stack([aux["raw_router_weights"] for aux in aux_list], dim=0)
        router_weights = torch.stack([aux["router_weights"] for aux in aux_list], dim=0)
        balance_loss = torch.stack([aux["balance_loss"] for aux in aux_list]).mean()
        router_mode_flags = torch.stack([
            aux["router_mode_fixed_uniform"].to(device=device)
            for aux in aux_list
        ])
        fixed_uniform_flags = torch.stack([
            aux["inference_router_mode_fixed_uniform"].to(device=device)
            for aux in aux_list
        ])
        ablation_modes = torch.stack([
            aux["moe_ablation_mode"].to(device=device)
            for aux in aux_list
        ])
        active_expert_mask = torch.stack([
            aux["active_expert_mask"].to(device=device)
            for aux in aux_list
        ])
        active_expert_names = torch.stack([
            aux["active_expert_names"].to(device=device)
            for aux in aux_list
        ])
        base_enabled = torch.stack([
            aux["base_enabled"].to(device=device)
            for aux in aux_list
        ])
        expert_outputs = {
            name: torch.stack([aux["expert_outputs"][name] for aux in aux_list], dim=0)
            for name in ("kg", "scgpt", "interaction")
        }
        if router_weights.numel() == 0:
            entropy = torch.zeros((), device=device)
        else:
            entropy = -torch.sum(
                router_weights * torch.log(router_weights + 1e-8),
                dim=-1,
            ).mean()
        return {
            "router_logits": router_logits,
            "raw_router_weights": raw_router_weights,
            "router_weights": router_weights,
            "balance_loss": balance_loss,
            "router_mode_fixed_uniform": router_mode_flags.any(),
            "inference_router_mode_fixed_uniform": fixed_uniform_flags.any(),
            "moe_ablation_mode": ablation_modes,
            "active_expert_mask": active_expert_mask,
            "active_expert_names": active_expert_names,
            "base_enabled": base_enabled.any(),
            "router_enabled": router_enabled.any(),
            "expert_outputs": expert_outputs,
            "router_entropy": entropy,
        }

    def _runtime_device(self):
        return self.control_emb.device

    def get_random_embedding(self, gene_name):
        device = self._runtime_device()
        seed = int(hashlib.sha1(gene_name.encode("utf-8")).hexdigest()[:16], 16)
        seed = seed % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        embedding = torch.randn(
            self.random_embedding_dim,
            generator=generator,
            dtype=torch.float32
        ) / np.sqrt(self.random_embedding_dim)
        logging.info(f"Generating deterministic random embedding for {gene_name}")
        return embedding.to(device)


    def get_gene_embedding(self, gene_name):

        if gene_name == 'ctrl':
            return torch.zeros(128, device=self._runtime_device()) 
        if gene_name in self.gene_to_index:
            # logging.info(f"get KGE {gene_name}")
            return self.loaded_embeddings[self.gene_to_index[gene_name]]
        return self.get_random_embedding(gene_name)[:128]

    def get_pert_embedding(self, pert_name):


        if pert_name == 'ctrl':
            return torch.zeros(512, device=self._runtime_device())  
            

        if pert_name in self.pert_to_index:
            return self.pert_embeddings[self.pert_to_index[pert_name]]
        

        return self.get_random_embedding(pert_name)[128:]

    def set_inference_router_mode(self, mode):
        if self.moe_fusion is None:
            raise RuntimeError("MoE fusion is not enabled.")
        self.moe_fusion.set_inference_router_mode(mode)

    def forward(self, pert_data):
        x, pert = pert_data.x, pert_data.pert
        device = x.device
        num_graphs = len(pert_data.batch.unique())
        moe_aux_list = [] if self.use_moe_fusion else None

        gene_emb = self.gene_emb
        gene_emb = gene_emb.repeat(num_graphs, 1)
        gene_emb = self.gene_dense(gene_emb)
        gene_emb = gene_emb.view(num_graphs, self.num_genes, -1)

        gene_interaction = torch.matmul(gene_emb, gene_emb.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        gene_interaction = F.softmax(gene_interaction, dim=-1)
        gene_context = torch.matmul(gene_interaction, gene_emb)
        gene_context = self.gene_interaction_layer(gene_context) 
        gene_emb = gene_emb + gene_context

        pert_emb_matrix = torch.zeros(num_graphs, self.num_genes, self.hidden_size, device=device)

        for idx, p in enumerate(pert):


            if p == 'ctrl':

                pert_emb_matrix[idx] = self.control_emb.expand(self.num_genes, -1)
                continue

            
            pert_list = p.split('+')
            pert_embs = []
            
            for pert_name in pert_list:
                new_pert_emb = self.get_gene_embedding(pert_name)
                scGPT_pert_emb = self.get_pert_embedding(pert_name)
                combined_pert_emb = self._fuse_pert_embedding(
                    new_pert_emb,
                    scGPT_pert_emb,
                    moe_aux_list=moe_aux_list,
                )
                pert_embs.append(combined_pert_emb)

            if pert_embs:

                if len(pert_embs) == 1:
                    pert_embs = pert_embs * 2
                    
                stacked_embs = torch.stack(pert_embs)
                emb_total = self.pert_fuse(stacked_embs).mean(0, keepdim=True)
                
                if len(emb_total.shape) == 2:
                    emb_total = emb_total.mean(0, keepdim=True)
                    
                pert_emb_matrix[idx] = emb_total.repeat(self.num_genes, 1)

        control_mask = (pert_emb_matrix.sum(dim=-1) == 0).unsqueeze(-1)
        pert_emb_matrix = torch.where(control_mask, self.control_emb.expand_as(pert_emb_matrix), pert_emb_matrix)


        pert_emb_matrix = pert_emb_matrix * self.emb_scale


        combined_emb = self.layer_norm1(gene_emb + pert_emb_matrix)
        combined_emb = self.dropout(combined_emb)


        for layer in self.transformer_layers:
            layer_output = layer(combined_emb)
            combined_emb = combined_emb + layer_output * 0.5

        combined_emb = self.layer_norm2(combined_emb)

        batch_size, seq_len, hidden_size = combined_emb.shape
        output = self.final_mlp(combined_emb.view(-1, hidden_size)).view(batch_size, seq_len, hidden_size)
        output = self.output_layer(output)


        output = output.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
        output = torch.split(torch.flatten(output), self.num_genes)

        output = torch.stack(output)
        if self.use_moe_fusion and self.moe_return_router_stats:
            return output, self._merge_moe_aux(moe_aux_list)
        return output
    
