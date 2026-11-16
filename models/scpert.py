from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch_geometric.data import Batch
try:
    from torch_geometric.nn import DataParallel as PyGDataParallel
except ImportError:
    PyGDataParallel = None

from .model import scPert_Model
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis
from .utils import loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")


def _is_pyg_parallel(model):
    return PyGDataParallel is not None and isinstance(model, PyGDataParallel)


def _is_parallel_model(model):
    return isinstance(model, nn.DataParallel) or _is_pyg_parallel(model)


def _unwrap_parallel_model(model):
    return model.module if _is_parallel_model(model) else model


def _prepare_train_batch(batch, device, model):
    if isinstance(batch, list):
        y = torch.cat([data.y for data in batch], dim=0).to(device)
        perts = [data.pert for data in batch]
        if _is_pyg_parallel(model):
            return batch, y, perts
        pyg_batch = Batch.from_data_list(batch)
        pyg_batch.to(device)
        return pyg_batch, y, perts

    batch.to(device)
    return batch, batch.y, batch.pert


def _split_model_output(model_output):
    if isinstance(model_output, tuple) and len(model_output) == 2:
        return model_output
    return model_output, None


def _moe_balance_loss(base_loss, moe_aux, model):
    if not isinstance(moe_aux, dict) or "balance_loss" not in moe_aux:
        return base_loss, None

    unwrapped_model = _unwrap_parallel_model(model)
    balance_weight = float(getattr(unwrapped_model, "moe_balance_weight", 0.0))
    if balance_weight <= 0:
        return base_loss, None

    balance_loss = moe_aux["balance_loss"]
    if not torch.is_tensor(balance_loss):
        balance_loss = torch.tensor(balance_loss, dtype=base_loss.dtype, device=base_loss.device)
    balance_loss = balance_loss.to(device=base_loss.device, dtype=base_loss.dtype).mean()
    return base_loss + balance_weight * balance_loss, balance_loss


def _tensor_any(value):
    if torch.is_tensor(value):
        return bool(value.detach().bool().any().cpu().item())
    return bool(value)


def _mode_from_aux(moe_aux):
    mode_names = {
        0: "full",
        1: "base_only",
        2: "no_interaction",
        3: "experts_only",
    }
    mode = moe_aux.get("moe_ablation_mode", 0)
    if torch.is_tensor(mode):
        flat = mode.detach().reshape(-1).cpu()
        mode_id = int(flat[0].item()) if flat.numel() else 0
    else:
        mode_id = int(mode)
    return mode_names.get(mode_id, f"unknown_{mode_id}")


def _active_expert_names_from_aux(moe_aux):
    names = ("kg", "scgpt", "interaction")
    mask = moe_aux.get("active_expert_mask", None)
    if mask is None:
        return list(names)
    if torch.is_tensor(mask):
        mask = mask.detach().reshape(-1, 3).bool().cpu()[0].tolist()
    return [name for name, is_active in zip(names, mask) if is_active]


def _format_moe_stats(moe_aux):
    if not isinstance(moe_aux, dict):
        return None

    moe_ablation_mode = _mode_from_aux(moe_aux)
    active_experts = _active_expert_names_from_aux(moe_aux)
    base_enabled = _tensor_any(moe_aux.get("base_enabled", True))
    router_enabled = _tensor_any(moe_aux.get("router_enabled", True))
    router_weights = moe_aux.get("router_weights", None)
    if not torch.is_tensor(router_weights) or router_weights.numel() == 0:
        return (
            f"MoE moe_ablation_mode={moe_ablation_mode} "
            f"active_experts={active_experts} "
            f"base_enabled={base_enabled} "
            f"router_enabled={router_enabled} "
            f"router_status=disabled"
        )

    router_weights = router_weights.detach().reshape(-1, router_weights.shape[-1])
    raw_router_weights = moe_aux.get("raw_router_weights", router_weights)
    if torch.is_tensor(raw_router_weights) and raw_router_weights.numel() > 0:
        raw_router_weights = raw_router_weights.detach().reshape(-1, raw_router_weights.shape[-1])
    else:
        raw_router_weights = router_weights
    avg_weights = router_weights.mean(dim=0)
    raw_avg_weights = raw_router_weights.mean(dim=0)
    min_weights = router_weights.min(dim=0).values
    max_weights = router_weights.max(dim=0).values
    entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()

    balance = moe_aux.get("balance_loss", None)
    if torch.is_tensor(balance):
        balance_value = balance.detach().mean().item()
    elif balance is None:
        balance_value = 0.0
    else:
        balance_value = float(balance)

    router_mode_fixed = _tensor_any(moe_aux.get("router_mode_fixed_uniform", False))
    moe_router_mode = "fixed_uniform" if router_mode_fixed else "dynamic"

    return (
        f"MoE balance={balance_value:.6f} "
        f"moe_ablation_mode={moe_ablation_mode} "
        f"moe_router_mode={moe_router_mode} "
        f"active_experts={active_experts} "
        f"base_enabled={base_enabled} "
        f"router_enabled={router_enabled} "
        f"applied_alpha_kg={avg_weights[0].item():.6f} "
        f"applied_alpha_scgpt={avg_weights[1].item():.6f} "
        f"applied_alpha_interaction={avg_weights[2].item():.6f} "
        f"raw_alpha_kg={raw_avg_weights[0].item():.6f} "
        f"raw_alpha_scgpt={raw_avg_weights[1].item():.6f} "
        f"raw_alpha_interaction={raw_avg_weights[2].item():.6f} "
        f"alpha_min=[{min_weights[0].item():.4f},{min_weights[1].item():.4f},{min_weights[2].item():.4f}] "
        f"alpha_max=[{max_weights[0].item():.4f},{max_weights[1].item():.4f},{max_weights[2].item():.4f}] "
        f"entropy={entropy.item():.4f}"
    )


class scPert:
    """
    scPert base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'scPert', 
                 exp_name = 'scPert',
                 embedding_path=None):
        """
        Initialize scPert model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'scPert'
        exp_name: str
            Experiment name for wandb. Default: 'scPert'

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}     
        self.embedding_path = embedding_path
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         use_moe_fusion = False,
                         moe_hidden_dim = None,
                         moe_dropout = 0.1,
                         moe_balance_weight = 0.01,
                         moe_router_mode = "dynamic",
                         moe_ablation_mode = "full",
                         moe_return_router_stats = True,
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        if moe_router_mode not in ("dynamic", "fixed_uniform"):
            raise ValueError(f"Unsupported MoE router mode: {moe_router_mode}")
        if moe_ablation_mode not in ("full", "base_only", "no_interaction", "experts_only"):
            raise ValueError(f"Unsupported MoE ablation mode: {moe_ablation_mode}")
        if moe_router_mode == "fixed_uniform" or moe_ablation_mode == "base_only":
            moe_balance_weight = 0.0

        self.config = {'hidden_size': hidden_size,
                      'embedding_size': 512,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'direction_lambda' : direction_lambda,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       "pert_names": self.pert_list,
                       "pert_node_map":self.node_map_pert,
                       "use_moe_fusion": use_moe_fusion,
                       "moe_hidden_dim": moe_hidden_dim,
                       "moe_dropout": moe_dropout,
                       "moe_balance_weight": moe_balance_weight,
                       "moe_router_mode": moe_router_mode,
                       "moe_ablation_mode": moe_ablation_mode,
                       "moe_return_router_stats": moe_return_router_stats,
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
            
        self.model = scPert_Model(self.config, self.embedding_path ).to(self.device)
        self.best_model = deepcopy(self.model).cpu()
        
    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        exclude_keys = ['device', 'num_genes', 'num_perts', 'embedding_size', 'pert_names', 'pert_node_map']
        # Remove parameters that are not needed by model_initialize
        for key in exclude_keys:
            config.pop(key, None)

        # Initialize the model with the loaded configuration
        if 'uncertainty' in config:
            del config['uncertainty']
            del config['uncertainty_reg']
        self.model_initialize(**config)
        self.config = config

        # Load the model state
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(_unwrap_parallel_model(self.best_model).state_dict(), os.path.join(path, 'model.pt'))

    def predict(self, pert_list, type_list=True, dir=None, num_samples=300, batch_size=8):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        num_samples: int
            number of control cells sampled for each perturbation
        batch_size: int
            number of cells per inference forward pass

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome

        """
        
        if num_samples < 1 or batch_size < 1:
            raise ValueError('num_samples and batch_size must be positive')

        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        
        from torch_geometric.data import DataLoader

        perturbations = pert_list if type_list else [pert_list]
        for pert in perturbations:
            if isinstance(pert, str):
                pert_key = pert.replace('+', '_')
                pert_for_model = pert
            else:
                pert_key = '_'.join(pert)
                pert_for_model = '+'.join(pert)

            all_pert_key = f'all_{pert_key}'
            if pert_key in self.saved_pred:
                results_pred[pert_key] = self.saved_pred[pert_key]
                if all_pert_key in self.saved_pred:
                    results_pred[all_pert_key] = self.saved_pred[all_pert_key]
                continue

            cg = create_cell_graph_dataset_for_prediction(pert_for_model, self.ctrl_adata,
                                                    self.pert_list, self.device,
                                                    num_samples=num_samples)
            loader = DataLoader(cg, batch_size=batch_size, shuffle=False)
            cell_predictions = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    p, _ = _split_model_output(self.best_model(batch))
                    cell_predictions.append(p.detach().cpu().numpy())

            all_predictions = np.concatenate(cell_predictions, axis=0)
            results_pred[pert_key] = all_predictions.mean(axis=0)
            results_pred[all_pert_key] = all_predictions
                
        self.saved_pred.update(results_pred)
        output_dir = dir if dir is not None else '.'
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, f'pred_dymoe_{pert_key}.npz'), **results_pred)
        return results_pred
    
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the GI scores following perturbation of a given gene combination

        Parameters
        ----------
        combo: list
            list of genes to be perturbed
        GI_genes_file: str
            path to the file containing genes with high mean expression

        Returns
        -------
        GI scores for the given combinatorial perturbation based on scPert
        predictions

        """
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[f'{combo[0]}+ctrl']
            pred[combo[1]] = self.saved_pred[f'{combo[1]}+ctrl']
            pred['+'.join(combo)] = self.saved_pred['+'.join(combo)]
        except:
            pred = self.predict([f'{combo[0]}+ctrl', f'{combo[1]}+ctrl', f'{combo[0]}+{combo[1]}'])
        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 
        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
        pred = {p: pred[p][GI_genes_idx] 
                for p in pred 
                if "all" not in p.lower()}
        
        return get_GI_params(pred, combo)

    
    def train(self, epochs=20, 
            lr=0.002,
            weight_decay=1e-5,
            use_parallel=True,
            device_ids=None,
            key_genes=None
            ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """

        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        parallel = use_parallel and device_ids is not None and len(device_ids) > 1
        self.model = self.model.to(self.device)
        if parallel:
            if PyGDataParallel is None:
                raise RuntimeError("torch_geometric.nn.DataParallel is unavailable")
            print_sys(f"Using PyG DataParallel on GPUs: {device_ids}")
            self.model = PyGDataParallel(
                self.model,
                device_ids=device_ids,
                output_device=device_ids[0]
            )

        best_model = deepcopy(_unwrap_parallel_model(self.model)).cpu()
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters available for optimizer.")
        optimizer = optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader))
        min_val = np.inf
        if key_genes:
            gene2idx = self.node_map
            gene_indices = [gene2idx[gene] for gene in key_genes]
        else:
            gene_indices = None

        unwrapped_train_model = _unwrap_parallel_model(self.model)
        print_sys(
            "MoE training config: "
            f"hidden_size={getattr(unwrapped_train_model, 'hidden_size', None)} "
            f"moe_hidden_dim={self.config.get('moe_hidden_dim')} "
            f"moe_ablation_mode={self.config.get('moe_ablation_mode', 'full')} "
            f"moe_router_mode={self.config.get('moe_router_mode', 'dynamic')} "
            f"moe_balance_weight={getattr(unwrapped_train_model, 'moe_balance_weight', 0.0)}"
        )
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                model_batch, y, perts = _prepare_train_batch(batch, self.device, self.model)
                optimizer.zero_grad()
                pred, moe_aux = _split_model_output(self.model(model_batch))
                task_loss = loss_fct(pred, y, perts=perts,
                                ctrl = self.ctrl_expression, 
                                dict_filter = self.dict_filter,
                            model_params=(p for p in self.model.parameters() if p.requires_grad),
                            class_weights_indices=gene_indices)
                loss, moe_balance = _moe_balance_loss(task_loss, moe_aux, self.model)
                loss.backward()
                nn.utils.clip_grad_value_(trainable_parameters, clip_value=1.0)
                optimizer.step()


                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}"
                    if moe_balance is not None:
                        log += " Task Loss: {:.4f} MoE Weighted Balance: {:.6f}"
                        print_sys(log.format(
                            epoch + 1,
                            step + 1,
                            loss.item(),
                            task_loss.item(),
                            (_unwrap_parallel_model(self.model).moe_balance_weight * moe_balance).item(),
                        ))
                    else:
                        print_sys(log.format(epoch + 1, step + 1, loss.item()))

                    moe_stats = _format_moe_stats(moe_aux)
                    if moe_stats is not None:
                        print_sys(moe_stats)

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,self.device)
            val_res = evaluate(val_loader, self.model,self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(_unwrap_parallel_model(self.model)).cpu()
                # best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model
        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_model = self.best_model.to(self.device)
        if parallel:
            test_model = PyGDataParallel(
                test_model,
                device_ids=device_ids,
                output_device=device_ids[0]
            )
        test_res = evaluate(test_loader, test_model,self.device)
        self.best_model = _unwrap_parallel_model(test_model)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        print("test_metrics is :", test_metrics)
        print("test_pert_res is :", test_pert_res)
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']       

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')


