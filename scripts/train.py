import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.append(r"./")
from models import ProcePertdata, scpert

EMBEDDING_FILES = {
    "norman": "gene_embeddings_norman_512.npy",
    "dixit": "gene_embeddings_dixit_512.npy",
    "adamson": "gene_embeddings_adamson_512.npy",
    "replogle_k562_essential": "gene_embeddings_replogle_k562_essential_512.npy",
    "replogle_rpe1_essential": "gene_embeddings_replogle_rpe1_essential_512.npy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DyMoE-Pert on a perturbation dataset.")
    parser.add_argument("--data-name", choices=EMBEDDING_FILES, default="norman")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--embedding-dir", default="./embeddings")
    parser.add_argument("--embedding-file", default=None, help="Override the dataset-aligned embedding file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--split", default="simulation")
    parser.add_argument("--model-save-name", default=None)
    parser.add_argument("--multi-gpu", action="store_true", help="Use PyG DataParallel with DataListLoader")
    parser.add_argument("--device-ids", default="0,1,2,3", help="GPU ids for --multi-gpu, e.g. 0,1,2,3")
    parser.add_argument("--moe-hidden-dim", type=int, default=None, help="Hidden dimension inside MoE fusion; default follows --hidden-size")
    parser.add_argument("--moe-dropout", type=float, default=0.1)
    parser.add_argument("--moe-balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--moe-ablation-mode",
        choices=("full", "base_only", "no_interaction", "experts_only"),
        default="full",
        help="Optional MoE structural ablation mode.",
    )
    parser.add_argument(
        "--moe-router-mode",
        choices=("dynamic", "fixed_uniform"),
        default="dynamic",
        help="MoE router mode: dynamic first-version router or strict fixed-uniform E3 control.",
    )
    parser.add_argument("--no-moe-router-stats", action="store_true", help="Disable returning/logging MoE router stats")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        return "cuda:0"
    return device_arg


def parse_device_ids(device_ids_arg):
    if device_ids_arg in ("auto", "all"):
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in device_ids_arg.split(",") if x.strip()]


def main():
    args = parse_args()
    args.use_moe_fusion = True
    if args.moe_router_mode == "fixed_uniform" or args.moe_ablation_mode == "base_only":
        args.moe_balance_weight = 0.0
    device = resolve_device(args.device)
    device_ids = None

    if args.multi_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--multi-gpu was requested but torch.cuda.is_available() is False")
        device_ids = parse_device_ids(args.device_ids)
        if len(device_ids) < 2:
            raise ValueError("--multi-gpu needs at least two GPU ids")
        gpu_count = torch.cuda.device_count()
        invalid_ids = [i for i in device_ids if i < 0 or i >= gpu_count]
        if invalid_ids:
            raise ValueError(f"Invalid GPU ids {invalid_ids}; torch sees {gpu_count} CUDA device(s)")
        first_device = f"cuda:{device_ids[0]}"
        if args.device in ("auto", "cuda"):
            device = first_device
        elif device != first_device:
            raise ValueError(f"--device must match the first --device-ids entry: expected {first_device}")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(torch.device(device))
        torch.cuda.empty_cache()

    test_batch_size = args.test_batch_size or args.batch_size
    embedding_file = args.embedding_file or EMBEDDING_FILES[args.data_name]
    embedding_path = os.path.join(args.embedding_dir, embedding_file)
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    print(f"\n===== Processing dataset: {args.data_name} with embedding: {embedding_file} =====\n")
    print(
        "Training config: "
        f"device={device}, batch_size={args.batch_size}, "
        f"test_batch_size={test_batch_size}, hidden_size={args.hidden_size}, "
        f"multi_gpu={args.multi_gpu}, device_ids={device_ids}, "
        f"use_moe_fusion={args.use_moe_fusion}, moe_hidden_dim={args.moe_hidden_dim}, "
        f"moe_dropout={args.moe_dropout}, moe_ablation_mode={args.moe_ablation_mode}, "
        f"moe_router_mode={args.moe_router_mode}, "
        f"moe_balance_weight={args.moe_balance_weight}"
    )

    pertData = ProcePertdata.PertData(args.data_path)
    pertData.load(DataName=args.data_name)
    pertData.prepare_split(split=args.split, seed=args.seed)
    pertData.get_dataloader(
        batch_size=args.batch_size,
        test_batch_size=test_batch_size,
        use_data_list_loader=args.multi_gpu,
    )

    SCPert = scpert.scPert(
        pertData,
        device=device,
        weight_bias_track=False,
        proj_name="pertnet",
        exp_name=f"pertnet_{args.data_name}",
        embedding_path=embedding_path,
    )

    SCPert.model_initialize(
        hidden_size=args.hidden_size,
        use_moe_fusion=args.use_moe_fusion,
        moe_hidden_dim=args.moe_hidden_dim,
        moe_dropout=args.moe_dropout,
        moe_balance_weight=args.moe_balance_weight,
        moe_router_mode=args.moe_router_mode,
        moe_ablation_mode=args.moe_ablation_mode,
        moe_return_router_stats=not args.no_moe_router_stats,
    )
    SCPert.train(epochs=args.epochs, lr=args.lr, use_parallel=args.multi_gpu, device_ids=device_ids)

    model_save_name = args.model_save_name or f"{args.data_name}_model_FINAL"
    SCPert.save_model(model_save_name)

    print(f"\n===== Completed dataset: {args.data_name} =====\n")
    print("Training and checkpoint saving completed.")


if __name__ == "__main__":
    main()
