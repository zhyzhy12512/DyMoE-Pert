"""Predict one single-gene or double-gene perturbation from a trained checkpoint.

Run this script from the package root. The checkpoint is created by train.py.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
from models import ProcePertdata, scpert


EMBEDDING_FILES = {
    "norman": "gene_embeddings_norman_512.npy",
    "dixit": "gene_embeddings_dixit_512.npy",
    "adamson": "gene_embeddings_adamson_512.npy",
    "replogle_k562_essential": "gene_embeddings_replogle_k562_essential_512.npy",
    "replogle_rpe1_essential": "gene_embeddings_replogle_rpe1_essential_512.npy",
}


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained DyMoE-Pert model.")
    parser.add_argument("--data-name", choices=EMBEDDING_FILES, default="norman")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--embedding-dir", default="./embeddings")
    parser.add_argument("--checkpoint", required=True, help="Directory with config.pkl and model.pt")
    parser.add_argument("--perturbation", required=True, help="For example: CEBPA+ctrl or CEBPA+JUN")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--num-cells", type=int, default=32, help="Number of control cells to predict (default: 32)")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference cells per forward pass (default: 8)")
    args = parser.parse_args()
    if args.num_cells < 1 or args.batch_size < 1:
        parser.error("--num-cells and --batch-size must be positive")

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        torch.cuda.set_device(torch.device(device))

    embedding_path = os.path.join(args.embedding_dir, EMBEDDING_FILES[args.data_name])
    for path in (
        embedding_path,
        os.path.join(args.checkpoint, "config.pkl"),
        os.path.join(args.checkpoint, "model.pt"),
    ):
        if not os.path.isfile(path):
            parser.error("Required file not found: " + path)

    data = ProcePertdata.PertData(args.data_path)
    data.load(DataName=args.data_name)
    data.prepare_split(split="no_test", seed=77)
    data.get_dataloader(batch_size=64, test_batch_size=64)

    model = scpert.scPert(
        data,
        device=device,
        weight_bias_track=False,
        proj_name="DyMoE-Pert",
        exp_name=args.data_name,
        embedding_path=embedding_path,
    )
    model.load_pretrained(args.checkpoint)
    predictions = model.predict(
        [args.perturbation], num_samples=args.num_cells, batch_size=args.batch_size
    )
    key = args.perturbation.replace("+", "_")
    print("Saved pred_dymoe_{}.npz; mean prediction shape: {}".format(
        key, predictions[key].shape
    ))


if __name__ == "__main__":
    main()
    
