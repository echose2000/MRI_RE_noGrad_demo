"""Run the 2D recurrent reconstruction experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import torch
import yaml
from einops import rearrange
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from mask import mask_generator
from report_2d import ReconstructionReport, StageMetrics
from results.model import ReconkneeFactory
from utils import A, setup_seed


DEFAULT_CONFIG = Path(__file__).resolve().parent / "results" / "config" / "2d_img.yaml"

CONFIG_OVERRIDES = {
    "data": "pth",
    "steps": "train_steps",
    "recursive_steps": "recursive_steps",
    "eval_every": "eval_every",
    "lr": "lr",
    "save_results": "save_results",
    "result_dir": "result_dir",
}


@dataclass
class Sample:
    undersampled: torch.Tensor
    input_mask: torch.Tensor
    csm: torch.Tensor
    loss_mask: torch.Tensor
    loss_target: torch.Tensor
    reference: torch.Tensor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML configuration file.")
    parser.add_argument("--data", type=Path, help="Override the HDF5 path from the YAML file.")
    parser.add_argument("--sample-index", type=int, default=0, help="HDF5 sample index (default: 0).")
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda:0 (default: auto).")
    parser.add_argument("--model", choices=("Basicmodel", "ISTAPlus"), help="Reconstruction model.")
    parser.add_argument("--steps", type=int, help="Number of optimizer updates.")
    parser.add_argument("--recursive-steps", type=int,
                        help="Total model calls per reconstruction, including the final call.")
    parser.add_argument("--eval-every", type=int, help="Evaluate every N iterations.")
    parser.add_argument("--lr", type=float, help="AdamW learning rate.")
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=None,
                        help="Save figures and metrics (default comes from YAML).")
    parser.add_argument("--result-dir", type=Path, help="Directory for saved experiment results.")
    parser.add_argument("--no-show", action="store_true", help="Run without Matplotlib windows.")
    return parser.parse_args(argv)


def load_config(args):
    with args.config.open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    for option, key in CONFIG_OVERRIDES.items():
        value = getattr(args, option)
        if value is not None:
            cfg[key] = value
    return cfg


def as_channels(tensor):
    return rearrange(torch.view_as_real(tensor), "coil h w two -> 1 (coil two) h w")


def kspace_to_image(ops, kspace, csm):
    combined = ops.combine_csm(ops.ktoi(kspace), csm)
    magnitude = torch.view_as_complex(
        rearrange(combined, "b two h w -> b 1 h w two").contiguous()
    ).abs()
    h, w = magnitude.shape[-2:]
    cropped = magnitude[..., h // 2 - 160 : h // 2 + 160, w // 2 - 160 : w // 2 + 160]
    return cropped / cropped.max()


def prepare_sample(cfg, index, ops, device):
    with h5py.File(cfg["pth"], "r") as dataset:
        if not 0 <= index < len(dataset["kspace"]):
            raise IndexError(f"sample-index {index} outside [0, {len(dataset['kspace']) - 1}]")
        kspace = torch.from_numpy(dataset["kspace"][index])
        csm = torch.from_numpy(dataset["csm"][index])

    csm = csm / (csm * csm.conj()).abs().sum(dim=0, keepdim=True).sqrt()
    csm[torch.isnan(csm)] = 0
    csm = as_channels(csm)

    coil_images = ops.ktoi(as_channels(kspace))
    complex_images = torch.view_as_complex(
        rearrange(coil_images, "1 (coil two) h w -> 1 coil h w two", two=2).contiguous()
    )
    kspace = ops.itok(coil_images / complex_images.abs().max())
    reference = kspace_to_image(ops, kspace, csm)

    mask_cfg = cfg["mask"]
    generate_mask = mask_generator(
        seed=cfg["seed"], shape=kspace.shape[-2:], accelerate=mask_cfg["acc"],
        acs_lines=mask_cfg["acs"], type=mask_cfg["type"],
    )
    # The two masks are fixed by their seeds for the entire experiment.
    input_mask = generate_mask(0).to(device)
    loss_mask = generate_mask(20).to(device)
    kspace = kspace.to(device)
    return Sample(
        undersampled=kspace * input_mask,
        input_mask=input_mask,
        csm=csm.to(device),
        loss_mask=loss_mask,
        loss_target=kspace * loss_mask,
        reference=reference.to(device),
    )


def reconstruct(predict, kspace, steps, on_stage=None):
    """Run recurrent model calls in the caller's gradient mode."""
    for _ in range(max(1, steps)):
        kspace = predict(kspace)
        if on_stage is not None:
            on_stage(kspace)
    return kspace


@torch.no_grad()
def evaluate(predict, steps, sample, ops, metrics):
    images = []
    scores = []

    def record_stage(prediction):
        image = kspace_to_image(ops, prediction, sample.csm)
        metrics.reset()
        values = metrics(image, sample.reference)
        images.append(image)
        scores.append(StageMetrics(
            psnr=values["PeakSignalNoiseRatio"].item(),
            ssim=values["StructuralSimilarityIndexMeasure"].item(),
        ))

    reconstruct(predict, sample.undersampled, steps, on_stage=record_stage)
    return images, scores


def train_step(predict, steps, sample, optimizer):
    optimizer.zero_grad()
    prediction = reconstruct(predict, sample.undersampled, steps)
    loss = torch.linalg.norm(prediction * sample.loss_mask - sample.loss_target)
    loss.backward()
    optimizer.step()


def main(argv=None):
    # Resolve the experiment settings and select a device.
    args = parse_args(argv)
    cfg = load_config(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or cfg.get("model_name", "Basicmodel")
    steps = max(1, int(cfg.get("train_steps", 5000)))
    recursive_steps = max(1, int(cfg.get("recursive_steps", 1)))
    eval_every = max(1, int(cfg.get("eval_every", 100)))

    # Load the selected slice and prepare its masks and reference image.
    setup_seed(cfg["seed"])
    ops = A(modality="2D")
    sample = prepare_sample(cfg, args.sample_index, ops, device)

    # Build the model and bind the inputs that stay fixed across recurrent calls.
    model_cfg = cfg["model"]
    model = ReconkneeFactory.get_recon_model(
        model_name=model_name, coils=sample.csm.shape[1] // 2,
        hidden_channels=model_cfg["hidden_channels"], num_layers=model_cfg["num_layers"],
        iter_num=model_cfg["iter_num"], kernel_size=model_cfg["kernel_size"], share=model_cfg["share"],
    ).to(device)
    predict = partial(model, mask=sample.input_mask, csm=sample.csm, kspace0=sample.undersampled)

    # Set up the optimizer and image quality metrics.
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    metrics = MetricCollection([
        PeakSignalNoiseRatio(data_range=1),
        StructuralSimilarityIndexMeasure(),
    ]).to(device)

    # Initialize reporting with the reference and zero-filled images.
    sampling_rate = sample.input_mask.mean().item()
    print(f"{model_name} on {device} | sample {args.sample_index} | sampling rate {sampling_rate:.2%}")
    print(f"{steps} optimizer updates | {recursive_steps} recurrent calls with gradients per update")
    zero_filled = kspace_to_image(ops, sample.undersampled, sample.csm)
    report = ReconstructionReport(sample.reference, zero_filled, recursive_steps, show=not args.no_show)

    # Train and periodically evaluate the updated model.
    with tqdm(range(steps)) as progress:
        for iteration in progress:
            train_step(predict, recursive_steps, sample, optimizer)
            if iteration % eval_every == 0:
                images, scores = evaluate(predict, recursive_steps, sample, ops, metrics)
                report.record(iteration, images, scores)
                final = scores[-1]
                progress.set_postfix(psnr=f"{final.psnr:.2f}", SSIM=f"{final.ssim:.4f}")

    # Finish the report and save results when requested.
    result_dir = Path(cfg.get("result_dir", "results")) if cfg.get("save_results", False) else None
    report.finish(model_name, sampling_rate, result_dir)


if __name__ == "__main__":
    main()
