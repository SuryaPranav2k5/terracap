"""Training pipeline for NWPU-Caption (ConvNeXt-Tiny + LoRA + Spatial Attention LSTM).

Optimized for both local execution and Kaggle GPU (T4/P100/V100/A100).
Supports mixed precision (AMP), differential learning rates, and automatic path detection.
"""

import argparse
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from image_recognition.dataset import (
    NWPUTorchDataset,
    caption_collate_fn,
    resolve_dataset_dir,
)
from image_recognition.model import NWPUCaptioner
from image_recognition.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Remote Sensing Image Captioner on NWPU-Caption")

    # Data paths
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to NWPU dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")

    # Model hyperparameters
    parser.add_argument("--model_name", type=str, default="convnext_tiny.in12k_ft_in1k", help="ConvNeXt backbone checkpoint")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--embed_dim", type=int, default=512, help="Word embedding dimension")
    parser.add_argument("--decoder_dim", type=int, default=512, help="LSTM decoder hidden dimension")
    parser.add_argument("--attention_dim", type=int, default=512, help="Attention projection dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Decoder dropout probability")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr_encoder", type=float, default=1e-4, help="Learning rate for LoRA adapter")
    parser.add_argument("--lr_decoder", type=float, default=4e-4, help="Learning rate for Attention LSTM Decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Max gradient norm clipping")
    parser.add_argument("--lambda_att", type=float, default=1.0, help="Doubly-stochastic attention penalty factor")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")

    return parser.parse_args()


def resolve_checkpoint_dir(checkpoint_dir: Optional[str] = None) -> Path:
    """Auto-detects output directory (/kaggle/working/checkpoints vs local checkpoints)."""
    if checkpoint_dir:
        out_path = Path(checkpoint_dir)
    elif Path("/kaggle/working").exists():
        out_path = Path("/kaggle/working/checkpoints")
    else:
        out_path = Path("checkpoints")
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def train_one_epoch(
    model: NWPUCaptioner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    grad_clip: float,
    lambda_att: float,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Runs one full training epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_att_loss = 0.0
    start_time = time.time()

    total_steps = len(loader)
    for step, batch in enumerate(loader):
        images, captions, lengths = batch[0], batch[1], batch[2]
        img_indices = batch[3] if len(batch) > 3 else None

        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        if img_indices is not None:
            img_indices = img_indices.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                packed_preds, packed_targets, decode_lengths, alphas = model(
                    images, captions, lengths, img_indices=img_indices
                )
                loss, ce_loss, att_loss = model.compute_loss(
                    packed_preds, packed_targets, decode_lengths, alphas, lambda_att=lambda_att
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            packed_preds, packed_targets, decode_lengths, alphas = model(
                images, captions, lengths, img_indices=img_indices
            )
            loss, ce_loss, att_loss = model.compute_loss(
                packed_preds, packed_targets, decode_lengths, alphas, lambda_att=lambda_att
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
            )
            optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_att_loss += att_loss.item()

        if (step + 1) % 100 == 0 or (step + 1) == total_steps:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(
                f"Epoch [{epoch}] Step [{step + 1}/{total_steps}] | "
                f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Att: {att_loss.item():.4f}) | "
                f"Speed: {steps_per_sec:.2f} steps/s"
            )

    avg_loss = total_loss / total_steps
    avg_ce = total_ce_loss / total_steps
    avg_att = total_att_loss / total_steps
    return avg_loss, avg_ce, avg_att


@torch.no_grad()
def evaluate_validation_loss(
    model: NWPUCaptioner,
    loader: DataLoader,
    device: torch.device,
    lambda_att: float,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Evaluates loss over validation split."""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_att_loss = 0.0
    total_steps = len(loader)

    for batch in loader:
        images, captions, lengths = batch[0], batch[1], batch[2]
        img_indices = batch[3] if len(batch) > 3 else None

        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        if img_indices is not None:
            img_indices = img_indices.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                packed_preds, packed_targets, decode_lengths, alphas = model(
                    images, captions, lengths, img_indices=img_indices
                )
                loss, ce_loss, att_loss = model.compute_loss(
                    packed_preds, packed_targets, decode_lengths, alphas, lambda_att=lambda_att
                )
        else:
            packed_preds, packed_targets, decode_lengths, alphas = model(
                images, captions, lengths, img_indices=img_indices
            )
            loss, ce_loss, att_loss = model.compute_loss(
                packed_preds, packed_targets, decode_lengths, alphas, lambda_att=lambda_att
            )

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_att_loss += att_loss.item()

    return (
        total_loss / total_steps,
        total_ce_loss / total_steps,
        total_att_loss / total_steps,
    )


def run_training(args: argparse.Namespace) -> None:
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)

    print("=" * 60)
    print("REMOTE SENSING IMAGE CAPTIONING - TRAINING PIPELINE")
    print(f"Dataset directory:    {dataset_dir.resolve()}")
    print(f"Checkpoints directory: {checkpoint_dir.resolve()}")
    print("=" * 60)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active compute device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        torch.backends.cudnn.benchmark = True

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # 2. Vocabulary
    vocab_path = dataset_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Run scripts/build_vocab.py first.")
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded Vocabulary: {len(vocab):,} tokens (Vocab path: {vocab_path})")

    # 3. Datasets and Loaders
    print("Initializing Datasets and DataLoaders...")
    train_dataset = NWPUTorchDataset(
        dataset_dir=str(dataset_dir),
        split="train",
        vocab=vocab,
        mode="train",
    )
    val_dataset = NWPUTorchDataset(
        dataset_dir=str(dataset_dir),
        split="val",
        vocab=vocab,
        mode="train",  # Unrolled for fast validation loss
    )
    val_eval_dataset = NWPUTorchDataset(
        dataset_dir=str(dataset_dir),
        split="val",
        vocab=vocab,
        mode="eval",  # Grouped for qualitative demo
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=caption_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=caption_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train samples: {len(train_dataset):,} ({len(train_loader)} batches/epoch)")
    print(f"Validation samples: {len(val_dataset):,} ({len(val_loader)} batches)")

    # 4. Model Construction
    print("Instantiating ConvNeXt-Tiny + LoRA + Spatial Attention LSTM Model...")
    model = NWPUCaptioner(
        vocab_size=len(vocab),
        model_name=args.model_name,
        pretrained=True,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    model.to(device)
    model.encoder.print_trainable_parameters()

    # 5. Optimizer with Differential Learning Rates
    lora_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": args.lr_encoder},
            {"params": decoder_params, "lr": args.lr_decoder},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    # 6. Training Loop
    best_val_loss = float("inf")
    print(f"\nBeginning training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_ce, train_att = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            lambda_att=args.lambda_att,
            use_amp=use_amp,
        )

        val_loss, val_ce, val_att = evaluate_validation_loss(
            model=model,
            loader=val_loader,
            device=device,
            lambda_att=args.lambda_att,
            use_amp=use_amp,
        )

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start
        current_lr_enc = optimizer.param_groups[0]["lr"]
        current_lr_dec = optimizer.param_groups[1]["lr"]

        print("-" * 60)
        print(
            f"Epoch [{epoch}/{args.epochs}] Complete in {epoch_time:.1f}s | "
            f"LR (Enc/Dec): {current_lr_enc:.2e} / {current_lr_dec:.2e}\n"
            f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Att: {train_att:.4f})\n"
            f"  Val Loss:   {val_loss:.4f} (CE: {val_ce:.4f}, Att: {val_att:.4f})"
        )

        # Inspect qualitative progress on 1 sample validation image
        sample_img, sample_refs, _ = val_eval_dataset[0]
        gen_caption, _, _ = model.generate(sample_img, vocab=vocab, max_len=25, beam_size=1)
        print(f"  [Sample Validation Prediction]: \"{gen_caption}\"")
        print(f"  [Sample Reference 1]:           \"{sample_refs[0]}\"")

        # Save best model (lean checkpoint excluding frozen 28M backbone weights)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            model.save_trainable_checkpoint(
                filepath=best_path,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                vocab=vocab,
                config=vars(args),
            )
            print(f"  --> Saved new best lean checkpoint to {best_path} (Val Loss: {val_loss:.4f})")

        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        model.save_trainable_checkpoint(
            filepath=latest_path,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            vocab=vocab,
            config=vars(args),
        )
        print("-" * 60)

    print("\nTraining completed successfully!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
