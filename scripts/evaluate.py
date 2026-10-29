"""Evaluation pipeline for trained NWPU-Caption model on test split.

Computes standard benchmark metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-L
Generates qualitative attention heatmaps for sample images and saves reports.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from image_recognition.dataset import (
    NWPUTorchDataset,
    resolve_dataset_dir,
)
from image_recognition.metrics import evaluate_corpus
from image_recognition.model import NWPUCaptioner
from image_recognition.visualizer import visualize_attention
from image_recognition.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Remote Sensing Captioner on Test Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to saved model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to NWPU dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Split to evaluate on")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size (1 = greedy, >1 = beam search)")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum generated caption length")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluation samples (for fast debug)")
    parser.add_argument("--num_visualizations", type=int, default=5, help="Number of qualitative attention heatmaps to render")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save evaluation reports and heatmaps")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[NWPUCaptioner, Vocabulary, Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    word2idx = checkpoint.get("vocab_word2idx")

    vocab = Vocabulary()
    if word2idx:
        vocab.word2idx = word2idx
        vocab.idx2word = {int(idx): word for word, idx in word2idx.items()}

    model = NWPUCaptioner(
        vocab_size=len(vocab),
        model_name=config.get("model_name", "convnext_tiny.in12k_ft_in1k"),
        pretrained=True,  # Loads base ImageNet weights; trainable adapter loaded from checkpoint
        use_lora=True,
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        embed_dim=config.get("embed_dim", 512),
        decoder_dim=config.get("decoder_dim", 512),
        attention_dim=config.get("attention_dim", 512),
        dropout=config.get("dropout", 0.5),
    )

    model.load_trainable_checkpoint(checkpoint_path, device)
    model.to(device)
    model.eval()

    print(f"Model successfully loaded (Epoch {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
    return model, vocab, config


def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active compute device: {device}")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    model, vocab, config = load_model_from_checkpoint(checkpoint_path, device)

    print(f"Loading {args.split} split dataset from {dataset_dir}...")
    eval_dataset = NWPUTorchDataset(
        dataset_dir=str(dataset_dir),
        split=args.split,
        vocab=vocab,
        mode="eval",
    )

    total_images = len(eval_dataset)
    num_eval = min(total_images, args.max_samples) if args.max_samples else total_images
    print(f"Evaluating on {num_eval:,} images (Total {args.split} images: {total_images:,})...")

    hypotheses: List[List[str]] = []
    references_list: List[List[List[str]]] = []
    sample_records: List[Dict] = []

    start_time = time.time()
    for idx in range(num_eval):
        image_tensor, raw_references, imgid = eval_dataset[idx]

        caption_str, tokens, attention_maps = model.generate(
            image=image_tensor,
            vocab=vocab,
            max_len=args.max_len,
            beam_size=args.beam_size,
        )

        hypotheses.append(tokens)

        # Tokenize reference captions using vocabulary tokenizer
        tokenized_refs = [vocab.tokenize(ref) for ref in raw_references]
        references_list.append(tokenized_refs)

        # Save visualization for first N samples
        if idx < args.num_visualizations:
            item = eval_dataset.eval_items[idx]
            pil_image = item.open_image()
            vis_filename = vis_dir / f"sample_{idx + 1}_{item.class_name}_{item.filename.split('.')[0]}_att.png"
            visualize_attention(
                image=pil_image,
                tokens=tokens,
                attention_maps=attention_maps,
                save_path=vis_filename,
            )
            sample_records.append({
                "index": idx + 1,
                "image": item.filename,
                "class": item.class_name,
                "generated": caption_str,
                "references": raw_references,
                "heatmap_path": str(vis_filename),
            })

        if (idx + 1) % 100 == 0 or (idx + 1) == num_eval:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"Processed [{idx + 1}/{num_eval}] images ({rate:.1f} imgs/s)")

    # Compute corpus-level metrics
    print("\nComputing benchmark captioning metrics (BLEU-1..4, ROUGE-L)...")
    metrics = evaluate_corpus(hypotheses, references_list)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print("=" * 50)
    for metric_name, score in metrics.items():
        print(f"  {metric_name:<10}: {score:.2f}%")
    print("=" * 50)

    # Save results to JSON
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_evaluated": num_eval,
        "beam_size": args.beam_size,
        "metrics": metrics,
        "samples": sample_records,
    }
    json_path = output_dir / f"evaluation_{args.split}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved JSON metrics to: {json_path}")

    # Save Markdown report
    md_path = output_dir / f"evaluation_{args.split}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Remote Sensing Image Captioning - {args.split.upper()} Split Evaluation\n\n")
        f.write(f"- **Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"- **Images Evaluated**: {num_eval:,}\n")
        f.write(f"- **Decoding Strategy**: {'Beam Search (k=' + str(args.beam_size) + ')' if args.beam_size > 1 else 'Greedy (k=1)'}\n\n")
        f.write("## Quantitative Benchmark Metrics\n\n")
        f.write("| Metric | Score (%) |\n")
        f.write("| :--- | :--- |\n")
        for k, v in metrics.items():
            f.write(f"| **{k}** | {v:.2f}% |\n")
        f.write("\n## Qualitative Predictions & Attention Heatmaps\n\n")
        for s in sample_records:
            f.write(f"### Sample {s['index']} — `{s['class']} / {s['image']}`\n\n")
            f.write(f"- **Generated**: *\"{s['generated']}\"*\n")
            f.write(f"- **Reference 1**: \"{s['references'][0]}\"\n")
            f.write(f"- **Reference 2**: \"{s['references'][1]}\"\n\n")
            f.write(f"![Attention Heatmap]({Path(s['heatmap_path']).name})\n\n")

    print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
