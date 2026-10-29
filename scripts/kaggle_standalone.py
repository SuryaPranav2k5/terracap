"""All-in-One Standalone Training & Evaluation Script for Kaggle Notebooks.

Paste this entire script into a single Kaggle Notebook code cell with GPU accelerator (T4 or P100).
Key features:
1. 1-Pass Batched Image Feature Expansion: ConvNeXt encoder runs ONLY ONCE per image (25,193 passes/epoch
   instead of 125,965), expanding visual features across all 5 captions in collate_fn for a 5x speedup.
2. Low-Rank Adaptation (LoRA PEFT): Fine-tunes 1.06M adapter parameters on mlp.fc1/fc2 while 27.8M backbone is frozen.
3. Show-Attend-Tell Deep Output Layer: Fuses LSTM hidden state, gated attention context, and word embedding.
4. True Beam Search (k=3): Evaluates test split with length-normalized beam search and spatial heatmap tracking.
5. Standard pycocoevalcap: Evaluates BLEU-1..4, CIDEr (unscaled), ROUGE-L, and METEOR (with PTBTokenizer when Java is present).
6. Lean Checkpoints: Saves only trainable parameters (~12MB vs 120MB), saving 90% Kaggle disk space.
"""

import os
import sys
import subprocess
import math
import time
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import Counter

# 1. Install prerequisites on Kaggle if missing
try:
    import timm
    import peft
    import pycocoevalcap
except ImportError:
    print("Installing timm, peft, and pycocoevalcap...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "peft", "pycocoevalcap"])
    import timm
    import peft
    import pycocoevalcap

# Bypass PEFT torchao incompatibility bug in Kaggle / Colab environments
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
CONFIG = {
    "model_name": "convnext_tiny.in12k_ft_in1k",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "embed_dim": 512,
    "decoder_dim": 512,
    "attention_dim": 512,
    "dropout": 0.5,
    "epochs": 15,
    "batch_size": 16,     # 16 unique images per batch (80 captions evaluated concurrently)
    "lr_encoder": 1e-4,
    "lr_decoder": 4e-4,
    "weight_decay": 1e-2,
    "grad_clip": 5.0,
    "lambda_att": 1.0,
    "num_workers": 2,
    "beam_size": 3,       # True Beam Search enabled during test evaluation
    "max_len": 30,
    "num_visualizations": 5,
    "max_eval_samples": None,  # None evaluates all 3,148 test images; set to e.g. 100 for a fast trial run
}

# ==============================================================================
# PATH AUTO-DETECTION
# ==============================================================================
def find_dataset_dir() -> Path:
    """Recursively locates directory containing dataset_nwpu.json across any directory structure."""
    # 1. Walk /kaggle/input directly (instant O(1) to O(N) traversal on Kaggle)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _, files in os.walk(kaggle_input):
            if "dataset_nwpu.json" in files:
                found = Path(root)
                print(f"[Auto-Detect] Successfully found dataset at: {found}")
                return found

    # 2. Walk /content (Google Colab)
    colab_root = Path("/content")
    if colab_root.exists():
        for root, _, files in os.walk(colab_root):
            if "dataset_nwpu.json" in files:
                found = Path(root)
                print(f"[Auto-Detect] Successfully found dataset at: {found}")
                return found

    # 3. Check local directory
    for root, _, files in os.walk("."):
        if "dataset_nwpu.json" in files:
            found = Path(root)
            print(f"[Auto-Detect] Successfully found dataset at: {found}")
            return found

    # 4. Diagnostic dump if not found anywhere
    diag_lines = []
    if kaggle_input.exists():
        diag_lines.append("Contents of /kaggle/input:")
        for item in kaggle_input.iterdir():
            diag_lines.append(f"  - {item}")
            if item.is_dir():
                try:
                    for sub in list(item.iterdir())[:8]:
                        diag_lines.append(f"      └── {sub.name}")
                except Exception as e:
                    diag_lines.append(f"      └── (error reading: {e})")
    else:
        diag_lines.append("/kaggle/input does not exist on this machine.")

    raise FileNotFoundError(
        "Could not find 'dataset_nwpu.json' anywhere in /kaggle/input or local directories.\n"
        + "\n".join(diag_lines) + "\n\n"
        "Kaggle Troubleshooting:\n"
        "1. Check if the dataset is attached in the right-hand panel under 'Input'.\n"
        "2. If you attached it while this session was active, please click the 'Restart' icon in Kaggle\n"
        "   to refresh the mounted filesystem."
    )

DATASET_DIR = find_dataset_dir()
OUTPUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORTS_DIR = OUTPUT_DIR / "reports"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset root detected:    {DATASET_DIR}")
print(f"Checkpoints destination:  {CHECKPOINT_DIR}")

# ==============================================================================
# VOCABULARY ENGINE (Strictly Identical to src/image_recognition/vocab.py)
# ==============================================================================
class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        for idx, token in enumerate([self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def __getitem__(self, item: Union[str, int]) -> Union[int, str]:
        if isinstance(item, str):
            return self.word2idx.get(item, self.UNK_IDX)
        return self.idx2word.get(item, self.UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.word2idx)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Splits sentence into lowercase alphanumeric tokens.
        Strictly identical to src/image_recognition/vocab.py to prevent silent mismatch.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        indices = [self[tok] for tok in tokens]
        if add_special_tokens:
            indices = [self.START_IDX] + indices + [self.END_IDX]
        return indices

    def decode_to_tokens(self, indices: List[int], skip_special: bool = True) -> List[str]:
        words = []
        for idx in indices:
            if idx == self.END_IDX and skip_special:
                break
            if skip_special and idx in (self.PAD_IDX, self.START_IDX):
                continue
            words.append(self[idx])
        return words

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        return " ".join(self.decode_to_tokens(indices, skip_special=skip_special))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in data["word2idx"].items()}
        return vocab

# ==============================================================================
# PYTORCH DATASET & 1-PASS BATCHED COLLATOR
# ==============================================================================
class NWPUTorchDataset(Dataset):
    """PyTorch Dataset for NWPU-Caption.

    In 'train' mode:
        Yields (image, list_of_5_encoded_captions). The ConvNeXt encoder runs
        only ONCE per image, expanding features across the 5 captions in collate_fn.
    In 'eval' mode:
        Yields (image, list_of_5_reference_strings, image_id, image_path, class_name).
    """

    def __init__(self, dataset_dir: Path, split: str = "train", vocab: Optional[Vocabulary] = None, mode: str = "train"):
        self.base_dir = dataset_dir
        self.split = split
        self.mode = mode
        self.vocab = vocab or Vocabulary.load(dataset_dir / "vocab.json")

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split == "train":
            self.transform = T.Compose([T.Resize((256, 256)), T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalize])
        else:
            self.transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), normalize])

        with open(dataset_dir / "dataset_nwpu.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        images_base = dataset_dir / "images"
        self.items = []
        for class_name, entries in data.items():
            for entry in entries:
                if entry.get("split") == split:
                    captions = [entry[k] for k in ["raw", "raw_1", "raw_2", "raw_3", "raw_4"] if k in entry]
                    img_path = images_base / class_name / entry["filename"]
                    self.items.append({
                        "filename": entry["filename"],
                        "class_name": class_name,
                        "image_path": img_path,
                        "captions": captions,
                        "imgid": entry.get("imgid", -1),
                    })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        image = Image.open(it["image_path"]).convert("RGB")
        tensor_img = self.transform(image)

        if self.mode == "train":
            cap_tensors = [
                torch.tensor(self.vocab.encode(c, add_special_tokens=True), dtype=torch.long)
                for c in it["captions"]
            ]
            return tensor_img, cap_tensors
        else:
            return tensor_img, it["captions"], it["imgid"], it["image_path"], it["class_name"]

def caption_collate_fn(batch):
    """Pads captions and sorts descending by length.

    Batch items: (image, list_of_5_caption_tensors).
    Returns (images, targets, lengths, img_indices) where images has shape (B, 3, 256, 256),
    allowing ConvNeXt to execute only ONCE per image while decoding all 5 captions concurrently.
    """
    images, multi_captions = zip(*batch)
    images = torch.stack(images, dim=0)  # (B, 3, 256, 256)

    flat_captions = []
    for img_idx, caps in enumerate(multi_captions):
        for cap in caps:
            flat_captions.append((cap, img_idx))

    # Sort all captions across the batch descending by length
    flat_captions.sort(key=lambda x: len(x[0]), reverse=True)

    sorted_caps = [c[0] for c in flat_captions]
    img_indices = torch.tensor([c[1] for c in flat_captions], dtype=torch.long)
    lengths = torch.tensor([len(c) for c in sorted_caps], dtype=torch.long)
    max_len = lengths.max().item()

    targets = torch.full((len(sorted_caps), max_len), fill_value=Vocabulary.PAD_IDX, dtype=torch.long)
    for i, cap in enumerate(sorted_caps):
        targets[i, : len(cap)] = cap

    return images, targets, lengths, img_indices

# ==============================================================================
# MODEL ARCHITECTURE (ConvNeXt-Tiny + LoRA + Spatial Attention LSTM)
# ==============================================================================
class ConvNeXtEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = 768

        if use_lora:
            for p in base_model.parameters():
                p.requires_grad = False
            lora_cfg = peft.LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["fc1", "fc2"], lora_dropout=lora_dropout, bias="none")
            self.backbone = peft.get_peft_model(base_model, lora_cfg)
        else:
            self.backbone = base_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(images)  # (B, C, H, W)
        b, c, h, w = feat.shape
        return feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

    def print_trainable_parameters(self) -> None:
        """Prints trainable parameter counts cleanly without raising AttributeError."""
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable %: {100 * trainable / total:.2f}%")

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int = 768, decoder_dim: int = 512, attention_dim: int = 512):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.energy_proj = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        att_enc = self.encoder_proj(encoder_out)
        att_dec = self.decoder_proj(decoder_hidden).unsqueeze(1)
        energy = self.energy_proj(torch.tanh(att_enc + att_dec)).squeeze(2)
        alpha = F.softmax(energy, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class AttentionLSTMDecoder(nn.Module):
    """LSTM Decoder with Spatial Attention and Show-Attend-Tell Deep Output."""

    def __init__(self, vocab_size: int, embed_dim: int = 512, decoder_dim: int = 512, encoder_dim: int = 768, attention_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocabulary.PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.beta_gate = nn.Linear(decoder_dim, encoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Show-Attend-Tell Deep Output Layer (Section 3.1.3)
        self.deep_output = nn.Linear(decoder_dim + encoder_dim + embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_feat = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean_feat)), torch.tanh(self.init_c(mean_feat))

    def step(self, word_idx: torch.Tensor, encoder_out: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        context, alpha = self.attention(encoder_out, h_prev)
        gate = torch.sigmoid(self.beta_gate(h_prev))
        gated_context = gate * context

        word_embed = self.embedding(word_idx)
        lstm_input = torch.cat([word_embed, gated_context], dim=1)
        h_new, c_new = self.lstm_cell(lstm_input, (h_prev, c_prev))

        # Deep output layer projection
        deep_input = torch.cat([h_new, gated_context, word_embed], dim=1)
        deep_feat = self.dropout(torch.tanh(self.deep_output(deep_input)))
        logits = self.fc_out(deep_feat)
        return logits, h_new, c_new, alpha

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        b, n, _ = encoder_out.shape
        decode_lengths = lengths - 1
        max_decode_len = int(decode_lengths.max().item())
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(b, max_decode_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(b, max_decode_len, n, device=encoder_out.device)

        for t in range(max_decode_len):
            b_t = int((decode_lengths > t).sum().item())
            preds, h_new, c_new, alpha = self.step(
                captions[:b_t, t], encoder_out[:b_t], h[:b_t], c[:b_t]
            )

            predictions[:b_t, t, :] = preds
            alphas[:b_t, t, :] = alpha

            h = torch.cat([h_new, h[b_t:]], dim=0)
            c = torch.cat([c_new, c[b_t:]], dim=0)

        targets = captions[:, 1:]
        packed_preds = pack_padded_sequence(predictions, decode_lengths.cpu(), batch_first=True, enforce_sorted=True).data
        packed_targets = pack_padded_sequence(targets, decode_lengths.cpu(), batch_first=True, enforce_sorted=True).data
        return packed_preds, packed_targets, decode_lengths, alphas

class NWPUCaptioner(nn.Module):
    def __init__(self, vocab_size: int, config: Dict, pretrained: bool = True):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            model_name=config["model_name"],
            pretrained=pretrained,
            use_lora=True,
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
        )
        self.decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            decoder_dim=config["decoder_dim"],
            encoder_dim=self.encoder.feature_dim,
            attention_dim=config["attention_dim"],
            dropout=config["dropout"],
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)

    def forward(self, images, captions, lengths, img_indices: Optional[torch.Tensor] = None):
        """ConvNeXt forward pass runs ONCE per unique image, then features are expanded for all captions."""
        features = self.encoder(images)  # (B, 64, 768)
        if img_indices is not None:
            features = features[img_indices]  # (5B, 64, 768)
        return self.decoder(features, captions, lengths)

    def compute_loss(self, packed_preds, packed_targets, decode_lengths, alphas, lambda_att: float = 1.0):
        ce_loss = self.criterion(packed_preds, packed_targets)
        max_len = alphas.size(1)
        mask = (torch.arange(max_len, device=alphas.device).unsqueeze(0) < decode_lengths.unsqueeze(1)).unsqueeze(2).float()
        alpha_sum = (alphas * mask).sum(dim=1)
        att_loss = ((1.0 - alpha_sum) ** 2).mean()
        return ce_loss + lambda_att * att_loss, ce_loss, att_loss

    def save_lean_checkpoint(self, filepath: Path, epoch: int, val_loss: float, vocab: Vocabulary, config: Dict):
        """Saves only trainable parameters (LoRA + Decoder) to reduce checkpoint from 120MB down to ~12MB."""
        trainable_weights = {n: p.data.cpu() for n, p in self.named_parameters() if p.requires_grad}
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "trainable_state_dict": trainable_weights,
                "vocab_word2idx": vocab.word2idx,
                "config": config,
            },
            filepath,
        )

    def load_lean_checkpoint(self, filepath: Path, device: torch.device):
        checkpoint = torch.load(filepath, map_location=device)
        state_dict = checkpoint.get("trainable_state_dict") or checkpoint.get("model_state_dict")
        self.load_state_dict(state_dict, strict=False)
        return checkpoint

    @torch.no_grad()
    def generate(self, image: torch.Tensor, vocab: Vocabulary, max_len: int = 30, beam_size: int = 1, length_penalty_alpha: float = 0.7):
        """Autoregressive caption generation with Greedy or True Beam Search."""
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.parameters()).device
        features = self.encoder(image.to(device))
        grid_size = int(features.size(1) ** 0.5)

        # BRANCH 1: Fast Greedy (k == 1)
        if beam_size == 1:
            h, c = self.decoder.init_hidden_state(features)
            prev_word = torch.tensor([Vocabulary.START_IDX], dtype=torch.long, device=device)
            generated_ids, att_weights = [], []

            for _ in range(max_len):
                logits, h, c, alpha = self.decoder.step(prev_word, features, h, c)
                logits[:, Vocabulary.PAD_IDX] = -float("inf")
                logits[:, Vocabulary.START_IDX] = -float("inf")
                next_id = int(logits.argmax(dim=1).item())

                if next_id == Vocabulary.END_IDX:
                    break

                generated_ids.append(next_id)
                att_weights.append(alpha.squeeze(0).view(grid_size, grid_size).cpu())
                prev_word = torch.tensor([next_id], dtype=torch.long, device=device)

            tokens = vocab.decode_to_tokens(generated_ids)
            caption_str = vocab.decode(generated_ids)
            att_tensor = torch.stack(att_weights, dim=0) if att_weights else torch.empty((0, grid_size, grid_size))
            return caption_str, tokens, att_tensor

        # BRANCH 2: True Beam Search (k > 1) with Length Normalization
        k = beam_size
        h, c = self.decoder.init_hidden_state(features)
        k_features = features.expand(k, -1, -1)
        k_h = h.expand(k, -1)
        k_c = c.expand(k, -1)

        seqs = torch.full((k, 1), Vocabulary.START_IDX, dtype=torch.long, device=device)
        top_k_scores = torch.zeros(k, 1, device=device)
        beam_att_history: List[torch.Tensor] = []

        completed_seqs: List[List[int]] = []
        completed_scores: List[float] = []
        completed_alphas: List[torch.Tensor] = []

        for step in range(max_len):
            words = seqs[:, -1]
            logits, k_h, k_c, alpha = self.decoder.step(words, k_features, k_h, k_c)

            log_probs = F.log_softmax(logits, dim=1)
            log_probs[:, Vocabulary.PAD_IDX] = -float("inf")
            log_probs[:, Vocabulary.START_IDX] = -float("inf")
            step_scores = top_k_scores.expand_as(log_probs) + log_probs

            if step == 0:
                top_k_scores, top_k_words = step_scores[0].topk(k, 0, True, True)
                prev_beam_indices = torch.zeros(k, dtype=torch.long, device=device)
            else:
                top_k_scores, top_k_flat = step_scores.view(-1).topk(k, 0, True, True)
                prev_beam_indices = top_k_flat // self.decoder.vocab_size
                top_k_words = top_k_flat % self.decoder.vocab_size

            top_k_scores = top_k_scores.unsqueeze(1)
            seqs = torch.cat([seqs[prev_beam_indices], top_k_words.unsqueeze(1)], dim=1)
            k_h = k_h[prev_beam_indices]
            k_c = k_c[prev_beam_indices]

            step_alpha_grid = alpha[prev_beam_indices].view(k, grid_size, grid_size).cpu()
            if step == 0:
                beam_att_history = [step_alpha_grid]
            else:
                beam_att_history = [hist[prev_beam_indices.cpu()] for hist in beam_att_history]
                beam_att_history.append(step_alpha_grid)

            incomplete_indices = []
            for beam_idx, word in enumerate(top_k_words):
                if word.item() == Vocabulary.END_IDX:
                    token_ids = seqs[beam_idx].tolist()[1:-1]
                    seq_len = max(len(token_ids), 1)
                    norm_score = top_k_scores[beam_idx].item() / (seq_len ** length_penalty_alpha)
                    completed_seqs.append(token_ids)
                    completed_scores.append(norm_score)
                    num_words = len(token_ids)
                    if beam_att_history and num_words > 0:
                        completed_alphas.append(torch.stack([h_s[beam_idx] for h_s in beam_att_history[:num_words]], dim=0))
                    else:
                        completed_alphas.append(torch.empty((0, grid_size, grid_size)))
                else:
                    incomplete_indices.append(beam_idx)

            k -= (len(top_k_words) - len(incomplete_indices))
            if k == 0:
                break

            inc_idx = torch.tensor(incomplete_indices, dtype=torch.long, device=device)
            seqs = seqs[inc_idx]
            k_h = k_h[inc_idx]
            k_c = k_c[inc_idx]
            k_features = k_features[:k]
            top_k_scores = top_k_scores[inc_idx]
            beam_att_history = [hist[inc_idx.cpu()] for hist in beam_att_history]

        if completed_seqs:
            best_idx = completed_scores.index(max(completed_scores))
            best_token_ids = completed_seqs[best_idx]
            best_att = completed_alphas[best_idx]
        else:
            best_token_ids = seqs[0].tolist()[1:]
            num_words = len(best_token_ids)
            if beam_att_history and num_words > 0:
                best_att = torch.stack([h_s[0] for h_s in beam_att_history[:num_words]], dim=0)
            else:
                best_att = torch.empty((0, grid_size, grid_size))

        tokens = vocab.decode_to_tokens(best_token_ids)
        caption_str = vocab.decode(best_token_ids)
        return caption_str, tokens, best_att

# ==============================================================================
# INLINE FALLBACK METRICS (True Corpus BLEU & ROUGE-L)
# ==============================================================================
def get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

def compute_corpus_bleu(hypotheses: List[List[str]], references_list: List[List[List[str]]], max_order: int = 4) -> Dict[str, float]:
    """Computes True Corpus BLEU (Papineni et al.) accumulating counts across test set."""
    total_hyp_len, total_ref_len = 0, 0
    clipped_matches_corpus = [0] * max_order
    total_hyp_ngrams_corpus = [0] * max_order

    for hyp, refs in zip(hypotheses, references_list):
        hyp_len = len(hyp)
        total_hyp_len += hyp_len
        ref_lens = [len(r) for r in refs]
        closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - hyp_len), r_len))
        total_ref_len += closest_ref_len

        for n in range(1, max_order + 1):
            hyp_ngrams = get_ngrams(hyp, n)
            total_hyp_ngrams_corpus[n - 1] += max(len(hyp) - n + 1, 0)
            max_ref: Dict[Tuple[str, ...], int] = {}
            for ref in refs:
                for ng, cnt in get_ngrams(ref, n).items():
                    max_ref[ng] = max(max_ref.get(ng, 0), cnt)
            clipped_matches_corpus[n - 1] += sum(min(cnt, max_ref.get(ng, 0)) for ng, cnt in hyp_ngrams.items())

    bp = 1.0 if total_hyp_len > total_ref_len else math.exp(1.0 - (total_ref_len / max(total_hyp_len, 1)))
    scores = {}
    for n in range(1, max_order + 1):
        weights = [1.0 / n] * n
        precisions = [
            (clipped_matches_corpus[i] + 1e-9) / (total_hyp_ngrams_corpus[i] + 1e-9)
            for i in range(n)
        ]
        bleu_n = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions))) * 100.0
        scores[f"BLEU-{n}"] = min(max(bleu_n, 0.0), 100.0)
    return scores

def compute_sentence_rouge_l(hypothesis: List[str], references: List[List[str]], beta: float = 1.2) -> float:
    m = len(hypothesis)
    if m == 0 or not references:
        return 0.0
    best_f = 0.0
    for ref in references:
        n = len(ref)
        if n == 0:
            continue
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev + 1 if hypothesis[i - 1] == ref[j - 1] else max(dp[j], dp[j - 1])
                prev = temp
        lcs = dp[n]
        rec = lcs / n
        prec = lcs / m
        denom = rec + (beta ** 2) * prec
        f = ((1.0 + beta ** 2) * rec * prec) / denom if denom > 0 else 0.0
        best_f = max(best_f, f)
    return best_f

# ==============================================================================
# BENCHMARK EVALUATION (Standard pycocoevalcap + PTBTokenizer)
# ==============================================================================
def evaluate_benchmark_metrics(hypotheses_dict: Dict[str, str], references_dict: Dict[str, List[str]]) -> Dict[str, float]:
    """Computes exact literature-comparable benchmark metrics via pycocoevalcap."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge

        gts = {str(k): [{"caption": r} if isinstance(r, str) else r for r in v] for k, v in references_dict.items()}
        res = {str(k): [{"caption": v}] for k, v in hypotheses_dict.items()}

        # Standard COCO evaluation protocol runs captions through PTBTokenizer when Java is available
        if shutil.which("java") is not None:
            try:
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
                tok = PTBTokenizer()
                gts = tok.tokenize(gts)
                res = tok.tokenize(res)
            except Exception as tok_err:
                print(f"PTBTokenizer failed ({tok_err}); falling back to raw-string tokenization.")
                gts = {str(k): v for k, v in references_dict.items()}
                res = {str(k): [v] for k, v in hypotheses_dict.items()}
        else:
            gts = {str(k): v for k, v in references_dict.items()}
            res = {str(k): [v] for k, v in hypotheses_dict.items()}

        results = {}
        bleu_scores, _ = Bleu(4).compute_score(gts, res, verbose=0)
        for i in range(4):
            results[f"BLEU-{i + 1}"] = float(bleu_scores[i] * 100.0)

        # CIDEr reported unscaled (e.g. 0.85 rather than 85.0) per remote sensing literature standard
        cider_score, _ = Cider().compute_score(gts, res)
        results["CIDEr"] = float(cider_score)

        rouge_score, _ = Rouge().compute_score(gts, res)
        results["ROUGE-L"] = float(rouge_score * 100.0)

        if shutil.which("java") is not None:
            try:
                from pycocoevalcap.meteor.meteor import Meteor
                meteor_score, _ = Meteor().compute_score(gts, res)
                results["METEOR"] = float(meteor_score * 100.0)
            except Exception as e:
                print(f"METEOR computation failed: {e}")

        return results
    except Exception as e:
        print(f"pycocoevalcap fallback ({e}). Running internal corpus metrics...")
        hyp_toks = [h.strip().split() for h in hypotheses_dict.values()]
        ref_toks = [[r.strip().split() for r in refs] for refs in references_dict.values()]
        scores = compute_corpus_bleu(hyp_toks, ref_toks)
        rouge_scores = [compute_sentence_rouge_l(h, r) for h, r in zip(hyp_toks, ref_toks)]
        scores["ROUGE-L"] = (sum(rouge_scores) / len(rouge_scores)) * 100.0
        scores["CIDEr"] = 0.0
        return scores

# ==============================================================================
# ATTENTION HEATMAP VISUALIZER
# ==============================================================================
def visualize_attention(image, tokens, attention_maps, save_path=None, cols=4):
    pil_img = image.convert("RGB")
    w, h = pil_img.size
    total = len(tokens) + 1
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes_flat[0].imshow(pil_img)
    axes_flat[0].set_title("Original Image", fontsize=10, fontweight="bold")
    axes_flat[0].axis("off")

    if len(tokens) > 0 and attention_maps.dim() == 3:
        upsampled = F.interpolate(attention_maps.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False).squeeze(1).numpy()
        for i, word in enumerate(tokens):
            ax = axes_flat[i + 1]
            ax.imshow(pil_img)
            att = upsampled[i]
            if att.max() > att.min():
                att = (att - att.min()) / (att.max() - att.min())
            ax.imshow(att, cmap="jet", alpha=0.55)
            ax.set_title(f"'{word}'", fontsize=10, fontweight="bold")
            ax.axis("off")

    for j in range(total, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig

# ==============================================================================
# MAIN TRAINING & EVALUATION WORKFLOW
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Device: {device} | AMP: {device.type == 'cuda'}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print(f"=======================================================\n")

    vocab = Vocabulary.load(DATASET_DIR / "vocab.json")
    print(f"Vocabulary loaded: {len(vocab):,} tokens")

    train_dataset = NWPUTorchDataset(DATASET_DIR, split="train", vocab=vocab, mode="train")
    val_dataset = NWPUTorchDataset(DATASET_DIR, split="val", vocab=vocab, mode="train")
    val_eval_dataset = NWPUTorchDataset(DATASET_DIR, split="val", vocab=vocab, mode="eval")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=caption_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=caption_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train images: {len(train_dataset):,} (1 encoder pass/image, {len(train_loader)} batches/epoch)")
    print(f"Val images:   {len(val_dataset):,} ({len(val_loader)} batches)")

    model = NWPUCaptioner(vocab_size=len(vocab), config=CONFIG, pretrained=True).to(device)
    model.encoder.print_trainable_parameters()

    lora_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
    optimizer = AdamW([{"params": lora_params, "lr": CONFIG["lr_encoder"]}, {"params": dec_params, "lr": CONFIG["lr_decoder"]}], weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_model_path = CHECKPOINT_DIR / "best_model.pt"

    print("\n--- STARTING TRAINING (1-Pass Batched Feature Expansion) ---")
    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_loss, train_ce, train_att = 0.0, 0.0, 0.0

        for step, (images, captions, lengths, img_indices) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            img_indices = img_indices.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    preds, targets, d_lens, alphas = model(images, captions, lengths, img_indices=img_indices)
                    loss, ce, att = model.compute_loss(preds, targets, d_lens, alphas, lambda_att=CONFIG["lambda_att"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                preds, targets, d_lens, alphas = model(images, captions, lengths, img_indices=img_indices)
                loss, ce, att = model.compute_loss(preds, targets, d_lens, alphas, lambda_att=CONFIG["lambda_att"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CONFIG["grad_clip"])
                optimizer.step()

            train_loss += loss.item()
            train_ce += ce.item()
            train_att += att.item()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{CONFIG['epochs']}] Step [{step + 1}/{len(train_loader)}] | Loss: {loss.item():.4f} (CE: {ce.item():.4f}, Att: {att.item():.4f})")

        # Validation
        model.eval()
        val_loss, val_ce, val_att = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, captions, lengths, img_indices in val_loader:
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                img_indices = img_indices.to(device, non_blocking=True)

                preds, targets, d_lens, alphas = model(images, captions, lengths, img_indices=img_indices)
                loss, ce, att = model.compute_loss(preds, targets, d_lens, alphas, lambda_att=CONFIG["lambda_att"])
                val_loss += loss.item()
                val_ce += ce.item()
                val_att += att.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} complete in {time.time() - t0:.1f}s | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # Qualitative peek
        sample_img_t, sample_refs, _, _, _ = val_eval_dataset[0]
        peek_cap, _, _ = model.generate(sample_img_t, vocab=vocab, max_len=25, beam_size=1)
        print(f"  Sample Prediction: \"{peek_cap}\"")
        print(f"  Sample Reference:  \"{sample_refs[0]}\"")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_lean_checkpoint(best_model_path, epoch=epoch, val_loss=val_loss, vocab=vocab, config=CONFIG)
            print(f"  ==> Saved new best lean checkpoint (~12MB) to {best_model_path} (val_loss: {val_loss:.4f})")

    # ==============================================================================
    # FINAL BENCHMARK EVALUATION ON TEST SET (True Beam Search k=3)
    # ==============================================================================
    print(f"\n--- RUNNING BENCHMARK TEST EVALUATION (Beam Size: {CONFIG['beam_size']}) ---")
    test_dataset = NWPUTorchDataset(DATASET_DIR, split="test", vocab=vocab, mode="eval")
    print(f"Loading best checkpoint from {best_model_path}...")
    model.load_lean_checkpoint(best_model_path, device=device)
    model.eval()

    hypotheses_dict: Dict[str, str] = {}
    references_dict: Dict[str, List[str]] = {}
    vis_dir = REPORTS_DIR / "heatmaps"
    vis_dir.mkdir(parents=True, exist_ok=True)

    num_eval = min(CONFIG["max_eval_samples"], len(test_dataset)) if CONFIG.get("max_eval_samples") else len(test_dataset)
    print(f"Generating captions for {num_eval:,} test images with beam search (k={CONFIG['beam_size']})...")
    start_eval_time = time.time()
    for idx in range(num_eval):
        img_t, raw_refs, imgid, img_path, class_name = test_dataset[idx]
        cap_str, tokens, att_maps = model.generate(
            img_t,
            vocab=vocab,
            max_len=CONFIG["max_len"],
            beam_size=CONFIG["beam_size"],
        )
        hypotheses_dict[str(imgid)] = cap_str
        references_dict[str(imgid)] = raw_refs

        if idx < CONFIG["num_visualizations"]:
            pil_img = Image.open(img_path).convert("RGB")
            out_vis = vis_dir / f"test_{idx + 1}_{class_name}_att.png"
            visualize_attention(pil_img, tokens, att_maps, save_path=out_vis)

        if (idx + 1) % 250 == 0 or (idx + 1) == num_eval:
            rate = (idx + 1) / (time.time() - start_eval_time)
            print(f"Evaluated [{idx + 1}/{num_eval}] images ({rate:.1f} imgs/s)")

    metrics = evaluate_benchmark_metrics(hypotheses_dict, references_dict)
    print("\n" + "=" * 55)
    print("TEST EVALUATION BENCHMARK METRICS (Standard pycocoevalcap):")
    print("=" * 55)
    for m, score in metrics.items():
        if m == "CIDEr":
            print(f"  {m:<10}: {score:.4f} (unscaled)")
        else:
            print(f"  {m:<10}: {score:.2f}%")
    print("=" * 55)

    # Save metrics JSON
    with open(REPORTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {REPORTS_DIR / 'test_metrics.json'}")
    print("Training, beam-search evaluation, and visualization completed successfully!")

if __name__ == "__main__":
    main()
