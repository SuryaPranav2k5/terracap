"""Model architecture for Remote Sensing Image Captioning.

Composite architecture combining:
1. ConvNeXt-Tiny image encoder (via timm) with LoRA parameter-efficient fine-tuning (PEFT)
2. Spatial Attention mechanism (Bahdanau-style)
3. Attention LSTM Decoder with Deep Output Layer (Show, Attend and Tell)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import timm
import peft

from image_recognition.vocab import Vocabulary


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny visual feature extractor with LoRA PEFT.

    Extracts spatial grid feature tokens (B, H*W, C) from input images.
    When use_lora=True, freezes backbone parameters and injects trainable
    low-rank adapters into the 1x1 pointwise MLP layers (fc1, fc2).
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny.in12k_ft_in1k",
        pretrained: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.feature_dim = 768  # ConvNeXt-Tiny stage 3 output channels

        # Create backbone without classifier head
        base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        if use_lora:
            # Freeze all base model parameters
            for param in base_model.parameters():
                param.requires_grad = False

            lora_config = peft.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["fc1", "fc2"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = peft.get_peft_model(base_model, lora_config)
        else:
            self.backbone = base_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts spatial grid tokens from images.

        Args:
            images: Tensor of shape (B, 3, H_img, W_img).

        Returns:
            features: Tensor of shape (B, N, feature_dim) where N = H_feat * W_feat.
        """
        # forward_features yields (B, C, H, W)
        features = self.backbone.forward_features(images)
        b, c, h, w = features.shape
        # Permute to (B, H*W, C)
        features = features.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        return features

    def print_trainable_parameters(self) -> None:
        """Prints parameter counts."""
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable %: {100 * trainable / total:.2f}%")


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau-style) Spatial Attention.

    Computes alignment scores between spatial image tokens and the decoder hidden state:
        e_{t,i} = v_a^T tanh(W_v V_i + W_h h_t + b_a)
        alpha_{t,i} = softmax(e_{t,i})
        z_t = sum_i alpha_{t,i} V_i
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        decoder_dim: int = 512,
        attention_dim: int = 512,
    ):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.energy_proj = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates attention context and spatial weights.

        Args:
            encoder_out: Spatial visual features of shape (B, N, encoder_dim).
            decoder_hidden: Current decoder hidden state of shape (B, decoder_dim).

        Returns:
            context: Attention-weighted visual context vector of shape (B, encoder_dim).
            alpha: Attention weights over spatial locations of shape (B, N).
        """
        # (B, N, attention_dim)
        att_enc = self.encoder_proj(encoder_out)
        # (B, 1, attention_dim)
        att_dec = self.decoder_proj(decoder_hidden).unsqueeze(1)
        # (B, N)
        energy = self.energy_proj(torch.tanh(att_enc + att_dec)).squeeze(2)
        alpha = F.softmax(energy, dim=1)
        # (B, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class AttentionLSTMDecoder(nn.Module):
    """LSTM Decoder with Spatial Attention and Deep Output Layer.

    Implements the Show, Attend and Tell decoding formulation:
    - Initial hidden/cell states initialized from mean-pooled visual features.
    - Context gating via sigmoid beta-gate: beta_t = sigmoid(W_beta * h_{t-1}).
    - Step-by-step LSTM cell unrolling with Teacher Forcing during training.
    - Deep output layer combining [h_t, beta_t * z_t, word_embed] for final word prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        decoder_dim: int = 512,
        encoder_dim: int = 768,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=Vocabulary.PAD_IDX,
        )
        self.dropout = nn.Dropout(p=dropout)

        self.attention = BahdanauAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
        )

        # Context gate (Show, Attend and Tell)
        self.beta_gate = nn.Linear(decoder_dim, encoder_dim)

        # LSTM cell consumes concatenated [word_embed, gated_context]
        self.lstm_cell = nn.LSTMCell(
            input_size=embed_dim + encoder_dim,
            hidden_size=decoder_dim,
        )

        # Projections to initialize h_0 and c_0 from mean visual features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Deep Output Layer (Show, Attend and Tell Section 3.1.3)
        # Fuses decoder hidden state, gated visual context, and word embedding
        self.deep_output = nn.Linear(decoder_dim + encoder_dim + embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def init_hidden_state(
        self, encoder_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes LSTM hidden and cell states from mean-pooled image features."""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder_out))
        c = torch.tanh(self.init_c(mean_encoder_out))
        return h, c

    def step(
        self,
        word_idx: torch.Tensor,
        encoder_out: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single decoding step with deep output projection.

        Args:
            word_idx: (B,) current input word token indices.
            encoder_out: (B, N, encoder_dim) visual tokens.
            h_prev: (B, decoder_dim) previous hidden state.
            c_prev: (B, decoder_dim) previous cell state.

        Returns:
            logits: (B, vocab_size) unnormalized word scores.
            h_new: (B, decoder_dim) updated hidden state.
            c_new: (B, decoder_dim) updated cell state.
            alpha: (B, N) spatial attention weights.
        """
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

    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unrolled teacher-forcing forward pass during training.

        Args:
            encoder_out: (B, N, encoder_dim)
            captions: (B, max_len) sorted in descending order of length
            lengths: (B,) lengths including <start> and <end> tokens

        Returns:
            predictions: Tensor of shape (total_unpadded_steps, vocab_size) packed logits
            targets: Tensor of shape (total_unpadded_steps,) packed target token IDs
            decode_lengths: Tensor of shape (B,) effective decoding steps (lengths - 1)
            alphas: Tensor of shape (B, max_decode_len, N) spatial attention weights
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        decode_lengths = lengths - 1
        max_decode_len = int(decode_lengths.max().item())

        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(
            batch_size, max_decode_len, self.vocab_size, device=encoder_out.device
        )
        alphas = torch.zeros(
            batch_size, max_decode_len, num_pixels, device=encoder_out.device
        )

        for t in range(max_decode_len):
            batch_size_t = int((decode_lengths > t).sum().item())

            enc_t = encoder_out[:batch_size_t]
            h_prev = h[:batch_size_t]
            c_prev = c[:batch_size_t]
            words_t = captions[:batch_size_t, t]

            preds, h_new, c_new, alpha = self.step(words_t, enc_t, h_prev, c_prev)

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            h = torch.cat([h_new, h[batch_size_t:]], dim=0)
            c = torch.cat([c_new, c[batch_size_t:]], dim=0)

        targets = captions[:, 1:]
        packed_preds = pack_padded_sequence(
            predictions, decode_lengths.cpu(), batch_first=True, enforce_sorted=True
        ).data
        packed_targets = pack_padded_sequence(
            targets, decode_lengths.cpu(), batch_first=True, enforce_sorted=True
        ).data

        return packed_preds, packed_targets, decode_lengths, alphas


class NWPUCaptioner(nn.Module):
    """End-to-End Composite Remote Sensing Image Captioning Model.

    Combines:
    - ConvNeXt-Tiny visual encoder with LoRA adaptation
    - Bahdanau spatial attention
    - LSTM decoder with Deep Output and doubly stochastic attention regularization
    """

    def __init__(
        self,
        vocab_size: int,
        model_name: str = "convnext_tiny.in12k_ft_in1k",
        pretrained: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        embed_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            model_name=model_name,
            pretrained=pretrained,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            encoder_dim=self.encoder.feature_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
        img_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        If img_indices is provided, ConvNeXt encoder runs only ONCE for the unique images,
        and features are expanded via index selection across all captions in batch.
        """
        features = self.encoder(images)
        if img_indices is not None:
            features = features[img_indices]
        return self.decoder(features, captions, lengths)


    def compute_loss(
        self,
        packed_preds: torch.Tensor,
        packed_targets: torch.Tensor,
        decode_lengths: torch.Tensor,
        alphas: torch.Tensor,
        lambda_att: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes cross-entropy loss and doubly stochastic attention penalty."""
        ce_loss = self.criterion(packed_preds, packed_targets)

        max_decode_len = alphas.size(1)
        mask = torch.arange(max_decode_len, device=alphas.device).unsqueeze(0) < decode_lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).float()
        masked_alphas = alphas * mask

        alpha_sum = masked_alphas.sum(dim=1)
        att_loss = ((1.0 - alpha_sum) ** 2).mean()

        total_loss = ce_loss + lambda_att * att_loss
        return total_loss, ce_loss, att_loss

    def save_trainable_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        val_loss: float = 0.0,
        vocab: Optional[Vocabulary] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves only trainable parameters (LoRA adapters + Decoder) to minimize file size (~12MB vs 120MB)."""
        trainable_state_dict = {
            name: param.data.cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "trainable_state_dict": trainable_state_dict,
            "vocab_word2idx": vocab.word2idx if vocab else None,
            "config": config or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    def load_trainable_checkpoint(
        self,
        filepath: Union[str, Path],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Loads trainable weights into model (requires base backbone initialized with pretrained weights)."""
        checkpoint = torch.load(filepath, map_location=device)
        state_dict = checkpoint.get("trainable_state_dict") or checkpoint.get("model_state_dict")
        # Load with strict=False because frozen backbone weights are excluded from lean checkpoints
        self.load_state_dict(state_dict, strict=False)
        return checkpoint

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        vocab: Vocabulary,
        max_len: int = 30,
        beam_size: int = 1,
        length_penalty_alpha: float = 0.7,
    ) -> Tuple[str, List[str], torch.Tensor]:
        """Generates a caption with spatial attention maps via Greedy (k=1) or Beam Search (k>1).

        Args:
            image: Tensor of shape (3, H, W) or (1, 3, H, W).
            vocab: Vocabulary instance for token encoding/decoding.
            max_len: Maximum tokens to generate.
            beam_size: Beam search width (1 = greedy decoding, >1 = full beam search).
            length_penalty_alpha: Exponent for length normalization (default: 0.7).

        Returns:
            caption_str: Generated natural language sentence.
            tokens: List of individual word tokens.
            attention_maps: Tensor of shape (num_tokens, H_feat, W_feat) for heatmaps.
        """
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)

        device = next(self.parameters()).device
        image = image.to(device)

        features = self.encoder(image)  # (1, N, C)
        n_pixels = features.size(1)
        grid_size = int(n_pixels ** 0.5)

        # -------------------------------------------------------------
        # BRANCH 1: Fast Greedy Decoding (beam_size == 1)
        # -------------------------------------------------------------
        if beam_size == 1:
            h, c = self.decoder.init_hidden_state(features)
            prev_word = torch.tensor([Vocabulary.START_IDX], dtype=torch.long, device=device)
            generated_ids: List[int] = []
            att_weights: List[torch.Tensor] = []

            for _ in range(max_len):
                logits, h, c, alpha = self.decoder.step(prev_word, features, h, c)

                logits[:, Vocabulary.PAD_IDX] = -float("inf")
                logits[:, Vocabulary.START_IDX] = -float("inf")
                next_word_id = int(logits.argmax(dim=1).item())

                if next_word_id == Vocabulary.END_IDX:
                    break

                generated_ids.append(next_word_id)
                att_weights.append(alpha.squeeze(0).view(grid_size, grid_size).cpu())
                prev_word = torch.tensor([next_word_id], dtype=torch.long, device=device)

            tokens = vocab.decode_to_tokens(generated_ids)
            caption_str = vocab.decode(generated_ids)
            attention_maps = (
                torch.stack(att_weights, dim=0)
                if att_weights
                else torch.empty((0, grid_size, grid_size))
            )
            return caption_str, tokens, attention_maps

        # -------------------------------------------------------------
        # BRANCH 2: True Beam Search Decoding (beam_size > 1)
        # -------------------------------------------------------------
        k = beam_size
        h, c = self.decoder.init_hidden_state(features)

        # Expand visual features to k beams: (k, N, C)
        k_features = features.expand(k, -1, -1)
        k_h = h.expand(k, -1)
        k_c = c.expand(k, -1)

        # Active beams state:
        # seqs: (k, step) token sequence IDs
        # scores: (k, 1) cumulative log probabilities
        # step_alphas: list of (k, grid_size, grid_size)
        seqs = torch.full((k, 1), Vocabulary.START_IDX, dtype=torch.long, device=device)
        top_k_scores = torch.zeros(k, 1, device=device)
        beam_att_history: List[torch.Tensor] = []

        completed_seqs: List[List[int]] = []
        completed_scores: List[float] = []
        completed_alphas: List[torch.Tensor] = []

        for step in range(max_len):
            words = seqs[:, -1]  # (k,)
            logits, k_h, k_c, alpha = self.decoder.step(words, k_features, k_h, k_c)

            log_probs = F.log_softmax(logits, dim=1)
            log_probs[:, Vocabulary.PAD_IDX] = -float("inf")
            log_probs[:, Vocabulary.START_IDX] = -float("inf")

            step_scores = top_k_scores.expand_as(log_probs) + log_probs

            if step == 0:
                # First step: all k beams have identical start history, only take top-k from beam 0
                top_k_scores, top_k_words = step_scores[0].topk(k, 0, True, True)
                prev_beam_indices = torch.zeros(k, dtype=torch.long, device=device)
            else:
                # Subsequent steps: flatten all (k * vocab_size) candidates and find global top-k
                top_k_scores, top_k_flat_indices = step_scores.view(-1).topk(k, 0, True, True)
                prev_beam_indices = top_k_flat_indices // self.decoder.vocab_size
                top_k_words = top_k_flat_indices % self.decoder.vocab_size

            top_k_scores = top_k_scores.unsqueeze(1)

            # Reorder sequence history and hidden states according to winning parent beams
            seqs = torch.cat([seqs[prev_beam_indices], top_k_words.unsqueeze(1)], dim=1)
            k_h = k_h[prev_beam_indices]
            k_c = k_c[prev_beam_indices]

            # Reorder and store attention history
            step_alpha_grid = alpha[prev_beam_indices].view(k, grid_size, grid_size).cpu()
            if step == 0:
                beam_att_history = [step_alpha_grid]
            else:
                # Reorder past alpha tensors to match new parent beams
                beam_att_history = [
                    hist[prev_beam_indices.cpu()] for hist in beam_att_history
                ]
                beam_att_history.append(step_alpha_grid)

            # Separate completed beams that reached <end>
            incomplete_indices = []
            for beam_idx, word in enumerate(top_k_words):
                if word.item() == Vocabulary.END_IDX:
                    full_token_ids = seqs[beam_idx].tolist()[1:-1]  # Exclude <start> and <end>
                    seq_len = max(len(full_token_ids), 1)
                    # Length-normalized log-probability
                    norm_score = top_k_scores[beam_idx].item() / (seq_len ** length_penalty_alpha)
                    completed_seqs.append(full_token_ids)
                    completed_scores.append(norm_score)
                    # Collect attention maps for this beam
                    num_words = len(full_token_ids)
                    if beam_att_history and num_words > 0:
                        this_beam_att = torch.stack(
                            [h_slice[beam_idx] for h_slice in beam_att_history[:num_words]], dim=0
                        )
                        completed_alphas.append(this_beam_att)
                    else:
                        completed_alphas.append(torch.empty((0, grid_size, grid_size)))
                else:
                    incomplete_indices.append(beam_idx)

            k -= (len(top_k_words) - len(incomplete_indices))
            if k == 0:
                break

            # Retain active incomplete beams for next step
            inc_idx_t = torch.tensor(incomplete_indices, dtype=torch.long, device=device)
            seqs = seqs[inc_idx_t]
            k_h = k_h[inc_idx_t]
            k_c = k_c[inc_idx_t]
            k_features = k_features[:k]
            top_k_scores = top_k_scores[inc_idx_t]

            inc_idx_cpu = torch.tensor(incomplete_indices, dtype=torch.long)
            beam_att_history = [hist[inc_idx_cpu] for hist in beam_att_history]

        if completed_seqs:
            best_idx = completed_scores.index(max(completed_scores))
            best_tokens_ids = completed_seqs[best_idx]
            best_att = completed_alphas[best_idx]
        else:
            # Fallback to top incomplete beam if no beam hit <end>
            best_tokens_ids = seqs[0].tolist()[1:]
            num_words = len(best_tokens_ids)
            if beam_att_history and num_words > 0:
                best_att = torch.stack([h_slice[0] for h_slice in beam_att_history[:num_words]], dim=0)
            else:
                best_att = torch.empty((0, grid_size, grid_size))


        tokens = vocab.decode_to_tokens(best_tokens_ids)
        caption_str = vocab.decode(best_tokens_ids)
        return caption_str, tokens, best_att
