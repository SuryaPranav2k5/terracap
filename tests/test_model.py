"""Unit tests for ConvNeXt-Tiny + LoRA + Spatial Attention LSTM Captioning Model."""

import torch
import torch.nn as nn

from image_recognition.model import (
    AttentionLSTMDecoder,
    BahdanauAttention,
    ConvNeXtEncoder,
    NWPUCaptioner,
)
from image_recognition.vocab import Vocabulary


def test_encoder_lora_and_shapes():
    print("Testing ConvNeXtEncoder feature extraction and LoRA parameter status...")
    # Test with pretrained=False to avoid network download during fast unit tests
    encoder = ConvNeXtEncoder(pretrained=False, use_lora=True, lora_r=16, lora_alpha=32)

    # Verify input (2, 3, 256, 256) yields (2, 64, 768)
    dummy_imgs = torch.randn(2, 3, 256, 256)
    features = encoder(dummy_imgs)

    assert features.dim() == 3, f"Expected 3D features, got {features.shape}"
    assert features.size(0) == 2, f"Batch size mismatch: {features.size(0)}"
    assert features.size(1) == 64, f"Expected 64 spatial tokens (8x8), got {features.size(1)}"
    assert features.size(2) == 768, f"Expected channel dim 768, got {features.size(2)}"

    # Check that base parameters are frozen while LoRA parameters require grad
    lora_param_count = 0
    frozen_base_count = 0
    for name, param in encoder.named_parameters():
        if "lora" in name:
            assert param.requires_grad is True, f"LoRA param {name} should require grad"
            lora_param_count += 1
        else:
            assert param.requires_grad is False, f"Base param {name} should be frozen"
            frozen_base_count += 1

    assert lora_param_count > 0, "No LoRA parameters found"
    assert frozen_base_count > 0, "No frozen base parameters found"
    print(f"  Encoder shapes and LoRA freeze verified ({lora_param_count} LoRA tensors, {frozen_base_count} frozen base tensors).")


def test_bahdanau_attention():
    print("Testing BahdanauAttention mechanism...")
    attention = BahdanauAttention(encoder_dim=768, decoder_dim=512, attention_dim=512)

    dummy_enc = torch.randn(4, 64, 768)
    dummy_hidden = torch.randn(4, 512)

    context, alpha = attention(dummy_enc, dummy_hidden)

    assert context.shape == (4, 768), f"Context shape mismatch: {context.shape}"
    assert alpha.shape == (4, 64), f"Alpha shape mismatch: {alpha.shape}"

    # Alpha must be valid probability distribution (summing to 1 over spatial tokens)
    alpha_sums = alpha.sum(dim=1)
    assert torch.allclose(alpha_sums, torch.ones_like(alpha_sums), atol=1e-5), "Alphas do not sum to 1.0"
    print("  Attention context and softmax distribution verified.")


def test_attention_lstm_decoder_forward():
    print("Testing AttentionLSTMDecoder unrolling and packing...")
    vocab_size = 100
    decoder = AttentionLSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=256,
        decoder_dim=512,
        encoder_dim=768,
        attention_dim=256,
    )

    batch_size = 4
    encoder_out = torch.randn(batch_size, 64, 768)

    # Synthetic sorted sequences: lengths [7, 6, 5, 4]
    lengths = torch.tensor([7, 6, 5, 4], dtype=torch.long)
    captions = torch.zeros(batch_size, 7, dtype=torch.long)
    captions[:, 0] = Vocabulary.START_IDX
    for i, l in enumerate(lengths):
        captions[i, 1 : l - 1] = torch.randint(4, vocab_size, (l - 2,))
        captions[i, l - 1] = Vocabulary.END_IDX

    packed_preds, packed_targets, decode_lengths, alphas = decoder(encoder_out, captions, lengths)

    total_unpadded_tokens = (lengths - 1).sum().item()
    assert packed_preds.shape == (total_unpadded_tokens, vocab_size), f"Preds shape mismatch: {packed_preds.shape}"
    assert packed_targets.shape == (total_unpadded_tokens,), f"Targets shape mismatch: {packed_targets.shape}"
    assert decode_lengths.tolist() == [6, 5, 4, 3]
    assert alphas.shape == (batch_size, 6, 64)
    print("  Decoder unrolling and packed sequences verified.")


def test_captioner_end_to_end_and_backward():
    print("Testing NWPUCaptioner end-to-end forward, loss, and gradient backprop...")
    vocab = Vocabulary()
    # Add dummy words
    for w in ["an", "airplane", "is", "parked", "on", "the", "runway", "near", "terminal"]:
        vocab.add_word(w)

    model = NWPUCaptioner(
        vocab_size=len(vocab),
        pretrained=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        embed_dim=128,
        decoder_dim=256,
        attention_dim=128,
    )

    batch_size = 2
    dummy_imgs = torch.randn(batch_size, 3, 256, 256)
    lengths = torch.tensor([6, 5], dtype=torch.long)
    captions = torch.zeros(batch_size, 6, dtype=torch.long)
    captions[0] = torch.tensor([1, 4, 5, 6, 7, 2])
    captions[1, :5] = torch.tensor([1, 8, 9, 10, 2])

    packed_preds, packed_targets, decode_lengths, alphas = model(dummy_imgs, captions, lengths)
    total_loss, ce_loss, att_loss = model.compute_loss(
        packed_preds, packed_targets, decode_lengths, alphas, lambda_att=1.0
    )

    assert total_loss.item() > 0.0
    assert ce_loss.item() > 0.0
    assert att_loss.item() >= 0.0

    # Test backward pass
    total_loss.backward()

    # Verify gradients: LoRA and decoder parameters should have non-zero gradients
    lora_has_grad = False
    for name, param in model.encoder.named_parameters():
        if "lora" in name:
            assert param.grad is not None, f"Expected grad for {name}"
            lora_has_grad = True
        else:
            assert param.grad is None, f"Frozen parameter {name} received grad!"

    assert lora_has_grad, "LoRA parameters did not receive gradients"

    # Decoder must have gradients
    decoder_has_grad = False
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Expected grad for decoder param {name}"
            decoder_has_grad = True

    assert decoder_has_grad, "Decoder parameters did not receive gradients"
    print("  Backward pass verified: gradients only flow to LoRA adapters and Decoder!")


def test_greedy_generation_and_heatmaps():
    print("Testing NWPUCaptioner greedy generation and attention maps...")
    vocab = Vocabulary()
    for w in ["an", "airplane", "on", "airport"]:
        vocab.add_word(w)

    model = NWPUCaptioner(
        vocab_size=len(vocab),
        pretrained=False,
        use_lora=True,
        lora_r=8,
        embed_dim=128,
        decoder_dim=256,
    )

    dummy_img = torch.randn(3, 256, 256)
    caption_str, tokens, attention_maps = model.generate(dummy_img, vocab=vocab, max_len=10, beam_size=1)

    assert isinstance(caption_str, str)
    assert isinstance(tokens, list)
    assert attention_maps.dim() == 3  # (num_tokens, H_feat, W_feat)
    if len(tokens) > 0:
        assert attention_maps.shape == (len(tokens), 8, 8), f"Expected shape ({len(tokens)}, 8, 8), got {attention_maps.shape}"
    print(f"  Generation verified: '{caption_str}' with attention maps shape {attention_maps.shape}.")


def test_beam_search_generation():
    print("Testing NWPUCaptioner true beam search (k=3) with length normalization...")
    vocab = Vocabulary()
    for w in ["an", "airplane", "on", "airport", "runway"]:
        vocab.add_word(w)

    model = NWPUCaptioner(
        vocab_size=len(vocab),
        pretrained=False,
        use_lora=True,
        lora_r=8,
        embed_dim=128,
        decoder_dim=256,
    )

    dummy_img = torch.randn(3, 256, 256)
    caption_str, tokens, attention_maps = model.generate(
        dummy_img, vocab=vocab, max_len=10, beam_size=3
    )

    assert isinstance(caption_str, str)
    assert isinstance(tokens, list)
    assert attention_maps.dim() == 3
    if len(tokens) > 0:
        assert attention_maps.shape == (len(tokens), 8, 8)
    print(f"  Beam search (k=3) verified: '{caption_str}' (tokens: {len(tokens)})")


def test_lean_checkpoint_saving():
    print("Testing lean checkpoint saving (excluding frozen backbone)...")
    from pathlib import Path
    import tempfile

    vocab = Vocabulary()
    model = NWPUCaptioner(
        vocab_size=len(vocab),
        pretrained=False,
        use_lora=True,
        lora_r=8,
        embed_dim=128,
        decoder_dim=256,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = Path(tmp_dir) / "test_best.pt"
        model.save_trainable_checkpoint(chk_path, epoch=1, val_loss=2.34, vocab=vocab)
        assert chk_path.exists()
        loaded = torch.load(chk_path)
        trainable_dict = loaded["trainable_state_dict"]

        # Ensure no frozen backbone weights are stored
        for key in trainable_dict.keys():
            assert "lora" in key or "decoder" in key or "criterion" in key, f"Frozen key {key} leaked into checkpoint!"

        print(f"  Lean checkpoint verified ({len(trainable_dict)} trainable tensors stored, frozen base omitted).")


if __name__ == "__main__":
    test_encoder_lora_and_shapes()
    test_bahdanau_attention()
    test_attention_lstm_decoder_forward()
    test_captioner_end_to_end_and_backward()
    test_greedy_generation_and_heatmaps()
    test_beam_search_generation()
    test_lean_checkpoint_saving()
    print("\nALL MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")

