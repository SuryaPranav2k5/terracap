"""Tests true beam search generation on real dataset images."""

from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

from image_recognition.dataset import load_dataset_items
from image_recognition.model import NWPUCaptioner
from image_recognition.vocab import Vocabulary


def test_real_images_beam_search():
    print("Testing true beam search (k=3) on 3 real test images...")
    vocab = Vocabulary.load("dataset/vocab.json")
    test_items = load_dataset_items(split="test")
    assert len(test_items) > 3

    # Initialize model
    model = NWPUCaptioner(
        vocab_size=len(vocab),
        pretrained=False,
        use_lora=True,
        lora_r=8,
        embed_dim=256,
        decoder_dim=256,
        dropout=0.1,
    )
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), normalize])

    for i in range(3):
        item = test_items[i]
        pil_img = item.open_image()
        img_t = transform(pil_img)

        caption, tokens, att_maps = model.generate(
            image=img_t,
            vocab=vocab,
            max_len=15,
            beam_size=3,
        )

        print(f"\nImage {i + 1} ({item.class_name} / {item.filename}):")
        print(f"  Generated Caption: \"{caption}\"")
        print(f"  Tokens count:     {len(tokens)}")
        print(f"  Attention maps:   {att_maps.shape}")
        print(f"  Reference 1:      \"{item.captions[0]}\"")

        # Invariants:
        assert isinstance(caption, str)
        assert isinstance(tokens, list)
        assert att_maps.dim() == 3
        if len(tokens) > 0:
            assert att_maps.shape == (len(tokens), 8, 8), (
                f"Mismatch: {len(tokens)} tokens vs attention shape {att_maps.shape}"
            )
        else:
            assert att_maps.shape[0] == 0

    print("\nReal image beam search verification passed successfully!")


if __name__ == "__main__":
    test_real_images_beam_search()
