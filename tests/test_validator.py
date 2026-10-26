"""Tests for dataset validation logic."""

from PIL import Image
from image_recognition.validator import compute_dhash, hamming_distance


def test_dhash_and_hamming():
    # Create two identical test images
    img1 = Image.new("RGB", (100, 100), color=(120, 120, 120))
    img2 = Image.new("RGB", (100, 100), color=(120, 120, 120))

    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Create distinct pattern
    img3 = Image.new("RGB", (100, 100), color=(0, 0, 0))
    for x in range(50):
        for y in range(100):
            img3.putpixel((x, y), (255, 255, 255))
    h3 = compute_dhash(img3)
    assert h1 != h3


if __name__ == "__main__":
    test_dhash_and_hamming()
    print("Validator unit test passed!")
