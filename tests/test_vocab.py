"""Unit tests for Vocabulary class."""

from pathlib import Path
from image_recognition.vocab import Vocabulary


def test_vocabulary_basics():
    vocab = Vocabulary()
    assert len(vocab) == 4
    assert vocab[Vocabulary.PAD_TOKEN] == Vocabulary.PAD_IDX
    assert vocab[Vocabulary.START_TOKEN] == Vocabulary.START_IDX
    assert vocab[Vocabulary.END_TOKEN] == Vocabulary.END_IDX
    assert vocab[Vocabulary.UNK_TOKEN] == Vocabulary.UNK_IDX

    # Add words
    idx1 = vocab.add_word("airplane")
    idx2 = vocab.add_word("runway")
    assert idx1 == 4
    assert idx2 == 5
    assert len(vocab) == 6

    # Test lookup
    assert vocab["airplane"] == 4
    assert vocab[4] == "airplane"
    assert vocab["nonexistent_word_xyz"] == Vocabulary.UNK_IDX

    # Test encoding / decoding
    sentence = "Airplane on runway"
    encoded = vocab.encode(sentence, add_special_tokens=True)
    assert encoded == [Vocabulary.START_IDX, 4, Vocabulary.UNK_IDX, 5, Vocabulary.END_IDX]

    decoded = vocab.decode(encoded, skip_special=True)
    assert decoded == "airplane <unk> runway"

    tokens = vocab.decode_to_tokens(encoded, skip_special=True)
    assert tokens == ["airplane", "<unk>", "runway"]



def test_saved_vocab():
    vocab_path = Path("dataset/vocab.json")
    assert vocab_path.exists(), "dataset/vocab.json should exist"
    loaded = Vocabulary.load(vocab_path)
    assert len(loaded) == 2661
    assert loaded["airplane"] != Vocabulary.UNK_IDX


if __name__ == "__main__":
    test_vocabulary_basics()
    test_saved_vocab()
    print("All vocabulary unit tests passed successfully!")
