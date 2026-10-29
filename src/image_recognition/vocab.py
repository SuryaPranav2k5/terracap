"""Word-level vocabulary and tokenizer for NWPU-Caption."""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Union


class Vocabulary:
    """Word-level vocabulary with canonical special tokens."""

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
        self.word_freqs: Dict[str, int] = {}
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.START_TOKEN: self.START_IDX,
            self.END_TOKEN: self.END_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self) -> int:
        return len(self.word2idx)

    def __getitem__(self, item: Union[str, int]) -> Union[int, str]:
        if isinstance(item, str):
            return self.word2idx.get(item, self.UNK_IDX)
        elif isinstance(item, int):
            return self.idx2word.get(item, self.UNK_TOKEN)
        raise TypeError(f"Invalid item type: {type(item)}")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize sentence into lowercase alphanumeric words."""
        return re.findall(r"\b\w+\b", text.lower())

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def build_from_annotations(
        self,
        annotations_path: Union[str, Path] = "dataset/dataset_nwpu.json",
        split: Optional[str] = "train",
        min_freq: int = 1,
    ) -> None:
        """Build vocabulary strictly from specified split (default: 'train')."""
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        counter: Counter = Counter()
        for class_name, entries in data.items():
            for entry in entries:
                if split is not None and entry.get("split") != split:
                    continue
                for key in ["raw", "raw_1", "raw_2", "raw_3", "raw_4"]:
                    if key in entry:
                        counter.update(self.tokenize(entry[key]))

        self._add_special_tokens()
        self.word_freqs = dict(counter)

        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.word2idx:
                self.add_word(word)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Converts text sentence into a list of token indices."""
        tokens = self.tokenize(text)
        indices = [self[tok] for tok in tokens]
        if add_special_tokens:
            indices = [self.START_IDX] + indices + [self.END_IDX]
        return indices

    def decode_to_tokens(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Converts token indices back into a list of word tokens."""
        words: List[str] = []
        for idx in indices:
            if idx == self.END_IDX and skip_special:
                break
            if skip_special and idx in (self.PAD_IDX, self.START_IDX):
                continue
            words.append(self[idx])
        return words

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Converts token indices back into a sentence string."""
        return " ".join(self.decode_to_tokens(indices, skip_special=skip_special))


    def save(self, filepath: Union[str, Path]) -> None:
        """Saves vocabulary mapping to a JSON file."""
        data = {
            "word2idx": self.word2idx,
            "word_freqs": self.word_freqs,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Loads vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        vocab.word_freqs = data.get("word_freqs", {})
        return vocab


def build_and_save_vocab(
    json_path: str = "dataset/dataset_nwpu.json",
    out_path: str = "dataset/vocab.json",
    split: str = "train",
    min_freq: int = 1,
) -> Vocabulary:
    """Helper to build vocabulary from train split and save to disk."""
    vocab = Vocabulary()
    print(f"Building vocabulary from '{split}' split of {json_path}...")
    vocab.build_from_annotations(annotations_path=json_path, split=split, min_freq=min_freq)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vocab.save(out_path)
    print(f"Vocabulary successfully built: {len(vocab):,} tokens (including 4 special tokens).")
    print(f"Saved to: {out_path}")
    return vocab


if __name__ == "__main__":
    build_and_save_vocab()
