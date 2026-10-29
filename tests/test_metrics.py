"""Tests for Image Captioning Evaluation Metrics."""

from image_recognition.metrics import (
    compute_corpus_bleu,
    compute_sentence_rouge_l,
    evaluate_corpus,
    get_ngrams,
    lcs_length,
)


def test_ngrams_and_lcs():
    tokens = ["many", "airplanes", "are", "parked", "on", "the", "runway"]
    unigrams = get_ngrams(tokens, 1)
    assert len(unigrams) == 7
    assert unigrams[("airplanes",)] == 1

    bigrams = get_ngrams(tokens, 2)
    assert len(bigrams) == 6
    assert bigrams[("many", "airplanes")] == 1

    assert lcs_length(["a", "b", "c", "d"], ["a", "c", "d", "e"]) == 3
    assert lcs_length([], ["a"]) == 0


def test_true_corpus_bleu():
    hypotheses = [
        ["there", "are", "many", "airplanes", "on", "the", "airport"],
        ["a", "green", "baseball", "field"],
    ]
    references_list = [
        [
            ["there", "are", "several", "airplanes", "on", "the", "airport"],
            ["many", "airplanes", "are", "parked", "at", "an", "airport"],
        ],
        [
            ["a", "large", "green", "baseball", "field"],
            ["a", "baseball", "field", "surrounded", "by", "trees"],
        ],
    ]

    bleu_scores = compute_corpus_bleu(hypotheses, references_list)
    assert "BLEU-1" in bleu_scores
    assert "BLEU-2" in bleu_scores
    assert "BLEU-3" in bleu_scores
    assert "BLEU-4" in bleu_scores

    assert 0.0 < bleu_scores["BLEU-1"] <= 100.0
    assert 0.0 < bleu_scores["BLEU-2"] <= bleu_scores["BLEU-1"]
    assert 0.0 <= bleu_scores["BLEU-4"] <= bleu_scores["BLEU-2"]


def test_rouge_l():
    candidate = ["there", "is", "a", "green", "baseball", "field"]
    references = [
        ["there", "is", "a", "large", "green", "baseball", "field"],
        ["a", "baseball", "field", "surrounded", "by", "trees"],
    ]

    rouge = compute_sentence_rouge_l(candidate, references)
    assert 0.0 < rouge <= 1.0, f"Unexpected ROUGE-L: {rouge}"

    perfect_rouge = compute_sentence_rouge_l(candidate, [candidate])
    assert abs(perfect_rouge - 1.0) < 1e-4, f"Expected 1.0 for identical match, got {perfect_rouge}"


def test_unified_corpus_evaluation_with_pycocoevalcap():
    hypotheses = [
        "there are airplanes on the airport",
        "a green baseball field",
    ]
    references_list = [
        ["there are many airplanes on the airport", "several airplanes parked on the runway"],
        ["a green baseball field surrounded by trees", "a sports baseball field"],
    ]

    scores = evaluate_corpus(hypotheses, references_list)
    print("Evaluated scores:", scores)
    assert "BLEU-1" in scores
    assert "BLEU-2" in scores
    assert "BLEU-3" in scores
    assert "BLEU-4" in scores
    assert "ROUGE-L" in scores
    assert "CIDEr" in scores
    for name, val in scores.items():
        assert 0.0 <= val, f"Score {name} should be >= 0: {val}"


if __name__ == "__main__":
    test_ngrams_and_lcs()
    test_true_corpus_bleu()
    test_rouge_l()
    test_unified_corpus_evaluation_with_pycocoevalcap()
    print("All metrics tests passed successfully!")
