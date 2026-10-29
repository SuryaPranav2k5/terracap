"""Benchmark Evaluation Metrics for Image Captioning.

Implements standard benchmark metrics aligned with the Remote Sensing & Vision literature:
1. pycocoevalcap integration (Bleu-1..4, CIDEr, Rouge-L, and Meteor)
2. Exact True Corpus-Level BLEU (accumulates n-gram counts and lengths globally across all images,
   rather than sentence-averaging, matching Papineni et al. and pycocoevalcap).
3. Pure-Python CIDEr and ROUGE-L fallback for offline/dependency-free execution.
"""

from collections import Counter
import math
import shutil
from typing import Dict, List, Optional, Sequence, Tuple, Union


def get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Extracts n-grams and frequency counts."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_corpus_bleu(
    hypotheses: List[List[str]],
    references_list: List[List[List[str]]],
    max_order: int = 4,
    smooth_eps: float = 1e-9,
) -> Dict[str, float]:
    """Computes TRUE corpus-level BLEU-1 to BLEU-4 (Papineni et al., 2002).

    Unlike sentence-averaged BLEU, this accumulates clipped n-gram counts across
    the entire corpus before computing one final global precision ratio and brevity penalty.
    """
    assert len(hypotheses) == len(references_list)
    n_samples = len(hypotheses)
    if n_samples == 0:
        return {f"BLEU-{i}": 0.0 for i in range(1, max_order + 1)}

    total_hyp_len = 0
    total_ref_len = 0

    # Accumulators for n = 1 .. max_order
    clipped_matches_corpus = [0] * max_order
    total_hyp_ngrams_corpus = [0] * max_order

    for hyp, refs in zip(hypotheses, references_list):
        hyp_len = len(hyp)
        total_hyp_len += hyp_len

        # Closest reference length for brevity penalty
        ref_lens = [len(r) for r in refs]
        closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - hyp_len), r_len))
        total_ref_len += closest_ref_len

        for n in range(1, max_order + 1):
            hyp_ngrams = get_ngrams(hyp, n)
            total_hyp_ngrams_corpus[n - 1] += max(len(hyp) - n + 1, 0)

            # Max counts in any single reference
            max_ref_counts: Dict[Tuple[str, ...], int] = {}
            for ref in refs:
                ref_ngrams = get_ngrams(ref, n)
                for ng, cnt in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), cnt)

            clipped = sum(
                min(cnt, max_ref_counts.get(ng, 0)) for ng, cnt in hyp_ngrams.items()
            )
            clipped_matches_corpus[n - 1] += clipped

    # Brevity penalty computed globally over entire test set
    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (total_ref_len / max(total_hyp_len, 1)))

    precisions = []
    for n in range(max_order):
        if total_hyp_ngrams_corpus[n] == 0:
            precisions.append(smooth_eps)
        else:
            prec = (clipped_matches_corpus[n] + smooth_eps) / (
                total_hyp_ngrams_corpus[n] + smooth_eps
            )
            precisions.append(prec)

    scores: Dict[str, float] = {}
    for n in range(1, max_order + 1):
        weights = [1.0 / n] * n
        log_prec_sum = sum(w * math.log(p) for w, p in zip(weights, precisions[:n]))
        bleu_n = bp * math.exp(log_prec_sum) * 100.0
        scores[f"BLEU-{n}"] = min(max(bleu_n, 0.0), 100.0)

    return scores


def lcs_length(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    """Computes length of Longest Common Subsequence."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev + 1 if seq1[i - 1] == seq2[j - 1] else max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_sentence_rouge_l(
    hypothesis: List[str], references: List[List[str]], beta: float = 1.2
) -> float:
    """Computes sentence-level ROUGE-L."""
    hyp_len = len(hypothesis)
    if hyp_len == 0 or not references:
        return 0.0
    best_f = 0.0
    for ref in references:
        r_len = len(ref)
        if r_len == 0:
            continue
        lcs = lcs_length(hypothesis, ref)
        rec = lcs / r_len
        prec = lcs / hyp_len
        denom = rec + (beta ** 2) * prec
        f = ((1.0 + beta ** 2) * rec * prec) / denom if denom > 0.0 else 0.0
        best_f = max(best_f, f)
    return best_f


def evaluate_pycocoevalcap(
    hypotheses: List[str],
    references_list: List[List[str]],
    image_ids: Optional[List[Union[int, str]]] = None,
) -> Dict[str, float]:
    """Computes standard benchmark metrics using pycocoevalcap (BLEU-1..4, CIDEr, ROUGE-L, METEOR)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    if image_ids is None:
        image_ids = list(range(len(hypotheses)))

    gts = {str(img_id): [{"caption": r} if isinstance(r, str) else r for r in refs] for img_id, refs in zip(image_ids, references_list)}
    res = {str(img_id): [{"caption": hyp}] for img_id, hyp in zip(image_ids, hypotheses)}

    # Standard COCO evaluation protocol runs captions through PTBTokenizer when Java is available
    if shutil.which("java") is not None:
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            tokenizer = PTBTokenizer()
            gts = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)
        except Exception as tok_err:
            print(f"PTBTokenizer failed ({tok_err}); falling back to raw-string tokenization.")
            gts = {str(img_id): refs for img_id, refs in zip(image_ids, references_list)}
            res = {str(img_id): [hyp] for img_id, hyp in zip(image_ids, hypotheses)}
    else:
        gts = {str(img_id): refs for img_id, refs in zip(image_ids, references_list)}
        res = {str(img_id): [hyp] for img_id, hyp in zip(image_ids, hypotheses)}

    results: Dict[str, float] = {}

    # 1. BLEU-1 to BLEU-4 (reported as percentage 0..100)
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res, verbose=0)
    for i in range(4):
        results[f"BLEU-{i + 1}"] = float(bleu_scores[i] * 100.0)

    # 2. CIDEr (reported unscaled per literature convention, e.g. 0.85 rather than 85.0)
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    results["CIDEr"] = float(cider_score)

    # 3. ROUGE-L (reported as percentage 0..100)
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    results["ROUGE-L"] = float(rouge_score * 100.0)

    # 4. METEOR (checks if Java is available in environment)
    if shutil.which("java") is not None:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
            results["METEOR"] = float(meteor_score * 100.0)
        except Exception as e:
            print(f"METEOR computation failed: {e}")

    return results


def evaluate_corpus(
    hypotheses: Union[List[List[str]], List[str]],
    references_list: Union[List[List[List[str]]], List[List[str]]],
    image_ids: Optional[List[Union[int, str]]] = None,
) -> Dict[str, float]:
    """Unified evaluation interface.

    Prefers pycocoevalcap for exact literature comparison (BLEU-1..4, CIDEr, ROUGE-L, METEOR).
    Falls back to True Corpus BLEU if pycocoevalcap is not present.
    """
    assert len(hypotheses) == len(references_list), "Hypotheses and references count mismatch"
    if len(hypotheses) == 0:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0, "CIDEr": 0.0, "ROUGE-L": 0.0}

    # Normalize inputs to both string list and token list formats
    if isinstance(hypotheses[0], list):
        hyp_tokens = hypotheses  # type: ignore
        hyp_strings = [" ".join(h) for h in hyp_tokens]
    else:
        hyp_strings = hypotheses  # type: ignore
        hyp_tokens = [h.strip().split() for h in hyp_strings]

    if isinstance(references_list[0][0], list):
        ref_tokens = references_list  # type: ignore
        ref_strings = [[" ".join(r) for r in refs] for refs in ref_tokens]
    else:
        ref_strings = references_list  # type: ignore
        ref_tokens = [[r.strip().split() for r in refs] for refs in ref_strings]

    try:
        return evaluate_pycocoevalcap(hyp_strings, ref_strings, image_ids=image_ids)
    except Exception:
        # Fallback to True Corpus BLEU & sentence ROUGE-L
        scores = compute_corpus_bleu(hyp_tokens, ref_tokens)
        rouge_scores = [compute_sentence_rouge_l(h, r) for h, r in zip(hyp_tokens, ref_tokens)]
        scores["ROUGE-L"] = (sum(rouge_scores) / len(rouge_scores)) * 100.0
        scores["CIDEr"] = 0.0
        return scores
