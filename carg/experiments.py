"""Experiment I and Experiment II runners (Section 4 & 5).

* Experiment I -- comparative evaluation of Vanilla RAG, Prompt-enhanced RAG,
  and CA-RAG on CAS and PCS.
* Experiment II -- module-level ablation of CA-RAG (full model + seven
  variants + Vanilla RAG) on CAS, PCS, and KGR.

The evaluation uses the deterministic ``MockLLM`` judge, so the runs are fully
reproducible and require no external API. The reported numbers are the actual
scores computed by the metrics from generated responses; they reproduce the
paper's *qualitative* findings (CA-RAG > Prompt-enhanced > Vanilla, and
full > single-module > dual-module > Vanilla). The paper's exact figures come
from a Gemma-3-4B evaluation run and are shown alongside for reference.
"""

from typing import Dict, List, Tuple

from .baselines import PromptEnhancedRAG, VanillaRAG
from .config import INTERMEDIATE
from .judge import Judge
from .metrics import (
    cognitive_alignment_score,
    knowledge_gap_reduction,
    mean,
    prerequisite_coverage_score,
)
from .pipeline import CARAGPipeline
from .types import Response, Sample


# --------------------------------------------------------------------------- #
# KGR: simulated post-interaction gap
# --------------------------------------------------------------------------- #
def simulate_learning(sample: Sample, response: Response, judge: Judge) -> float:
    """Return the fraction of the target gap closed by ``response``.

    KGR requires the post-interaction gap vector, which is not directly
    observable offline. We simulate the learning outcome from two observable
    properties of the response: how well its difficulty matches the user
    (alignment) and whether it grounds the answer in the missing prerequisites
    (grounding). A response that is both appropriately pitched and
    prerequisite-grounded closes more of the gap.
    """
    difficulty = judge.rate_difficulty(response.text)
    alignment = cognitive_alignment_score(sample.user_level, difficulty)

    required_missing = set(sample.required_prereqs) & set(sample.missing_concepts)
    covered = {c for c in required_missing
               if judge.concept_explained(response.text, c)}
    grounding = (len(covered) / len(required_missing)) if required_missing else 1.0

    # Effective learning rate: 0.8 is a calibration constant bounding KGR < 1.
    return 0.8 * alignment * (0.5 + 0.5 * grounding)


def compute_kgr(sample: Sample, response: Response, judge: Judge) -> float:
    gap_pre = 1.0 - sample.kv[sample.target_concept]
    if gap_pre <= 0:
        raise ValueError("sample with zero pre-interaction gap")
    learning = simulate_learning(sample, response, judge)
    gap_post = max(0.0, gap_pre * (1.0 - learning))
    return knowledge_gap_reduction(gap_pre, gap_post)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def _evaluate(responses: List[Response], samples: List[Sample], judge: Judge,
              include_kgr: bool = False) -> Dict[str, float]:
    cas_values, pcs_values, kgr_values = [], [], []
    for sample, response in zip(samples, responses):
        difficulty = judge.rate_difficulty(response.text)
        cas_values.append(cognitive_alignment_score(sample.user_level, difficulty))
        pcs_values.append(prerequisite_coverage_score(
            response.text, sample.target_concept, sample.required_prereqs,
            sample.missing_concepts, judge,
        ))
        if include_kgr:
            kgr_values.append(compute_kgr(sample, response, judge))

    result = {"CAS": mean(cas_values), "PCS": mean(pcs_values)}
    if include_kgr:
        result["KGR"] = mean(kgr_values)
    return result


# --------------------------------------------------------------------------- #
# Experiment I
# --------------------------------------------------------------------------- #
def run_experiment_i(samples: List[Sample], pipeline: CARAGPipeline,
                     vanilla: VanillaRAG, prompt: PromptEnhancedRAG,
                     judge: Judge) -> Dict[str, Dict[str, float]]:
    responses = {
        "Vanilla RAG": [vanilla.answer(s) for s in samples],
        "Prompt-enhanced RAG": [prompt.answer(s) for s in samples],
        "CA-RAG": [pipeline.answer_sample(s) for s in samples],
    }
    return {name: _evaluate(rs, samples, judge)
            for name, rs in responses.items()}


# --------------------------------------------------------------------------- #
# Experiment II (ablation)
# --------------------------------------------------------------------------- #
ABLATIONS: List[Tuple[str, Dict[str, bool]]] = [
    ("CA-RAG",        {"cm": True,  "kgd": True,  "ag": True}),
    ("-CM",           {"cm": False, "kgd": True,  "ag": True}),
    ("-KGD",          {"cm": True,  "kgd": False, "ag": True}),
    ("-AG",           {"cm": True,  "kgd": True,  "ag": False}),
    ("-CM-KGD",       {"cm": False, "kgd": False, "ag": True}),
    ("-CM-AG",        {"cm": False, "kgd": True,  "ag": False}),
    ("-KGD-AG",       {"cm": True,  "kgd": False, "ag": False}),
]


def run_experiment_ii(samples: List[Sample], pipeline: CARAGPipeline,
                      vanilla: VanillaRAG,
                      judge: Judge) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for name, flags in ABLATIONS:
        responses = [pipeline.answer_sample(s, **flags) for s in samples]
        results[name] = _evaluate(responses, samples, judge, include_kgr=True)

    vanilla_responses = [vanilla.answer(s) for s in samples]
    results["Vanilla RAG"] = _evaluate(vanilla_responses, samples, judge,
                                       include_kgr=True)
    return results


# --------------------------------------------------------------------------- #
# Table formatting
# --------------------------------------------------------------------------- #
def format_table(results: Dict[str, Dict[str, float]],
                 order: List[str]) -> str:
    """Render results as an aligned markdown table."""
    metrics = list(next(iter(results.values())).keys())
    header = "| Model | " + " | ".join(metrics) + " |"
    sep = "|---|" + "|".join("---" for _ in metrics) + "|"
    lines = [header, sep]
    for name in order:
        if name in results:
            row = " | ".join(f"{results[name][m]:.2f}" for m in metrics)
            lines.append(f"| {name} | {row} |")
    return "\n".join(lines)


# Paper's reported results, for reference alongside the simulation.
PAPER_EXPERIMENT_I = {
    "Vanilla RAG": {"CAS": 0.54, "PCS": 0.49},
    "Prompt-enhanced RAG": {"CAS": 0.63, "PCS": 0.58},
    "CA-RAG": {"CAS": 0.81, "PCS": 0.78},
}

PAPER_EXPERIMENT_II = {
    "CA-RAG": {"CAS": 0.81, "PCS": 0.78, "KGR": 0.76},
    "-CM": {"CAS": 0.69, "PCS": 0.66, "KGR": 0.61},
    "-KGD": {"CAS": 0.67, "PCS": 0.63, "KGR": 0.58},
    "-AG": {"CAS": 0.70, "PCS": 0.65, "KGR": 0.60},
    "-CM-KGD": {"CAS": 0.61, "PCS": 0.57, "KGR": 0.49},
    "-CM-AG": {"CAS": 0.59, "PCS": 0.55, "KGR": 0.47},
    "-KGD-AG": {"CAS": 0.58, "PCS": 0.52, "KGR": 0.45},
    "Vanilla RAG": {"CAS": 0.54, "PCS": 0.49, "KGR": 0.42},
}
