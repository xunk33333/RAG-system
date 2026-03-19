from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

from .text_utils import first_sentence, normalize_text, split_sentences, tokenize

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STANDARD_UNCERTAINTY_ANSWER = "I do not have enough evidence in the retrieved context to answer confidently."
PREFIX_RE = re.compile(r"^(answer|final answer|response|assistant)\s*[:：-]\s*", re.IGNORECASE)

VARIANT_DEFAULT_PROMPTS = {
    "G0": "baseline",
    "G1": "evidence_constrained",
    "G2": "uncertainty_gated",
    "G3": "structured_answer",
    "G4": "structured_answer",
}


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    backend: str
    prompt: str
    variant_id: str = "G0"
    prompt_id: str = "baseline"
    postprocess_id: str = "none"
    model_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "backend": self.backend,
            "prompt": self.prompt,
            "variant_id": self.variant_id,
            "prompt_id": self.prompt_id,
            "postprocess_id": self.postprocess_id,
            "model_id": self.model_id,
        }


class Generator(Protocol):
    def generate(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> GenerationResult: ...


SYSTEM_PROMPT = """You are an East Asian cuisine assistant specialized in factual QA.
Rules:
1) Answer using only the provided evidence.
2) If evidence is weak, say uncertainty explicitly.
3) Keep the answer concise and factual.
4) Do not invent facts beyond evidence.
"""


def _resolve_prompt_id(variant_id: str, prompt_id: str | None) -> str:
    if prompt_id:
        return prompt_id
    return VARIANT_DEFAULT_PROMPTS.get(variant_id, "baseline")


def _resolve_postprocess_enabled(variant_id: str, enable_postprocess: bool | None) -> bool:
    if enable_postprocess is not None:
        return bool(enable_postprocess)
    return variant_id == "G4"


def _build_evidence_block(retrieved_chunks: list[dict[str, Any]]) -> str:
    evidence_lines: list[str] = []
    for item in retrieved_chunks[:5]:
        cid = item.get("retrieved_chunk_id", "unknown_chunk")
        src = item.get("retrieved_source", "unknown_source")
        txt = normalize_text(str(item.get("retrieved_answer", "")))
        evidence_lines.append(f"[{cid}] ({src}) {txt}")
    return "\n".join(evidence_lines) if evidence_lines else "[NO EVIDENCE]"


def build_prompt(query: str, retrieved_chunks: list[dict[str, Any]], prompt_id: str = "baseline") -> str:
    evidence_block = _build_evidence_block(retrieved_chunks)
    instructions: dict[str, str] = {
        "baseline": (
            "Return a direct answer in one or two sentences, grounded in evidence only."
        ),
        "evidence_constrained": (
            "Return only evidence-grounded claims and include at least one chunk citation like [chunk_123]."
        ),
        "uncertainty_gated": (
            "If evidence is insufficient or contradictory, output exactly: "
            f"{STANDARD_UNCERTAINTY_ANSWER}"
        ),
        "structured_answer": (
            "Format exactly as: 'Conclusion: ... Key details: ...' and keep it concise."
        ),
    }
    style_instruction = instructions.get(prompt_id, instructions["baseline"])
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Prompt style: {prompt_id}\n"
        f"{style_instruction}\n\n"
        f"Question:\n{query}\n\n"
        f"Evidence:\n{evidence_block}\n"
    )


def _top_chunk_id(retrieved_chunks: list[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return ""
    first = retrieved_chunks[0]
    cid = first.get("retrieved_chunk_id", "")
    return str(cid) if cid else ""


def _has_weak_evidence(retrieved_chunks: list[dict[str, Any]], threshold: float = 0.12) -> bool:
    if not retrieved_chunks:
        return True
    try:
        top_score = float(retrieved_chunks[0].get("score", 0.0))
    except (TypeError, ValueError):
        top_score = 0.0
    return top_score < threshold


def _format_answer_for_prompt(answer: str, prompt_id: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    answer = normalize_text(answer)
    if not answer:
        return STANDARD_UNCERTAINTY_ANSWER

    cid = _top_chunk_id(retrieved_chunks)
    if prompt_id == "evidence_constrained":
        if cid and f"[{cid}]" not in answer:
            answer = f"{answer} [{cid}]"
    elif prompt_id == "structured_answer":
        if not answer.lower().startswith("conclusion:"):
            detail = f"evidence from [{cid}]" if cid else "evidence limited"
            answer = f"Conclusion: {answer} Key details: {detail}."
    elif prompt_id == "uncertainty_gated":
        if _has_weak_evidence(retrieved_chunks):
            return STANDARD_UNCERTAINTY_ANSWER
    return answer


def apply_rule_based_postprocess(answer: str, query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """Deterministic and explainable postprocess for G4."""
    if _has_weak_evidence(retrieved_chunks):
        return STANDARD_UNCERTAINTY_ANSWER

    cleaned = normalize_text(answer)
    cleaned = PREFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return STANDARD_UNCERTAINTY_ANSWER

    if re.search(r"(insufficient evidence|not enough evidence|cannot determine|can't determine|unknown)", cleaned, re.IGNORECASE):
        return STANDARD_UNCERTAINTY_ANSWER

    query_terms = set(tokenize(query))
    evidence_terms: set[str] = set()
    for item in retrieved_chunks[:3]:
        evidence_terms.update(tokenize(str(item.get("retrieved_answer", ""))))

    kept: list[str] = []
    for sentence in split_sentences(cleaned):
        sentence_terms = set(tokenize(sentence))
        if not sentence_terms:
            continue
        overlap = len(sentence_terms & query_terms) + len(sentence_terms & evidence_terms)
        if overlap > 0:
            kept.append(sentence)

    if not kept:
        kept = split_sentences(cleaned)[:1]

    compressed = " ".join(kept[:2]).strip()
    compressed = re.sub(r"\s+", " ", compressed)
    if compressed and compressed[-1] not in ".!?":
        compressed += "."
    return compressed or STANDARD_UNCERTAINTY_ANSWER


class ExtractiveGenerator:
    """Deterministic fallback for CPU-only and offline demo safety."""

    def __init__(
        self,
        variant_id: str = "G0",
        prompt_id: str = "baseline",
        enable_postprocess: bool = False,
    ) -> None:
        self.variant_id = variant_id
        self.prompt_id = prompt_id
        self.enable_postprocess = enable_postprocess

    def generate(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> GenerationResult:
        prompt = build_prompt(query, retrieved_chunks, prompt_id=self.prompt_id)
        if not retrieved_chunks:
            return GenerationResult(
                answer=STANDARD_UNCERTAINTY_ANSWER,
                backend="extractive",
                prompt=prompt,
                variant_id=self.variant_id,
                prompt_id=self.prompt_id,
                postprocess_id="rule_v1" if self.enable_postprocess else "none",
            )
        best = self._best_chunk(query, retrieved_chunks)
        candidate = first_sentence(str(best.get("retrieved_answer", "")))
        candidate = _format_answer_for_prompt(candidate, self.prompt_id, retrieved_chunks)
        if self.enable_postprocess:
            candidate = apply_rule_based_postprocess(candidate, query, retrieved_chunks)
        if not candidate:
            candidate = STANDARD_UNCERTAINTY_ANSWER
        return GenerationResult(
            answer=candidate,
            backend="extractive",
            prompt=prompt,
            variant_id=self.variant_id,
            prompt_id=self.prompt_id,
            postprocess_id="rule_v1" if self.enable_postprocess else "none",
        )

    def _best_chunk(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        q_terms = set(tokenize(query))
        ranked = []
        for row in retrieved_chunks:
            text = str(row.get("retrieved_answer", ""))
            overlap = len(q_terms & set(tokenize(text)))
            score = float(row.get("score", 0.0))
            ranked.append((overlap, score, row))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return ranked[0][2]


class QwenGenerator:
    """Hugging Face backend for required model Qwen/Qwen2.5-0.5B-Instruct."""

    def __init__(
        self,
        variant_id: str = "G0",
        prompt_id: str = "baseline",
        enable_postprocess: bool = False,
        model_id: str = DEFAULT_QWEN_MODEL_ID,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for qwen backend. Install with: pip install '.[qwen]'"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.variant_id = variant_id
        self.prompt_id = prompt_id
        self.enable_postprocess = enable_postprocess
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)

    def generate(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> GenerationResult:
        prompt = build_prompt(query, retrieved_chunks, prompt_id=self.prompt_id)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        model_inputs = self._tokenizer([text], return_tensors="pt")
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=max(self.temperature, 1e-6),
            top_p=0.9,
        )
        out_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        answer = self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        answer = _format_answer_for_prompt(answer, self.prompt_id, retrieved_chunks)
        if self.enable_postprocess:
            answer = apply_rule_based_postprocess(answer, query, retrieved_chunks)
        if not answer:
            answer = STANDARD_UNCERTAINTY_ANSWER
        return GenerationResult(
            answer=answer,
            backend="qwen",
            prompt=prompt,
            variant_id=self.variant_id,
            prompt_id=self.prompt_id,
            postprocess_id="rule_v1" if self.enable_postprocess else "none",
            model_id=self.model_id,
        )


def build_generator(
    preferred_backend: str = "qwen",
    *,
    variant_id: str = "G0",
    prompt_id: str | None = None,
    enable_postprocess: bool | None = None,
    strict_qwen: bool = False,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> Generator:
    resolved_prompt_id = _resolve_prompt_id(variant_id=variant_id, prompt_id=prompt_id)
    resolved_postprocess = _resolve_postprocess_enabled(
        variant_id=variant_id,
        enable_postprocess=enable_postprocess,
    )

    if preferred_backend == "qwen":
        try:
            return QwenGenerator(
                variant_id=variant_id,
                prompt_id=resolved_prompt_id,
                enable_postprocess=resolved_postprocess,
                model_id=qwen_model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            if strict_qwen:
                raise RuntimeError(
                    "Qwen backend is required for this run, but initialization failed. "
                    f"Original error: {exc}"
                ) from exc
            warnings.warn(
                f"Qwen backend unavailable ({exc}); falling back to extractive backend.",
                stacklevel=2,
            )
            return ExtractiveGenerator(
                variant_id=variant_id,
                prompt_id=resolved_prompt_id,
                enable_postprocess=resolved_postprocess,
            )
    if preferred_backend == "extractive":
        return ExtractiveGenerator(
            variant_id=variant_id,
            prompt_id=resolved_prompt_id,
            enable_postprocess=resolved_postprocess,
        )
    raise ValueError(f"Unknown backend: {preferred_backend}")
