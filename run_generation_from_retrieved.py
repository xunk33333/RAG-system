from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .generation import DEFAULT_QWEN_MODEL_ID, build_generator
from .text_utils import load_json, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run generation variants (G0-G4) directly from retrieved_results.json "
            "without running retrieval/indexing."
        )
    )
    parser.add_argument("--input", default="retrieved_results.json")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--strict-qwen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if generation backend is not qwen.",
    )
    parser.add_argument("--qwen-model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen-temperature", type=float, default=0.1)
    parser.add_argument(
        "--g4-prompt-id",
        default="baseline",
        help="Prompt strategy used by G4 before rule-based postprocess.",
    )
    return parser.parse_args()


def load_retrieved_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise RuntimeError("Input payload must be a list.")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Row {i} is not an object.")
        if "id" not in row or "question" not in row or "retrieved" not in row:
            raise RuntimeError(f"Row {i} missing one of required keys: id/question/retrieved")
        if not isinstance(row["retrieved"], list):
            raise RuntimeError(f"Row {i} key `retrieved` must be a list.")
        rows.append(row)
    return rows


def _evidence_chunk_ids(retrieved: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        cid = item.get("retrieved_chunk_id")
        if cid is None:
            continue
        out.append(str(cid))
    return out


def run_variant(
    *,
    rows: list[dict[str, Any]],
    variant_id: str,
    prompt_id: str,
    enable_postprocess: bool,
    strict_qwen: bool,
    qwen_model_id: str,
    qwen_max_new_tokens: int,
    qwen_temperature: float,
) -> list[dict[str, Any]]:
    generator = build_generator(
        preferred_backend="qwen",
        variant_id=variant_id,
        prompt_id=prompt_id,
        enable_postprocess=enable_postprocess,
        strict_qwen=strict_qwen,
        qwen_model_id=qwen_model_id,
        max_new_tokens=qwen_max_new_tokens,
        temperature=qwen_temperature,
    )
    outputs: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row["id"])
        question = str(row["question"])
        retrieved = row["retrieved"]
        generated = generator.generate(question, retrieved)
        outputs.append(
            {
                "id": qid,
                "question": question,
                "answer": generated.answer,
                "generation_backend": generated.backend,
                "variant_id": generated.variant_id,
                "prompt_id": generated.prompt_id,
                "postprocess_id": generated.postprocess_id,
                "retrieved": retrieved,
                "evidence_chunk_ids": _evidence_chunk_ids(retrieved),
            }
        )

    if strict_qwen:
        backends = sorted({str(r.get("generation_backend", "")) for r in outputs})
        if backends != ["qwen"]:
            raise RuntimeError(
                "strict-qwen is enabled but non-qwen backend observed. "
                f"Observed backends: {backends}"
            )
    return outputs


def main() -> None:
    args = parse_args()
    rows = load_retrieved_rows(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("G0", "baseline", False),
        ("G1", "evidence_constrained", False),
        ("G2", "uncertainty_gated", False),
        ("G3", "structured_answer", False),
        ("G4", args.g4_prompt_id, True),
    ]

    for variant_id, prompt_id, enable_postprocess in variants:
        outputs = run_variant(
            rows=rows,
            variant_id=variant_id,
            prompt_id=prompt_id,
            enable_postprocess=enable_postprocess,
            strict_qwen=args.strict_qwen,
            qwen_model_id=args.qwen_model_id,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            qwen_temperature=args.qwen_temperature,
        )
        out_path = out_dir / f"predictions.gen.{variant_id}.json"
        write_json(out_path, outputs)
        print(f"{variant_id}: saved {len(outputs)} rows -> {out_path}")


if __name__ == "__main__":
    main()

