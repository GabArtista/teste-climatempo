"""
Quantitative validation of LLM function calling precision.

Methodology: Prompt sampling with binary classification.
Each prompt is sent to the agent. The 'tool_called' field in the response
indicates whether the model decided to invoke the weather tool.

Metrics:
    Precision = TP / (TP + FP)  — when tool was called, was it correct?
    Recall    = TP / (TP + FN)  — of weather prompts, how many triggered the tool?
    F1-Score  = 2 * P * R / (P + R)

Results are saved to tests/Validation/results.json for the PDF report.
"""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from main import app
from config.settings import get_settings

# --- Dataset ---

POSITIVE_PROMPTS = [
    # Should trigger the weather tool (expected: tool_called = True)
    # City is clearly stated — model must call the tool
    "Como está o tempo em São Paulo?",
    "Qual a previsão do tempo para Curitiba nos próximos 3 dias?",
    "Vai chover em Manaus amanhã?",
    "Me diga a temperatura máxima em Brasília",
    "Previsão do tempo para o Rio de Janeiro essa semana",
    "Como vai estar o clima em Fortaleza?",
    "Está chovendo em Porto Alegre?",
    "Qual a temperatura mínima em Belo Horizonte hoje?",
]

NEGATIVE_PROMPTS = [
    # Should NOT trigger the weather tool (expected: tool_called = False)

    # Off-topic: greetings, math, general knowledge
    "Olá, tudo bem?",
    "Qual a capital do Brasil?",
    "Me conte uma piada",
    "Quanto é 2 + 2?",
    "Qual o seu nome?",
    "O que é machine learning?",
    "Me explique o que é uma API REST",

    # Incomplete: weather topic but NO city provided
    # Correct behavior: ask for city instead of calling tool
    "Qual é a previsão do tempo?",
    "Vai chover amanhã?",
    "Como está o clima hoje?",
]


def _run_validation() -> dict:
    """Execute the full validation suite and return metrics."""
    client = TestClient(app)
    tp, fp, fn, tn = 0, 0, 0, 0
    details = []

    print(f"\n{'='*60}")
    print("  RUNNING FUNCTION CALLING VALIDATION")
    print(f"{'='*60}")

    for prompt in POSITIVE_PROMPTS:
        try:
            resp = client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": []},
                timeout=60,
            )
            called = resp.json().get("tool_called", False) if resp.status_code == 200 else False
        except Exception:
            called = False

        result = "TP ✅" if called else "FN ❌"
        print(f"  [POS] {result} | {prompt[:50]}")
        tp += int(called)
        fn += int(not called)
        details.append({"prompt": prompt, "expected": True, "actual": called, "correct": called})

    for prompt in NEGATIVE_PROMPTS:
        try:
            resp = client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": []},
                timeout=60,
            )
            called = resp.json().get("tool_called", False) if resp.status_code == 200 else False
        except Exception:
            called = False

        result = "TN ✅" if not called else "FP ❌"
        print(f"  [NEG] {result} | {prompt[:50]}")
        tn += int(not called)
        fp += int(called)
        details.append({"prompt": prompt, "expected": False, "actual": called, "correct": not called})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Model:      {get_settings().ollama_model}")
    print(f"  Dataset:    {len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)} prompts")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"{'='*60}\n")

    return {
        "model": get_settings().ollama_model,
        "methodology": "prompt-sampling-binary-classification",
        "dataset_size": len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS),
        "positive_count": len(POSITIVE_PROMPTS),
        "negative_count": len(NEGATIVE_PROMPTS),
        "negative_breakdown": {
            "off_topic": 7,
            "incomplete_no_city": 3,
        },
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "details": details,
    }


def test_function_calling_f1():
    """
    Run the full F1 validation suite.

    Note: This test requires Ollama running with qwen2.5:1.5b.
    Run: ollama serve (in another terminal)

    Per challenge spec, the construction and methodology are evaluated,
    not the metric values themselves.
    """
    results = _run_validation()

    # Save results for PDF report
    output_path = Path(__file__).parent / "results.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results saved to: {output_path}")

    # Validation: metrics must be calculable (not assert on values per spec)
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0
    assert results["dataset_size"] == len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
