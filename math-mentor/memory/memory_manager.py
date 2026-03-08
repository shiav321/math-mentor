"""
Memory Manager — stores solved problems, feedback, and learning signals
Uses JSON file storage — no model retraining needed
"""

import json
import os
from datetime import datetime
from typing import Optional

MEMORY_FILE = "memory/memory_store.json"


def _load_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        os.makedirs("memory", exist_ok=True)
        return {"problems": [], "corrections": [], "ocr_rules": []}
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"problems": [], "corrections": [], "ocr_rules": []}


def _save_memory(data: dict):
    os.makedirs("memory", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def store_solved_problem(
    input_text: str,
    parsed_problem: dict,
    retrieved_context: str,
    final_answer: str,
    explanation: str,
    verifier_outcome: dict,
    user_feedback: Optional[str] = None,
    input_type: str = "text"
):
    """Store a successfully solved problem in memory."""
    memory = _load_memory()
    entry = {
        "id": len(memory["problems"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "input_type": input_type,
        "input_text": input_text[:500],
        "parsed_problem": parsed_problem,
        "retrieved_context": retrieved_context[:800],
        "final_answer": final_answer,
        "explanation": explanation[:1000],
        "verifier_outcome": verifier_outcome,
        "user_feedback": user_feedback,
        "topic": parsed_problem.get("topic", "unknown"),
    }
    memory["problems"].append(entry)
    _save_memory(memory)
    return entry["id"]


def store_correction(original_input: str, corrected_answer: str, comment: str):
    """Store user correction as a learning signal."""
    memory = _load_memory()
    correction = {
        "timestamp": datetime.now().isoformat(),
        "original_input": original_input[:300],
        "corrected_answer": corrected_answer,
        "comment": comment,
    }
    memory["corrections"].append(correction)
    _save_memory(memory)


def store_ocr_correction(original_ocr: str, corrected_text: str):
    """Store OCR correction rule for future reference."""
    memory = _load_memory()
    rule = {
        "timestamp": datetime.now().isoformat(),
        "original_ocr": original_ocr[:200],
        "corrected_text": corrected_text[:200],
    }
    memory["ocr_rules"].append(rule)
    _save_memory(memory)


def find_similar_problems(query: str, topic: str = "", limit: int = 3) -> list:
    """
    Retrieve similar past solved problems for context reuse.
    Simple keyword matching — no embeddings needed.
    """
    memory = _load_memory()
    problems = memory.get("problems", [])

    if not problems:
        return []

    # Filter by topic first if provided
    if topic:
        relevant = [p for p in problems if p.get("topic", "").lower() == topic.lower()]
    else:
        relevant = problems

    # Score by keyword overlap
    query_words = set(query.lower().split())
    scored = []
    for p in relevant:
        problem_words = set(p.get("input_text", "").lower().split())
        overlap = len(query_words & problem_words)
        if overlap > 0:
            scored.append((overlap, p))

    # Sort by overlap score
    scored.sort(key=lambda x: x[0], reverse=True)

    return [p for _, p in scored[:limit]]


def get_ocr_correction_rules() -> list:
    """Get stored OCR correction patterns."""
    memory = _load_memory()
    return memory.get("ocr_rules", [])


def get_memory_stats() -> dict:
    """Get memory statistics."""
    memory = _load_memory()
    problems = memory.get("problems", [])
    corrections = memory.get("corrections", [])
    ocr_rules = memory.get("ocr_rules", [])

    topics = {}
    for p in problems:
        topic = p.get("topic", "unknown")
        topics[topic] = topics.get(topic, 0) + 1

    positive = sum(1 for p in problems if p.get("user_feedback") == "correct")
    negative = sum(1 for p in problems if p.get("user_feedback") == "incorrect")

    return {
        "total_problems_solved": len(problems),
        "total_corrections": len(corrections),
        "ocr_rules_learned": len(ocr_rules),
        "topics_breakdown": topics,
        "positive_feedback": positive,
        "negative_feedback": negative,
    }


def format_similar_for_context(similar_problems: list) -> str:
    """Format similar problems as context string for agents."""
    if not similar_problems:
        return ""

    context_parts = ["=== SIMILAR PROBLEMS FROM MEMORY ===\n"]
    for i, p in enumerate(similar_problems, 1):
        context_parts.append(
            f"[Memory {i}] Topic: {p.get('topic', 'unknown')}\n"
            f"Problem: {p.get('input_text', '')[:200]}\n"
            f"Answer: {p.get('final_answer', '')[:300]}\n"
            f"Feedback: {p.get('user_feedback', 'not rated')}\n"
        )
    return "\n".join(context_parts)
