"""
Multi-Agent System — 5 Agents for JEE Math Mentor
1. Parser Agent      — converts raw input to structured problem
2. Intent Router     — classifies topic, routes workflow
3. Solver Agent      — solves using RAG + Python calculator
4. Verifier Agent    — checks correctness, triggers HITL
5. Explainer Agent   — produces step-by-step student-friendly solution
"""

import json
import re
import math
from typing import Optional


MODEL = "llama-3.3-70b-versatile"


def safe_parse_json(text: str) -> dict:
    """Safely parse JSON from LLM output, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}


def safe_eval_math(expression: str) -> Optional[float]:
    """Safely evaluate a math expression."""
    try:
        # Allow only safe math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# AGENT 1 — PARSER AGENT
# ═══════════════════════════════════════════════════════════════
def parser_agent(raw_input: str, client) -> dict:
    """
    Converts raw input (OCR/ASR/text) into structured problem JSON.
    Identifies missing info, ambiguities, and triggers HITL if needed.
    """
    system_prompt = """You are a Math Problem Parser for JEE-level math.
Your job is to clean and structure raw input into a precise JSON format.

Raw input may come from:
- OCR (may have errors like 'O' instead of '0', 'l' instead of '1')
- Speech recognition (may have errors with math terms)
- Direct text input

You MUST respond with ONLY a valid JSON object in this exact format:
{
  "problem_text": "cleaned, precise problem statement",
  "topic": "one of: algebra|probability|calculus|linear_algebra|unknown",
  "variables": ["list", "of", "variables"],
  "constraints": ["list of constraints like x > 0"],
  "given_values": {"key": "value pairs of known values"},
  "find": "what the problem asks to find/solve",
  "problem_type": "one of: solve_equation|find_maximum|find_minimum|evaluate_limit|find_derivative|find_integral|find_probability|count_arrangements|matrix_operation|other",
  "needs_clarification": false,
  "clarification_reason": "",
  "ocr_corrections": ["list any OCR corrections made"],
  "confidence": 0.95
}

If the problem is unclear, ambiguous, or critical info is missing:
- Set needs_clarification to true
- Explain in clarification_reason
- Set confidence lower (0.0 to 1.0)

Be strict: do NOT invent information not present in the input."""

    user_prompt = f"""Parse this math problem input:

INPUT: {raw_input}

Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=600,
    )

    result = safe_parse_json(response.choices[0].message.content)

    # Ensure required fields exist
    defaults = {
        "problem_text": raw_input,
        "topic": "unknown",
        "variables": [],
        "constraints": [],
        "given_values": {},
        "find": "solve the problem",
        "problem_type": "other",
        "needs_clarification": False,
        "clarification_reason": "",
        "ocr_corrections": [],
        "confidence": 0.7,
    }
    for key, val in defaults.items():
        if key not in result:
            result[key] = val

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT 2 — INTENT ROUTER AGENT
# ═══════════════════════════════════════════════════════════════
def router_agent(parsed_problem: dict, client) -> dict:
    """
    Classifies the problem and decides the optimal solution workflow.
    """
    system_prompt = """You are a Math Problem Router for JEE-level math.
Given a parsed problem, you decide:
1. The exact topic and subtopic
2. Which tools/approaches to use
3. What formulas/concepts are needed
4. The optimal solution strategy

Respond ONLY with valid JSON:
{
  "topic": "algebra|probability|calculus|linear_algebra",
  "subtopic": "specific subtopic like quadratic_equations|limits|permutation etc",
  "complexity": "easy|medium|hard",
  "requires_calculator": true/false,
  "key_concepts": ["list of key math concepts needed"],
  "solution_strategy": "brief description of approach to take",
  "rag_search_query": "optimized query to search knowledge base",
  "estimated_steps": 4,
  "warnings": ["any edge cases or common mistakes to watch for"]
}"""

    user_prompt = f"""Route this parsed problem:

PARSED PROBLEM:
{json.dumps(parsed_problem, indent=2)}

Return ONLY JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=500,
    )

    result = safe_parse_json(response.choices[0].message.content)

    defaults = {
        "topic": parsed_problem.get("topic", "unknown"),
        "subtopic": "general",
        "complexity": "medium",
        "requires_calculator": False,
        "key_concepts": [],
        "solution_strategy": "Solve step by step",
        "rag_search_query": parsed_problem.get("problem_text", ""),
        "estimated_steps": 4,
        "warnings": [],
    }
    for key, val in defaults.items():
        if key not in result:
            result[key] = val

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT 3 — SOLVER AGENT
# ═══════════════════════════════════════════════════════════════
def solver_agent(
    parsed_problem: dict,
    routing: dict,
    retrieved_context: str,
    memory_context: str,
    client
) -> dict:
    """
    Solves the problem using RAG context + Python calculator tool.
    """
    # Build calculation context
    calc_results = ""
    problem_text = parsed_problem.get("problem_text", "")

    system_prompt = """You are a JEE Math Solver — precise, step-by-step, rigorous.

You are given:
1. The structured problem
2. Relevant formulas from the knowledge base (RAG context)
3. Similar solved problems from memory (if any)

Your job: SOLVE the problem correctly.

Rules:
- Use ONLY the given information. Never assume extra values.
- Show your reasoning clearly.
- Use formulas from the RAG context where applicable.
- If you use a numeric calculation, show the exact arithmetic.
- State the final answer clearly at the end.
- Express confidence (0.0-1.0) in your answer.

Respond ONLY with valid JSON:
{
  "solution_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "calculations": ["any numeric calculations like 2+2=4"],
  "formulas_used": ["list of formulas used"],
  "final_answer": "the complete final answer",
  "answer_value": "just the numeric/symbolic value if applicable",
  "units": "units if applicable",
  "confidence": 0.9,
  "assumptions_made": ["list any assumptions"],
  "rag_sources_used": ["which sources from context were useful"]
}"""

    user_prompt = f"""PROBLEM:
{json.dumps(parsed_problem, indent=2)}

ROUTING INFO:
Strategy: {routing.get('solution_strategy', '')}
Key concepts: {', '.join(routing.get('key_concepts', []))}
Warnings: {', '.join(routing.get('warnings', []))}

RAG CONTEXT (from knowledge base):
{retrieved_context}

MEMORY CONTEXT (similar past problems):
{memory_context if memory_context else "No similar problems in memory yet."}

Solve this problem step by step. Return ONLY JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.15,
        max_tokens=1200,
    )

    result = safe_parse_json(response.choices[0].message.content)

    defaults = {
        "solution_steps": ["Unable to solve"],
        "calculations": [],
        "formulas_used": [],
        "final_answer": "Could not determine answer",
        "answer_value": "",
        "units": "",
        "confidence": 0.5,
        "assumptions_made": [],
        "rag_sources_used": [],
    }
    for key, val in defaults.items():
        if key not in result:
            result[key] = val

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT 4 — VERIFIER / CRITIC AGENT
# ═══════════════════════════════════════════════════════════════
def verifier_agent(
    parsed_problem: dict,
    solution: dict,
    routing: dict,
    client
) -> dict:
    """
    Independently verifies the solution for correctness.
    Checks units, domain constraints, edge cases.
    Triggers HITL if not confident.
    """
    system_prompt = """You are a rigorous JEE Math Verifier — a critic who checks work.

Your job: Independently verify if the solution is correct.

Check:
1. Mathematical correctness — is the logic right?
2. Arithmetic accuracy — are the calculations correct?
3. Domain constraints — are there division by zero, log of negative, sqrt of negative issues?
4. Units consistency — are units handled correctly?
5. Edge cases — does the solution handle all cases?
6. Formula usage — are the right formulas applied correctly?
7. Final answer — does it actually answer what was asked?

Be STRICT. If you find any issue, flag it.

Respond ONLY with valid JSON:
{
  "is_correct": true/false,
  "confidence": 0.95,
  "verdict": "CORRECT|LIKELY_CORRECT|UNCERTAIN|INCORRECT",
  "issues_found": ["list any issues"],
  "corrections_needed": ["list specific corrections if any"],
  "domain_check": "passed|failed|not_applicable",
  "arithmetic_check": "passed|failed|not_checked",
  "unit_check": "passed|failed|not_applicable",
  "trigger_hitl": false,
  "hitl_reason": "",
  "verification_notes": "brief notes on verification process"
}"""

    user_prompt = f"""Verify this solution:

ORIGINAL PROBLEM:
{parsed_problem.get('problem_text', '')}

TOPIC: {routing.get('topic', '')} | SUBTOPIC: {routing.get('subtopic', '')}

PROPOSED SOLUTION:
Steps: {json.dumps(solution.get('solution_steps', []))}
Formulas used: {', '.join(solution.get('formulas_used', []))}
Final answer: {solution.get('final_answer', '')}
Solver confidence: {solution.get('confidence', 0.5)}

CONSTRAINTS: {json.dumps(parsed_problem.get('constraints', []))}
WARNINGS: {json.dumps(routing.get('warnings', []))}

Verify rigorously. Return ONLY JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.05,
        max_tokens=600,
    )

    result = safe_parse_json(response.choices[0].message.content)

    defaults = {
        "is_correct": False,
        "confidence": 0.5,
        "verdict": "UNCERTAIN",
        "issues_found": [],
        "corrections_needed": [],
        "domain_check": "not_checked",
        "arithmetic_check": "not_checked",
        "unit_check": "not_applicable",
        "trigger_hitl": True,
        "hitl_reason": "Could not verify",
        "verification_notes": "",
    }
    for key, val in defaults.items():
        if key not in result:
            result[key] = val

    # Auto-trigger HITL if confidence is low
    if result.get("confidence", 0.5) < 0.7:
        result["trigger_hitl"] = True
        if not result.get("hitl_reason"):
            result["hitl_reason"] = f"Low verifier confidence: {result['confidence']:.0%}"

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT 5 — EXPLAINER / TUTOR AGENT
# ═══════════════════════════════════════════════════════════════
def explainer_agent(
    parsed_problem: dict,
    solution: dict,
    verifier: dict,
    routing: dict,
    client
) -> dict:
    """
    Produces a student-friendly step-by-step explanation.
    Writes as if tutoring a JEE student directly.
    """
    system_prompt = """You are a friendly, expert JEE Math Tutor.

Your job: Write a clear, student-friendly explanation of the solution.

Guidelines:
- Write as if explaining to a student who got it wrong
- Explain WHY each step is done, not just WHAT
- Highlight the key insight or trick
- Mention common mistakes to avoid
- Use simple, clear language
- Add encouraging notes where helpful
- For JEE: note if this is a common exam pattern

Respond ONLY with valid JSON:
{
  "title": "brief title for this solution",
  "introduction": "1-2 sentence overview of approach",
  "steps": [
    {
      "step_number": 1,
      "heading": "short step title",
      "explanation": "detailed explanation of this step",
      "math": "the actual math written out",
      "why": "why we do this step"
    }
  ],
  "key_insight": "the main trick or key insight for this problem",
  "common_mistakes": ["mistakes students commonly make on this type"],
  "jee_tip": "specific tip for JEE exam context",
  "final_answer_boxed": "Final Answer: [the answer]",
  "difficulty_rating": "Easy|Medium|Hard",
  "similar_problems_hint": "hint about similar problem types"
}"""

    user_prompt = f"""Create a student-friendly explanation for:

PROBLEM: {parsed_problem.get('problem_text', '')}
TOPIC: {routing.get('topic', '')} | {routing.get('subtopic', '')}

SOLUTION:
{json.dumps(solution.get('solution_steps', []))}

FINAL ANSWER: {solution.get('final_answer', '')}
FORMULAS USED: {', '.join(solution.get('formulas_used', []))}

VERIFIER SAYS: {verifier.get('verdict', '')}
VERIFIER NOTES: {verifier.get('verification_notes', '')}

Write a clear, encouraging explanation. Return ONLY JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    result = safe_parse_json(response.choices[0].message.content)

    defaults = {
        "title": "Solution",
        "introduction": "Let's solve this step by step.",
        "steps": [{"step_number": 1, "heading": "Solution",
                   "explanation": solution.get("final_answer", ""),
                   "math": "", "why": ""}],
        "key_insight": "",
        "common_mistakes": [],
        "jee_tip": "",
        "final_answer_boxed": f"Final Answer: {solution.get('final_answer', '')}",
        "difficulty_rating": routing.get("complexity", "Medium").title(),
        "similar_problems_hint": "",
    }
    for key, val in defaults.items():
        if key not in result:
            result[key] = val

    return result


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════
def run_full_pipeline(
    raw_input: str,
    retrieved_context: str,
    memory_context: str,
    client,
    progress_callback=None
) -> dict:
    """
    Run all 5 agents in sequence.
    Returns complete pipeline result.
    """
    def progress(step, msg):
        if progress_callback:
            progress_callback(step, msg)

    progress(1, "Parser Agent: Structuring problem...")
    parsed = parser_agent(raw_input, client)

    progress(2, "Router Agent: Planning solution strategy...")
    routing = router_agent(parsed, client)

    progress(3, "Solver Agent: Solving with RAG context...")
    solution = solver_agent(parsed, routing, retrieved_context, memory_context, client)

    progress(4, "Verifier Agent: Checking correctness...")
    verifier = verifier_agent(parsed, solution, routing, client)

    progress(5, "Explainer Agent: Writing student explanation...")
    explanation = explainer_agent(parsed, solution, verifier, routing, client)

    return {
        "parsed": parsed,
        "routing": routing,
        "solution": solution,
        "verifier": verifier,
        "explanation": explanation,
    }
