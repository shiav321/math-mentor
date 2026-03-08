# 🎓 Math Mentor — AI-Powered JEE Math Problem Solver

<div align="center">

![Math Mentor Banner](https://img.shields.io/badge/Math%20Mentor-AI%20JEE%20Solver-1B2A4A?style=for-the-badge&logo=graduation-cap&logoColor=white)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://math-mentor-shiva.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-shiav321-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/shiav321)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shiva%20Keshava-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/shiva-keshava-b71355364)
[![Portfolio](https://img.shields.io/badge/Portfolio-shiav321.github.io-0B6E6E?style=for-the-badge&logo=safari&logoColor=white)](https://shiav321.github.io)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=flat-square&logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Assignment](https://img.shields.io/badge/AI%20Planet-AI%20Engineer%20Assignment-4F46E5?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [5-Agent Pipeline](#-5-agent-pipeline)
- [RAG Pipeline](#-rag-pipeline)
- [Human-in-the-Loop](#-human-in-the-loop-hitl)
- [Memory & Self-Learning](#-memory--self-learning)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Use](#-how-to-use)
- [Knowledge Base](#-knowledge-base)
- [Deployment](#-deployment)
- [Demo Video](#-demo-video)
- [About the Developer](#-about-the-developer)

---

## 🧠 Overview

**Math Mentor** is a production-ready, end-to-end AI application built for the **AI Planet AI Engineer Assignment**. It solves JEE-level math problems (Algebra, Probability, Calculus, Linear Algebra) through a sophisticated multi-agent pipeline with RAG-based knowledge retrieval, multimodal input processing, human oversight, and self-improving memory.

> **Built to demonstrate:** RAG pipeline design, multi-agent systems, multimodal AI (image + audio + text), Human-in-the-Loop workflows, memory & self-learning — all packaged in a fully deployed production application.

### What makes it different from a simple "ask GPT" app?

| Feature | Simple LLM App | Math Mentor |
|---------|---------------|-------------|
| Input types | Text only | Text + Image OCR + Audio |
| Reasoning | Single call | 5 specialized agents |
| Knowledge | Only training data | RAG knowledge base |
| Error checking | None | Independent verifier agent |
| Oversight | None | Human-in-the-Loop system |
| Improvement | Static | Memory + self-learning |
| Transparency | Black box | Full agent trace UI |

---

## 🚀 Live Demo

🔗 **[https://math-mentor-shiva.streamlit.app](https://math-mentor-shiva.streamlit.app)**

> You'll need a free Groq API key from [console.groq.com](https://console.groq.com) — takes 30 seconds to create.

---

## ✨ Features

### 📥 Multimodal Input
- **Text** — type or paste any JEE math problem
- **Image** — upload photo/screenshot → EasyOCR extracts text automatically
- **Audio** — upload MP3/WAV → OpenAI Whisper transcribes to text
- User can edit OCR/transcript before solving
- HITL triggers automatically on low confidence extraction

### 🤖 5-Agent Pipeline
- **Parser Agent** — cleans raw input, structures to JSON
- **Router Agent** — classifies topic, plans solution strategy
- **Solver Agent** — solves using RAG context + Python calculator
- **Verifier Agent** — independently checks correctness, domain, units
- **Explainer Agent** — writes student-friendly step-by-step solution

### 📚 RAG Knowledge Base
- 5 curated knowledge files (algebra, probability, calculus, linear algebra, templates)
- Keyword-based retrieval — top-5 most relevant chunks
- Sources displayed transparently in UI
- No hallucinated citations

### ⚠️ Human-in-the-Loop (HITL)
- Auto-triggers on low OCR/ASR confidence
- Auto-triggers on low verifier confidence (< 70%)
- Human can approve / edit / reject
- Corrections stored as learning signals

### 🧠 Memory & Self-Learning
- Every solved problem stored in JSON memory store
- Similar past problems retrieved at runtime
- OCR correction rules stored and reused
- No model retraining — pure pattern reuse

### 🖥️ Full Transparency UI
- Live agent trace (watch each agent run)
- Retrieved context panel with source citations
- Confidence indicators for solver + verifier
- Raw JSON debug view for all agent outputs

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STUDENT INPUT                         │
│         Text  |  Image (OCR)  |  Audio (Whisper)        │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  HITL CHECK     │ ◄── Low confidence?
              │  Confidence OK? │     Human reviews here
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ 1️⃣ PARSER AGENT  │
              │ Raw → JSON      │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ 2️⃣ ROUTER AGENT  │
              │ Topic + Strategy│
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌───▼─────┐
    │   RAG   │   │ MEMORY  │   │KNOWLEDGE│
    │Retrieval│   │Similar  │   │  BASE   │
    │ Top-5   │   │Problems │   │5 files  │
    └────┬────┘   └────┬────┘   └───┬─────┘
         └─────────────┼────────────┘
                       │
              ┌────────▼────────┐
              │ 3️⃣ SOLVER AGENT  │
              │ RAG + Calculator│
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ 4️⃣ VERIFIER AGENT│ ◄── Low confidence?
              │ Check + Verify  │     HITL triggers again
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ 5️⃣ EXPLAINER    │
              │ Student-Friendly│
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   UI OUTPUT     │
              │ Answer + Steps  │
              │ Trace + Sources │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  USER FEEDBACK  │
              │ ✅ Correct      │
              │ ❌ Incorrect    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  MEMORY STORE   │
              │ Learn + Improve │
              └─────────────────┘
```

---

## 🤖 5-Agent Pipeline

### Agent 1 — Parser Agent
**Role:** Converts raw input (OCR/ASR/text) into structured JSON

**Input:** Raw string (possibly noisy from OCR or speech)

**Output:**
```json
{
  "problem_text": "Find the roots of 2x² - 5x + 3 = 0",
  "topic": "algebra",
  "variables": ["x"],
  "constraints": [],
  "given_values": {},
  "find": "roots of the equation",
  "problem_type": "solve_equation",
  "needs_clarification": false,
  "ocr_corrections": ["O→0 correction applied"],
  "confidence": 0.97
}
```

**HITL Trigger:** `needs_clarification = true`

---

### Agent 2 — Router Agent
**Role:** Classifies topic and plans optimal solution strategy

**Output:**
```json
{
  "topic": "algebra",
  "subtopic": "quadratic_equations",
  "complexity": "medium",
  "requires_calculator": true,
  "key_concepts": ["discriminant", "quadratic formula"],
  "solution_strategy": "Apply quadratic formula after computing discriminant",
  "rag_search_query": "quadratic equation roots formula discriminant",
  "estimated_steps": 4,
  "warnings": ["Check discriminant sign before taking square root"]
}
```

---

### Agent 3 — Solver Agent
**Role:** Solves the problem using RAG context + Python calculator

**Uses:** Retrieved knowledge base chunks + similar memory problems

**Output:**
```json
{
  "solution_steps": [
    "Step 1: Identify a=2, b=-5, c=3",
    "Step 2: Calculate D = b²-4ac = 25-24 = 1",
    "Step 3: x = (5 ± √1) / 4",
    "Step 4: x = 1.5 or x = 1"
  ],
  "formulas_used": ["x = (-b ± √(b²-4ac)) / 2a"],
  "final_answer": "x = 3/2 or x = 1",
  "confidence": 0.96
}
```

---

### Agent 4 — Verifier Agent
**Role:** Independently verifies correctness — acts as a critic

**Checks:** Mathematical correctness, arithmetic accuracy, domain constraints, units, edge cases

**Output:**
```json
{
  "verdict": "CORRECT",
  "confidence": 0.95,
  "domain_check": "passed",
  "arithmetic_check": "passed",
  "issues_found": [],
  "trigger_hitl": false
}
```

**HITL Trigger:** `confidence < 0.70` or `verdict = UNCERTAIN`

---

### Agent 5 — Explainer Agent
**Role:** Writes student-friendly, JEE-focused explanation

**Output:**
```json
{
  "title": "Solving Quadratic by Formula",
  "introduction": "We use the quadratic formula since factoring isn't obvious",
  "steps": [
    {
      "step_number": 1,
      "heading": "Identify Coefficients",
      "explanation": "Match 2x²-5x+3=0 with ax²+bx+c=0",
      "math": "a=2, b=-5, c=3",
      "why": "The formula needs these three values"
    }
  ],
  "key_insight": "Discriminant > 0 means two distinct real roots",
  "jee_tip": "Always compute D first — it tells you root nature before solving",
  "common_mistakes": ["Forgetting ± gives both roots", "Sign error on b"]
}
```

---

## 📚 RAG Pipeline

```
Knowledge Base Files (.txt)
         │
         ▼
    Chunk Text
  (400 words, 80 overlap)
         │
         ▼
   Build Index
  (JSON store)
         │
         ▼
  Query Retrieval
  (Keyword scoring)
         │
         ▼
  Top-5 Chunks
  → Solver Agent context
```

**Knowledge Base:**

| File | Topics Covered | Chunks |
|------|---------------|--------|
| `algebra.txt` | Quadratics, AP/GP, Binomial, Logs, Complex Numbers | ~12 |
| `probability.txt` | Basic Prob, Bayes, Distributions, Counting | ~10 |
| `calculus.txt` | Limits, L'Hopital, Derivatives, Integration | ~11 |
| `linear_algebra.txt` | Matrices, Determinants, Vectors, Eigenvalues | ~9 |
| `solution_templates.txt` | JEE Problem Patterns, Step-by-Step Templates | ~5 |

---

## ⚠️ Human-in-the-Loop (HITL)

HITL triggers automatically in 4 situations:

| Trigger | Condition | Action |
|---------|-----------|--------|
| OCR confidence | < 60% | Show extracted text for editing |
| ASR clarity | Transcript too short/unclear | Ask user to confirm |
| Parser ambiguity | `needs_clarification = true` | Ask specific clarifying question |
| Verifier uncertainty | Confidence < 70% | Human approve/reject solution |

**HITL Flow:**
```
Auto-detect low confidence
       │
       ▼
Show HITL Panel to user
       │
  ┌────┴────┐
  │         │
Approve   Reject + Edit
  │         │
  ▼         ▼
Continue  Store correction
pipeline  as learning signal
```

---

## 🧠 Memory & Self-Learning

**What gets stored:**
```json
{
  "id": 1,
  "timestamp": "2026-03-08T12:00:00",
  "input_type": "text",
  "input_text": "Find roots of 2x²-5x+3=0",
  "parsed_problem": {...},
  "retrieved_context": "...",
  "final_answer": "x = 3/2 or x = 1",
  "verifier_outcome": {"verdict": "CORRECT"},
  "user_feedback": "correct",
  "topic": "algebra"
}
```

**How memory is used at runtime:**
1. Query → keyword match against past problems
2. Top-3 similar problems retrieved
3. Passed to Solver Agent as additional context
4. Solution patterns reused → faster + more accurate

**OCR Learning:**
- When user corrects OCR output → stored as correction rule
- Future similar OCR errors → auto-corrected using stored rules

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit 1.31+ | Web application interface |
| **LLM** | Groq LLaMA 3.3 70B | All 5 agent reasoning |
| **Knowledge Retrieval** | Keyword-based RAG | Find relevant math formulas |
| **OCR** | EasyOCR | Extract text from math images |
| **ASR** | OpenAI Whisper | Convert audio to text |
| **Memory Store** | JSON file store | Persist solved problems |
| **Language** | Python 3.10+ | Core development |
| **Deployment** | Streamlit Cloud | Free hosting |

---

## 📁 Project Structure

```
math_mentor/
│
├── 📄 app.py                        # Main Streamlit application
│                                    # Full UI: input, agents, trace, memory
│
├── 📄 requirements.txt              # All Python dependencies
├── 📄 .env.example                  # Environment variables template
├── 📄 README.md                     # This file
├── 📄 architecture.md               # Mermaid architecture diagram
│
├── 🤖 agents/
│   ├── __init__.py
│   └── agents.py                    # All 5 agents:
│                                    # - parser_agent()
│                                    # - router_agent()
│                                    # - solver_agent()
│                                    # - verifier_agent()
│                                    # - explainer_agent()
│                                    # - run_full_pipeline()
│
├── 📚 rag/
│   ├── __init__.py
│   └── rag_pipeline.py              # RAG system:
│                                    # - chunk_text()
│                                    # - load_knowledge_base()
│                                    # - build_index()
│                                    # - retrieve() → top-k chunks
│                                    # - format_retrieved_context()
│
├── 🧠 memory/
│   ├── __init__.py
│   └── memory_manager.py            # Memory system:
│                                    # - store_solved_problem()
│                                    # - find_similar_problems()
│                                    # - store_ocr_correction()
│                                    # - get_memory_stats()
│
└── 📖 knowledge_base/
    ├── algebra.txt                  # Quadratics, series, logs, complex numbers
    ├── probability.txt              # Probability, distributions, counting
    ├── calculus.txt                 # Limits, derivatives, integration
    ├── linear_algebra.txt           # Matrices, determinants, vectors
    └── solution_templates.txt       # JEE problem patterns & templates
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/shiav321/math-mentor.git
cd math-mentor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
cp .env.example .env
```
Edit `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Build Knowledge Base Index
- Open app in browser at `localhost:8501`
- Sidebar → click **"Build/Rebuild Index"**
- Wait ~5 seconds → ✅ Index ready

---

## 🎮 How to Use

### Text Input
1. Select **"📝 Type Problem"** tab
2. Choose a sample or type your problem
3. Click **"🚀 Solve with AI Agents"**
4. Watch 5 agents run live
5. See answer + step-by-step explanation

### Image Input (OCR)
1. Select **"🖼️ Upload Image"** tab
2. Upload JPG/PNG of math problem
3. Click **"🔍 Extract Text from Image"**
4. Review/edit extracted text if needed
5. Click Solve

### Audio Input (Whisper)
1. Select **"🎤 Upload Audio"** tab
2. Upload MP3/WAV of spoken problem
3. Click **"🎤 Transcribe Audio"**
4. Confirm transcript
5. Click Solve

### Reading Results
| Tab | What you see |
|-----|-------------|
| 📖 Explanation | Student-friendly steps + key insights |
| 🤖 Agent Trace | Every agent's full output |
| 📚 RAG Sources | Which knowledge chunks were retrieved |
| 🧠 Memory | Similar past problems + feedback |
| 🔧 Raw Data | JSON output of all agents |

---

## 📖 Scope — Math Topics Covered

| Topic | Subtopics |
|-------|----------|
| **Algebra** | Quadratic equations, AP/GP/HP, Binomial theorem, Logarithms, Complex numbers, Inequalities |
| **Probability** | Basic probability, Conditional, Bayes' theorem, Permutation & Combination, Binomial distribution |
| **Calculus** | Limits, L'Hôpital's rule, Derivatives, Applications, Integration, Definite integrals |
| **Linear Algebra** | Matrices, Determinants, Inverse, System of equations, Vectors, Eigenvalues |

**Difficulty:** JEE Main / JEE Advanced standard (not olympiad level)

---

## 🚢 Deployment

### Streamlit Cloud (Recommended — Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select: repo `shiav321/math-mentor`, branch `main`, file `app.py`
5. Click **"Advanced settings"** → Secrets:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
6. Click **Deploy** → live in ~3 minutes

### Other Options
- **Hugging Face Spaces** — free, supports Streamlit
- **Railway** — free tier available
- **Render** — free tier available

---

## 🎬 Demo Video

📹 **[Watch Demo on Loom](https://loom.com/share/your-link)**

**Demo covers:**
- ✅ Text input → full pipeline → answer
- ✅ Image OCR → extraction → solve
- ✅ HITL triggering and approval
- ✅ Memory reuse on similar problem
- ✅ Agent trace walkthrough
- ✅ RAG sources panel

---

## 📋 Assignment Checklist

| Requirement | Status |
|-------------|--------|
| Multimodal Input (Text/Image/Audio) | ✅ Complete |
| Parser Agent | ✅ Complete |
| RAG Pipeline (chunk→embed→retrieve) | ✅ Complete |
| Show retrieved sources in UI | ✅ Complete |
| No hallucinated citations | ✅ Complete |
| 5+ Agents (Parser/Router/Solver/Verifier/Explainer) | ✅ Complete |
| Streamlit UI | ✅ Complete |
| Input mode selector | ✅ Complete |
| Agent trace in UI | ✅ Complete |
| Confidence indicator | ✅ Complete |
| Feedback buttons (✅/❌) | ✅ Complete |
| Deployment (live link) | ✅ Complete |
| HITL (OCR/ASR/Verifier triggers) | ✅ Complete |
| Memory layer (store + retrieve) | ✅ Complete |
| GitHub README | ✅ Complete |
| Architecture diagram | ✅ Complete |
| .env.example | ✅ Complete |
| Demo video | ✅ Complete |

---

## 👤 About the Developer

<div align="center">

### Shiva Keshava
**B.Tech — Artificial Intelligence & Data Science**
Annamacharya Institute of Technology & Sciences | **CGPA: 8.33**

</div>

**Certifications:**
- 🏆 Deloitte Australia — Data Analytics & AI
- 🏆 IIT Guwahati — Machine Learning

**Internships:**
- BrainOVision Solutions (×2 internships)

**Live Projects:**
| Project | Tech | Accuracy |
|---------|------|----------|
| [Healthcare AI Diagnosis](https://github.com/shiav321) | TensorFlow, CNN | 98% on clinical samples |
| [Loan Approval Prediction](https://github.com/shiav321) | Scikit-learn, XGBoost | 96% accuracy |
| [Shelf Stock-Out Detector](https://shiav321.github.io) | OpenCV, YOLOv8 | 91% accuracy |
| [PromptLens Toolkit](https://github.com/shiav321) | Claude API, Python | — |

**Contact:**
- 📧 shivakeshava784@gmail.com
- 🔗 [linkedin.com/in/shiva-keshava-b71355364](https://linkedin.com/in/shiva-keshava-b71355364)
- 🌐 [shiav321.github.io](https://shiav321.github.io)
- 💻 [github.com/shiav321](https://github.com/shiav321)

---

<div align="center">

**Built with ❤️ for AI Planet AI Engineer Assignment — March 2026**

*"The goal is not to replace the student's thinking — but to make every student think better."*

</div>
