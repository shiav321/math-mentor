"""
🎓 Math Mentor — AI-Powered JEE Math Problem Solver
Built for AI Planet AI Engineer Assignment

Features:
- Multimodal Input (Text / Image OCR / Audio Whisper)
- 5-Agent Pipeline (Parser → Router → Solver → Verifier → Explainer)
- RAG Pipeline (FAISS + OpenAI Embeddings)
- Human-in-the-Loop (HITL)
- Memory & Self-Learning
- Full Agent Trace UI
"""

import streamlit as st
import os
import json
import tempfile
import base64
from pathlib import Path

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Math Mentor — AI JEE Solver",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── IMPORTS ───────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("⚠️ OpenAI package not installed. Run: pip install openai")
    st.stop()

# Import our modules
import sys
sys.path.insert(0, os.path.dirname(__file__))

from agents.agents import run_full_pipeline, parser_agent, safe_parse_json
from rag.rag_pipeline import retrieve, format_retrieved_context, build_index, get_index_stats, load_index
from memory.memory_manager import (store_solved_problem, store_correction,
                                    store_ocr_correction, find_similar_problems,
                                    format_similar_for_context, get_memory_stats)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #0B6E6E 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .agent-card {
        background: #f8fafc;
        border-left: 4px solid #0B6E6E;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .agent-card-pending {
        background: #f8fafc;
        border-left: 4px solid #e5e7eb;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        opacity: 0.5;
    }
    .answer-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px solid #16a34a;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }
    .hitl-box {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    .source-tag {
        background: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
    .step-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px;
        margin: 6px 0;
        border-left: 4px solid #4F46E5;
    }
    .confidence-high { color: #16a34a; font-weight: bold; }
    .confidence-med  { color: #d97706; font-weight: bold; }
    .confidence-low  { color: #dc2626; font-weight: bold; }
    .memory-badge {
        background: #8b5cf6;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)


# ── INITIALIZE CLIENT ─────────────────────────────────────────
def get_client():
    api_key = st.session_state.get("api_key") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


# ── SESSION STATE ──────────────────────────────────────────────
def init_state():
    defaults = {
        "pipeline_result": None,
        "current_input": "",
        "input_type": "text",
        "ocr_text": "",
        "audio_transcript": "",
        "hitl_active": False,
        "hitl_reason": "",
        "user_corrected_answer": "",
        "feedback_given": False,
        "solving": False,
        "agent_steps": [],
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "index_built": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Math Mentor")
    st.markdown("*AI-Powered JEE Problem Solver*")
    st.divider()

    # API Key input
    api_key_input = st.text_input(
        "🔑 OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your OpenAI API key"
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.divider()

    # Build RAG Index
    st.markdown("### 📚 Knowledge Base")
    stats = get_index_stats()
    if stats["index_built"]:
        st.success(f"✅ Index ready: {stats['total_chunks']} chunks")
        for topic, count in stats.get("topics", {}).items():
            st.caption(f"• {topic}: {count} chunks")
    else:
        st.warning("⚠️ Index not built yet")

    if st.button("🔨 Build/Rebuild Index", use_container_width=True):
        client = get_client()
        if client:
            with st.spinner("Building knowledge base index..."):
                try:
                    build_index(client)
                    st.session_state.index_built = True
                    st.success("✅ Index built!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Add API key first")

    st.divider()

    # Memory Stats
    st.markdown("### 🧠 Memory")
    mem_stats = get_memory_stats()
    st.metric("Problems Solved", mem_stats["total_problems_solved"])
    st.metric("Corrections Learned", mem_stats["total_corrections"])
    st.metric("OCR Rules", mem_stats["ocr_rules_learned"])

    if mem_stats["topics_breakdown"]:
        st.caption("**Topics:**")
        for topic, count in mem_stats["topics_breakdown"].items():
            st.caption(f"  • {topic}: {count}")

    st.divider()

    # About
    st.markdown("### 🤖 5-Agent System")
    agents_info = [
        ("1️⃣", "Parser Agent", "Structures raw input"),
        ("2️⃣", "Router Agent", "Plans solution strategy"),
        ("3️⃣", "Solver Agent", "Solves with RAG context"),
        ("4️⃣", "Verifier Agent", "Checks correctness"),
        ("5️⃣", "Explainer Agent", "Student-friendly steps"),
    ]
    for emoji, name, desc in agents_info:
        st.caption(f"{emoji} **{name}**: {desc}")

    st.divider()
    st.caption("Built by **Shiva Keshava**")
    st.caption("Tools: Python • OpenAI • FAISS • Streamlit")


# ── MAIN HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🎓 Math Mentor</h1>
    <p style="margin:4px 0 0 0; opacity:0.85;">
        AI-Powered JEE Math Solver — RAG + Multi-Agent + Memory + HITL
    </p>
</div>
""", unsafe_allow_html=True)


# ── INPUT SECTION ──────────────────────────────────────────────
st.markdown("## 📥 Input Your Math Problem")

tab_text, tab_image, tab_audio = st.tabs([
    "📝 Type Problem", "🖼️ Upload Image", "🎤 Upload Audio"
])

problem_input = ""
input_type = "text"

# ── TEXT INPUT TAB ────────────────────────────────────────────
with tab_text:
    st.markdown("Type or paste your JEE math problem below:")

    sample_problems = {
        "Select a sample...": "",
        "Quadratic: Find roots of 2x²-5x+3=0":
            "Find the roots of the equation 2x² - 5x + 3 = 0",
        "Probability: At least one head in 3 coin flips":
            "Three coins are tossed simultaneously. Find the probability of getting at least one head.",
        "Calculus: Differentiate x³sin(x)":
            "Find the derivative of f(x) = x³ sin(x)",
        "Limit: Evaluate lim(x→0) sin(3x)/x":
            "Evaluate the limit: lim(x→0) sin(3x)/x",
        "Permutation: 5 students in 3 chairs":
            "In how many ways can 5 students be arranged in 3 chairs?",
        "Matrix: Find determinant of [[1,2],[3,4]]":
            "Find the determinant of the matrix A = [[1,2],[3,4]]",
    }

    selected_sample = st.selectbox("🎯 Try a sample problem:", list(sample_problems.keys()))
    if selected_sample != "Select a sample...":
        default_text = sample_problems[selected_sample]
    else:
        default_text = ""

    text_input = st.text_area(
        "Your problem:",
        value=default_text,
        height=120,
        placeholder="e.g. Find the roots of x² - 5x + 6 = 0",
    )

    if text_input:
        problem_input = text_input
        input_type = "text"


# ── IMAGE INPUT TAB ──────────────────────────────────────────
with tab_image:
    st.markdown("Upload a photo/screenshot of your math problem:")
    uploaded_image = st.file_uploader(
        "Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("**OCR Extraction:**")

            if st.button("🔍 Extract Text from Image", use_container_width=True):
                with st.spinner("Running OCR..."):
                    try:
                        import easyocr
                        import numpy as np
                        from PIL import Image
                        import io

                        image_bytes = uploaded_image.read()
                        image = Image.open(io.BytesIO(image_bytes))
                        img_array = np.array(image)

                        reader = easyocr.Reader(['en'], gpu=False)
                        results = reader.readtext(img_array)

                        extracted_text = " ".join([r[1] for r in results])
                        confidence_scores = [r[2] for r in results]
                        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

                        st.session_state.ocr_text = extracted_text
                        st.session_state.ocr_confidence = avg_confidence

                        if avg_confidence < 0.6:
                            st.warning(f"⚠️ Low OCR confidence ({avg_confidence:.0%}) — please review and edit")
                            st.session_state.hitl_active = True
                            st.session_state.hitl_reason = f"Low OCR confidence: {avg_confidence:.0%}"

                    except ImportError:
                        st.warning("EasyOCR not installed. Using manual input mode.")
                        st.session_state.ocr_text = ""
                    except Exception as e:
                        st.error(f"OCR Error: {e}")
                        st.session_state.ocr_text = ""

            if st.session_state.ocr_text:
                conf = st.session_state.get("ocr_confidence", 1.0)
                conf_color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                st.caption(f"{conf_color} OCR Confidence: {conf:.0%}")

            # Allow editing OCR result
            ocr_editable = st.text_area(
                "Extracted text (edit if needed):",
                value=st.session_state.ocr_text,
                height=100,
                placeholder="OCR extracted text will appear here...",
                key="ocr_edit"
            )

            if ocr_editable:
                # Store correction if user edited
                if ocr_editable != st.session_state.ocr_text and st.session_state.ocr_text:
                    store_ocr_correction(st.session_state.ocr_text, ocr_editable)

                problem_input = ocr_editable
                input_type = "image"
                st.success(f"✅ Using: {ocr_editable[:60]}...")


# ── AUDIO INPUT TAB ──────────────────────────────────────────
with tab_audio:
    st.markdown("Upload an audio file of your spoken math question:")
    uploaded_audio = st.file_uploader(
        "Upload audio (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"],
        key="audio_uploader"
    )

    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")

        if st.button("🎤 Transcribe Audio", use_container_width=True):
            with st.spinner("Transcribing with Whisper..."):
                try:
                    client = get_client()
                    if not client:
                        st.error("Add API key first")
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=f".{uploaded_audio.name.split('.')[-1]}",
                            delete=False
                        ) as tmp:
                            tmp.write(uploaded_audio.read())
                            tmp_path = tmp.name

                        with open(tmp_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                language="en"
                            )

                        os.unlink(tmp_path)
                        transcript_text = transcript.text

                        # Handle math-specific phrases
                        math_corrections = {
                            "square root of": "√(",
                            "raised to the power of": "^",
                            "x squared": "x²",
                            "x cubed": "x³",
                            "pi": "π",
                            "infinity": "∞",
                        }
                        for phrase, symbol in math_corrections.items():
                            transcript_text = transcript_text.replace(phrase, symbol)

                        st.session_state.audio_transcript = transcript_text

                        # Check clarity
                        if len(transcript_text.split()) < 4:
                            st.warning("⚠️ Transcript seems incomplete — please confirm")
                            st.session_state.hitl_active = True
                            st.session_state.hitl_reason = "Short/unclear audio transcript"

                except ImportError:
                    st.error("OpenAI not installed properly")
                except Exception as e:
                    st.error(f"Transcription error: {e}")

        # Show and allow editing transcript
        transcript_editable = st.text_area(
            "Transcript (edit if needed):",
            value=st.session_state.audio_transcript,
            height=100,
            placeholder="Transcript will appear here...",
            key="transcript_edit"
        )

        if transcript_editable:
            problem_input = transcript_editable
            input_type = "audio"
            st.success(f"✅ Using: {transcript_editable[:60]}...")


# ── SOLVE BUTTON ──────────────────────────────────────────────
st.divider()

col_solve, col_clear = st.columns([3, 1])
with col_solve:
    solve_btn = st.button(
        "🚀 Solve with AI Agents",
        use_container_width=True,
        type="primary",
        disabled=not problem_input or not st.session_state.api_key,
    )
with col_clear:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.pipeline_result = None
        st.session_state.feedback_given = False
        st.session_state.hitl_active = False
        st.rerun()

if not st.session_state.api_key:
    st.info("👈 Add your OpenAI API key in the sidebar to start")


# ── PIPELINE EXECUTION ────────────────────────────────────────
if solve_btn and problem_input and st.session_state.api_key:
    client = get_client()
    st.session_state.pipeline_result = None
    st.session_state.feedback_given = False
    st.session_state.agent_steps = []

    # Agent trace container
    st.markdown("## 🤖 Agent Pipeline — Live Trace")
    trace_container = st.container()

    with trace_container:
        agent_placeholders = {}
        agent_names = [
            ("1️⃣ Parser Agent", "Structuring your problem..."),
            ("2️⃣ Router Agent", "Planning solution strategy..."),
            ("3️⃣ Solver Agent", "Solving with RAG knowledge base..."),
            ("4️⃣ Verifier Agent", "Checking solution correctness..."),
            ("5️⃣ Explainer Agent", "Writing student-friendly explanation..."),
        ]
        placeholders = [st.empty() for _ in agent_names]
        for i, (name, msg) in enumerate(agent_names):
            with placeholders[i]:
                st.markdown(
                    f'<div class="agent-card-pending">⏳ <b>{name}</b>: Waiting...</div>',
                    unsafe_allow_html=True
                )

    progress_bar = st.progress(0, text="Starting agents...")

    def on_progress(step, msg):
        progress_bar.progress(step / 6, text=msg)
        with placeholders[step - 1]:
            st.markdown(
                f'<div class="agent-card">✅ <b>{agent_names[step-1][0]}</b>: {msg}</div>',
                unsafe_allow_html=True
            )

    try:
        # Retrieve from RAG
        on_progress(0, "")
        progress_bar.progress(0.1, text="Retrieving from knowledge base...")

        # Build index if needed
        index_data = load_index()
        if not index_data.get("docs"):
            progress_bar.progress(0.05, text="Building knowledge base index first...")
            build_index(client)

        rag_results = retrieve(problem_input, client, top_k=5)
        retrieved_context = format_retrieved_context(rag_results)

        # Memory context
        parsed_quick = parser_agent(problem_input, client)
        similar = find_similar_problems(
            problem_input,
            topic=parsed_quick.get("topic", ""),
            limit=2
        )
        memory_context = format_similar_for_context(similar)

        # Run pipeline
        result = run_full_pipeline(
            raw_input=problem_input,
            retrieved_context=retrieved_context,
            memory_context=memory_context,
            client=client,
            progress_callback=on_progress,
        )

        result["retrieved_context"] = retrieved_context
        result["rag_results"] = [(d, s) for d, s in rag_results]
        result["memory_context"] = memory_context
        result["input_text"] = problem_input
        result["input_type"] = input_type

        progress_bar.progress(1.0, text="✅ All agents complete!")
        st.session_state.pipeline_result = result

        # Check if HITL needed
        verifier = result.get("verifier", {})
        parsed = result.get("parsed", {})
        if (verifier.get("trigger_hitl") or
                parsed.get("needs_clarification") or
                st.session_state.hitl_active):
            st.session_state.hitl_active = True
            st.session_state.hitl_reason = (
                verifier.get("hitl_reason") or
                parsed.get("clarification_reason") or
                st.session_state.hitl_reason or
                "Please review this solution"
            )

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        progress_bar.progress(0, text="Error occurred")
        st.exception(e)


# ── DISPLAY RESULTS ────────────────────────────────────────────
if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    parsed    = result.get("parsed", {})
    routing   = result.get("routing", {})
    solution  = result.get("solution", {})
    verifier  = result.get("verifier", {})
    expl      = result.get("explanation", {})
    rag_res   = result.get("rag_results", [])

    st.markdown("---")

    # ── FINAL ANSWER BOX ─────────────────────────────────────
    verdict = verifier.get("verdict", "UNCERTAIN")
    conf = solution.get("confidence", 0.5)
    v_conf = verifier.get("confidence", 0.5)

    verdict_colors = {
        "CORRECT": "#16a34a",
        "LIKELY_CORRECT": "#0B6E6E",
        "UNCERTAIN": "#d97706",
        "INCORRECT": "#dc2626",
    }
    verdict_color = verdict_colors.get(verdict, "#374151")

    st.markdown(f"""
    <div class="answer-box">
        <h2 style="margin:0 0 8px 0;">
            🎯 {expl.get('final_answer_boxed', solution.get('final_answer', 'See solution below'))}
        </h2>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:8px;">
            <span style="background:{verdict_color};color:white;padding:3px 10px;border-radius:12px;font-size:13px;font-weight:bold;">
                {verdict}
            </span>
            <span style="background:#e0f2fe;color:#0369a1;padding:3px 10px;border-radius:12px;font-size:13px;">
                📊 Topic: {routing.get('topic','?').title()} / {routing.get('subtopic','?')}
            </span>
            <span style="background:#f3e8ff;color:#7c3aed;padding:3px 10px;border-radius:12px;font-size:13px;">
                🎚️ {expl.get('difficulty_rating','Medium')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence meters
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Solver Confidence", f"{solution.get('confidence',0.5):.0%}")
    with c2:
        st.metric("Verifier Confidence", f"{verifier.get('confidence',0.5):.0%}")
    with c3:
        issues = len(verifier.get("issues_found", []))
        st.metric("Issues Found", issues, delta=f"-{issues} issues" if issues else "Clean")

    # ── HUMAN IN THE LOOP ─────────────────────────────────────
    if st.session_state.hitl_active:
        st.markdown("---")
        st.markdown(f"""
        <div class="hitl-box">
            <h3 style="margin:0 0 8px 0;">⚠️ Human Review Requested</h3>
            <p style="margin:0;"><b>Reason:</b> {st.session_state.hitl_reason}</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_r = st.columns(2)
        with col_a:
            if st.button("✅ Approve Solution", use_container_width=True, type="primary"):
                st.session_state.hitl_active = False
                st.success("Solution approved!")
                st.rerun()

        with col_r:
            with st.expander("❌ Reject & Provide Correction"):
                correction = st.text_area("Your corrected answer:")
                correction_note = st.text_input("What was wrong?")
                if st.button("Submit Correction"):
                    if correction:
                        store_correction(
                            result.get("input_text", ""),
                            correction,
                            correction_note
                        )
                        st.success("✅ Correction stored — system will learn from this!")
                        st.session_state.hitl_active = False
                        st.rerun()

    # ── TABS FOR RESULTS ──────────────────────────────────────
    tab_explain, tab_trace, tab_rag, tab_memory, tab_raw = st.tabs([
        "📖 Explanation", "🤖 Agent Trace", "📚 RAG Sources",
        "🧠 Memory", "🔧 Raw Data"
    ])

    # ── EXPLANATION TAB ───────────────────────────────────────
    with tab_explain:
        st.markdown(f"### {expl.get('title', 'Solution')}")
        st.info(expl.get("introduction", ""))

        steps = expl.get("steps", [])
        if steps:
            for step in steps:
                st.markdown(f"""
                <div class="step-card">
                    <b>Step {step.get('step_number', '')}: {step.get('heading', '')}</b><br>
                    <span style="color:#374151">{step.get('explanation', '')}</span>
                    {'<br><code style="background:#f1f5f9;padding:4px 8px;border-radius:4px;">' + step.get('math', '') + '</code>' if step.get('math') else ''}
                    {'<br><small style="color:#6b7280;font-style:italic;">Why: ' + step.get('why', '') + '</small>' if step.get('why') else ''}
                </div>
                """, unsafe_allow_html=True)

        col_insight, col_tip = st.columns(2)
        with col_insight:
            if expl.get("key_insight"):
                st.success(f"💡 **Key Insight:** {expl['key_insight']}")
        with col_tip:
            if expl.get("jee_tip"):
                st.info(f"📝 **JEE Tip:** {expl['jee_tip']}")

        if expl.get("common_mistakes"):
            with st.expander("⚠️ Common Mistakes to Avoid"):
                for m in expl["common_mistakes"]:
                    st.markdown(f"• {m}")

        if expl.get("similar_problems_hint"):
            st.caption(f"📌 Similar problems: {expl['similar_problems_hint']}")

    # ── AGENT TRACE TAB ───────────────────────────────────────
    with tab_trace:
        st.markdown("### 🔍 Full Agent Pipeline Trace")

        with st.expander("1️⃣ Parser Agent Output", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Problem text:** {parsed.get('problem_text', '')}")
                st.markdown(f"**Topic:** `{parsed.get('topic', '')}`")
                st.markdown(f"**Problem type:** `{parsed.get('problem_type', '')}`")
            with col2:
                st.markdown(f"**Variables:** {', '.join(parsed.get('variables', []))}")
                st.markdown(f"**Find:** {parsed.get('find', '')}")
                conf_p = parsed.get("confidence", 0.5)
                cls = "confidence-high" if conf_p > 0.8 else "confidence-med" if conf_p > 0.6 else "confidence-low"
                st.markdown(f'**Confidence:** <span class="{cls}">{conf_p:.0%}</span>', unsafe_allow_html=True)
            if parsed.get("ocr_corrections"):
                st.caption(f"OCR Corrections: {', '.join(parsed['ocr_corrections'])}")
            if parsed.get("needs_clarification"):
                st.warning(f"⚠️ Clarification needed: {parsed.get('clarification_reason')}")

        with st.expander("2️⃣ Router Agent Output", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Topic:** `{routing.get('topic', '')}`")
                st.markdown(f"**Subtopic:** `{routing.get('subtopic', '')}`")
                st.markdown(f"**Complexity:** `{routing.get('complexity', '')}`")
                st.markdown(f"**Strategy:** {routing.get('solution_strategy', '')}")
            with col2:
                st.markdown(f"**Key concepts:** {', '.join(routing.get('key_concepts', []))}")
                st.markdown(f"**Est. steps:** {routing.get('estimated_steps', '')}")
                st.markdown(f"**Calculator needed:** {'Yes' if routing.get('requires_calculator') else 'No'}")
            if routing.get("warnings"):
                for w in routing["warnings"]:
                    st.warning(f"⚠️ {w}")

        with st.expander("3️⃣ Solver Agent Output", expanded=False):
            st.markdown("**Solution Steps:**")
            for i, step in enumerate(solution.get("solution_steps", []), 1):
                st.markdown(f"{i}. {step}")
            st.markdown(f"**Formulas Used:** {', '.join(solution.get('formulas_used', []))}")
            st.markdown(f"**Final Answer:** `{solution.get('final_answer', '')}`")
            if solution.get("assumptions_made"):
                st.caption(f"Assumptions: {', '.join(solution['assumptions_made'])}")

        with st.expander("4️⃣ Verifier Agent Output", expanded=False):
            verdict_color_map = {
                "CORRECT": "green", "LIKELY_CORRECT": "blue",
                "UNCERTAIN": "orange", "INCORRECT": "red"
            }
            vc = verdict_color_map.get(verifier.get("verdict", ""), "gray")
            st.markdown(f"**Verdict:** :{vc}[{verifier.get('verdict', 'UNCERTAIN')}]")
            st.markdown(f"**Confidence:** {verifier.get('confidence', 0.5):.0%}")
            st.markdown(f"**Domain check:** {verifier.get('domain_check', '')}")
            st.markdown(f"**Arithmetic check:** {verifier.get('arithmetic_check', '')}")
            if verifier.get("issues_found"):
                st.markdown("**Issues:**")
                for issue in verifier["issues_found"]:
                    st.warning(f"• {issue}")
            if verifier.get("corrections_needed"):
                st.markdown("**Corrections needed:**")
                for c in verifier["corrections_needed"]:
                    st.error(f"• {c}")
            if verifier.get("verification_notes"):
                st.caption(verifier["verification_notes"])

        with st.expander("5️⃣ Explainer Agent Output", expanded=False):
            st.json({
                "title": expl.get("title"),
                "key_insight": expl.get("key_insight"),
                "difficulty": expl.get("difficulty_rating"),
                "steps_count": len(expl.get("steps", [])),
                "jee_tip": expl.get("jee_tip"),
            })

    # ── RAG SOURCES TAB ───────────────────────────────────────
    with tab_rag:
        st.markdown("### 📚 Retrieved Knowledge Base Sources")
        st.caption(f"Query: *{routing.get('rag_search_query', result.get('input_text', ''))[:80]}*")

        if rag_res:
            for i, (doc, score) in enumerate(rag_res, 1):
                with st.expander(
                    f"📄 Source {i}: {doc['source']} (Relevance: {score:.2f})",
                    expanded=(i == 1)
                ):
                    st.markdown(
                        f'<span class="source-tag">{doc["topic"]}</span> '
                        f'<span class="source-tag">Chunk {doc["chunk_id"]}</span> '
                        f'<span class="source-tag">Score: {score:.3f}</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(doc["text"])
        else:
            st.info("No sources retrieved. Build the knowledge base index first.")
            if st.button("Build Index Now"):
                client = get_client()
                if client:
                    with st.spinner("Building..."):
                        build_index(client)
                    st.rerun()

    # ── MEMORY TAB ────────────────────────────────────────────
    with tab_memory:
        st.markdown("### 🧠 Memory & Learning")

        mem_stats = get_memory_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Problems in Memory", mem_stats["total_problems_solved"])
        col2.metric("Positive Feedback", mem_stats["positive_feedback"])
        col3.metric("Corrections Learned", mem_stats["total_corrections"])

        if result.get("memory_context"):
            st.markdown("**Similar problems found in memory:**")
            st.code(result["memory_context"])
        else:
            st.info("No similar problems in memory yet. Solve more problems to build memory!")

        # Feedback buttons
        if not st.session_state.feedback_given:
            st.markdown("---")
            st.markdown("**📊 Was this solution helpful?**")
            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button("✅ Correct — Save to Memory", use_container_width=True):
                    store_solved_problem(
                        input_text=result.get("input_text", ""),
                        parsed_problem=parsed,
                        retrieved_context=result.get("retrieved_context", "")[:500],
                        final_answer=solution.get("final_answer", ""),
                        explanation=json.dumps(expl.get("steps", []))[:500],
                        verifier_outcome=verifier,
                        user_feedback="correct",
                        input_type=result.get("input_type", "text")
                    )
                    st.success("✅ Saved to memory!")
                    st.session_state.feedback_given = True
                    st.rerun()

            with col_no:
                with st.expander("❌ Incorrect — Add Comment"):
                    feedback_comment = st.text_input("What was wrong?")
                    if st.button("Submit Feedback"):
                        store_solved_problem(
                            input_text=result.get("input_text", ""),
                            parsed_problem=parsed,
                            retrieved_context=result.get("retrieved_context", "")[:500],
                            final_answer=solution.get("final_answer", ""),
                            explanation=json.dumps(expl.get("steps", []))[:500],
                            verifier_outcome=verifier,
                            user_feedback=f"incorrect: {feedback_comment}",
                            input_type=result.get("input_type", "text")
                        )
                        st.warning("Feedback stored — system will improve!")
                        st.session_state.feedback_given = True
                        st.rerun()
        else:
            st.success("✅ Feedback recorded — thank you!")

    # ── RAW DATA TAB ──────────────────────────────────────────
    with tab_raw:
        st.markdown("### 🔧 Raw Pipeline Data (Debug)")
        with st.expander("Parsed Problem JSON"):
            st.json(parsed)
        with st.expander("Routing JSON"):
            st.json(routing)
        with st.expander("Solution JSON"):
            st.json(solution)
        with st.expander("Verifier JSON"):
            st.json(verifier)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🎓 <b>Math Mentor</b> — Built by Shiva Keshava | "
    "AI Planet AI Engineer Assignment | "
    "Python • OpenAI GPT-4o-mini • FAISS • Streamlit</small></center>",
    unsafe_allow_html=True
)
