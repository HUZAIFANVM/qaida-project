"""Gradio demo for Qaida Pronunciation Checker (acoustic-only scoring)."""

import os
import sys
import json
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gradio as gr
from qaida_project.config.settings import UNIFIED_DATASET_JSON

# --- Load dataset ---
with open(UNIFIED_DATASET_JSON, "r", encoding="utf-8") as f:
    ALL_ENTRIES = json.load(f)
BY_AUDIO_ID = {e["audio_id"]: e for e in ALL_ENTRIES}

# Build lesson dropdown choices
LESSONS = {}
for e in ALL_ENTRIES:
    lnum = e["lesson_num"]
    if lnum not in LESSONS:
        LESSONS[lnum] = e["type"]
LESSON_CHOICES = [f"L{n}: {LESSONS[n]}" for n in sorted(LESSONS)]

# Build item choices per lesson
ITEMS_BY_LESSON = {}
for e in ALL_ENTRIES:
    lnum = e["lesson_num"]
    ITEMS_BY_LESSON.setdefault(lnum, []).append(e)

# Copy reference audio files to a temp dir that Gradio can serve
TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "qaida_ref_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def get_safe_ref_audio(audio_path):
    """Copy reference audio to temp dir so Gradio can serve it."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    basename = os.path.basename(audio_path)
    dest = os.path.join(TEMP_AUDIO_DIR, basename)
    if not os.path.exists(dest):
        shutil.copy2(audio_path, dest)
    return dest


def get_items_for_lesson(lesson_str):
    """Return dropdown choices for items in the selected lesson."""
    if not lesson_str:
        return gr.update(choices=[], value=None), "", None
    lnum = int(lesson_str.split(":")[0].replace("L", ""))
    items = ITEMS_BY_LESSON.get(lnum, [])
    choices = [f"{e['audio_id']}  |  {e['arabic_text']}  ({e['transliteration']})" for e in items]
    return gr.update(choices=choices, value=choices[0] if choices else None), "", None


def get_item_info(item_str):
    """Show reference info and audio for selected item."""
    if not item_str:
        return "", None
    audio_id = item_str.split("|")[0].strip()
    entry = BY_AUDIO_ID.get(audio_id)
    if not entry:
        return "Not found", None
    info = (
        f"**Arabic:** {entry['arabic_text']}\n\n"
        f"**Transliteration:** {entry['transliteration']}\n\n"
        f"**Type:** {entry['type']}"
    )
    return info, get_safe_ref_audio(entry["audio_path"])


# --- Load models eagerly at startup ---
print("Loading scorer (this may take a moment)...")
os.environ["HF_HUB_OFFLINE"] = "1"
from qaida_project.inference.combined_scorer import QaidaPronunciationScorer
SCORER = QaidaPronunciationScorer()
print("Models loaded! Starting UI...")


def score_audio(item_str, audio_path):
    """Score the student's recording against the selected reference."""
    if not item_str:
        return "Please select a lesson item first."
    if audio_path is None:
        return "Please record or upload audio."

    audio_id = item_str.split("|")[0].strip()
    entry = BY_AUDIO_ID.get(audio_id)
    if not entry:
        return "Invalid item selected."

    try:
        result = SCORER.score(
            student_audio_path=audio_path,
            audio_id=audio_id,
            expected_arabic=entry["arabic_text"],
        )

        feedback_label = {
            "excellent": "Excellent ✅",
            "acceptable": "Acceptable 👍",
            "needs_practice": "Needs Practice ❌",
        }

        detected = result.get("detected_arabic", "")
        detected_id = result.get("detected_audio_id", "")
        match_correct = detected_id == audio_id
        top3 = result.get("top3", [])

        output = (
            f"## {feedback_label.get(result['feedback'], result['feedback'])}\n\n"
            f"**Expected:** {result['expected_arabic']}\n\n"
            f"**Detected:** {detected}"
        )
        uncertain = result["acoustic"].get("detection_uncertain", False)
        det_margin = result["acoustic"].get("detection_margin", 0.0)

        if match_correct:
            output += "  ✅ Correct match\n\n"
        else:
            det_entry = BY_AUDIO_ID.get(detected_id, {})
            det_translit = det_entry.get("transliteration", "")
            if uncertain:
                output += f"  ⚠️ Low confidence ({det_translit}) — margin {det_margin:.4f}\n\n"
            else:
                output += f"  ❌ ({det_translit})\n\n"

        ac = result["acoustic"]
        rank = ac.get("rank_in_lesson", "?")
        lsize = ac.get("lesson_size", "?")
        margin = ac.get("margin", 0.0)
        pass_t = result.get("pass_threshold", 0.6)
        good_t = result.get("good_threshold", 0.8)

        output += (
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| **Score** | **{result['combined_score']:.2f}** "
            f"(pass {pass_t:.2f} · good {good_t:.2f}) |\n"
            f"| Mean-pool sim | {ac.get('cosine_similarity', 0.0):.4f} |\n"
            f"| DTW sim | {ac.get('dtw_similarity', 0.0):.4f} |\n"
            f"| Hybrid sim | {ac.get('hybrid_similarity', 0.0):.4f} |\n"
            f"| Rank in lesson | {rank} / {lsize} |\n"
            f"| Margin (vs 2nd) | {margin:+.4f} |\n"
        )

        if top3:
            output += "\n**Top matches (same lesson):**\n\n"
            for i, m in enumerate(top3, 1):
                marker = " *" if m["audio_id"] == audio_id else ""
                t_entry = BY_AUDIO_ID.get(m["audio_id"], {})
                t_translit = t_entry.get("transliteration", "")
                m_dtw = m.get("dtw_sim", 0.0)
                m_mean = m.get("mean_sim", 0.0)
                output += (
                    f"{i}. {m['arabic_text']} ({t_translit}) — "
                    f"hybrid {m['similarity']:.4f} "
                    f"(mean {m_mean:.3f} · dtw {m_dtw:.3f}){marker}\n\n"
                )
        return output

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"**Error:** {e}"


# --- Build UI ---
with gr.Blocks(title="Qaida Pronunciation Checker") as demo:
    gr.Markdown("# Qaida Pronunciation Checker\nSelect a lesson item, listen to the reference, then record your pronunciation.")

    # Pre-populate first lesson items
    first_lnum = sorted(LESSONS.keys())[0]
    first_items = ITEMS_BY_LESSON[first_lnum]
    first_item_choices = [f"{e['audio_id']}  |  {e['arabic_text']}  ({e['transliteration']})" for e in first_items]

    with gr.Row():
        with gr.Column(scale=1):
            lesson_dd = gr.Dropdown(choices=LESSON_CHOICES, label="Lesson", value=LESSON_CHOICES[0])
            item_dd = gr.Dropdown(choices=first_item_choices, label="Item", value=first_item_choices[0], interactive=True)
            item_info = gr.Markdown("")
            ref_audio = gr.Audio(label="Reference Audio", type="filepath", interactive=False)

        with gr.Column(scale=1):
            student_audio = gr.Audio(label="Your Recording", sources=["microphone", "upload"], type="filepath")
            score_btn = gr.Button("Score Pronunciation", variant="primary", size="lg")
            result_md = gr.Markdown("")

    # Wire events
    lesson_dd.change(get_items_for_lesson, inputs=lesson_dd, outputs=[item_dd, item_info, ref_audio])
    item_dd.change(get_item_info, inputs=item_dd, outputs=[item_info, ref_audio])
    score_btn.click(score_audio, inputs=[item_dd, student_audio], outputs=result_md)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[TEMP_AUDIO_DIR],
    )
