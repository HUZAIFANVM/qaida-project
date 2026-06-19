"""
Qaida Pronunciation Checker - FastAPI Application.

Usage:
    uvicorn qaida_project.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import tempfile
import torch
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qaida_project.config.settings import UNIFIED_DATASET_JSON
from qaida_project.api.schemas import (
    ScoringResponse, LessonInfo, LessonItem, HealthResponse,
)


def load_dataset():
    """Load the unified dataset indexed by audio_id."""
    with open(UNIFIED_DATASET_JSON, "r", encoding="utf-8") as f:
        entries = json.load(f)
    by_audio_id = {e["audio_id"]: e for e in entries}
    return entries, by_audio_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    # Load dataset
    entries, by_audio_id = load_dataset()
    app.state.entries = entries
    app.state.by_audio_id = by_audio_id

    # Load scorer (this loads both Whisper + wav2vec2)
    from qaida_project.inference.combined_scorer import QaidaPronunciationScorer
    app.state.scorer = QaidaPronunciationScorer()

    yield

    # Cleanup
    del app.state.scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Qaida Pronunciation Checker",
    version="1.0.0",
    description="API for checking Qaida (Quran reading basics) pronunciation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=hasattr(app.state, "scorer"),
        gpu_available=torch.cuda.is_available(),
        total_entries=len(app.state.entries),
    )


@app.post("/api/v1/score", response_model=ScoringResponse)
async def score_pronunciation(
    audio: UploadFile = File(..., description="Student's WAV audio recording"),
    audio_id: str = Form(..., description="Reference audio ID (e.g., '01_001')"),
):
    """
    Score student pronunciation against reference.

    Upload a WAV file and provide the audio_id of the reference pronunciation.
    Returns a combined score with STT and acoustic breakdowns.
    """
    # Validate audio_id exists
    if audio_id not in app.state.by_audio_id:
        raise HTTPException(404, f"audio_id '{audio_id}' not found")

    entry = app.state.by_audio_id[audio_id]

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Validate audio file
        try:
            info = sf.info(tmp_path)
        except Exception:
            raise HTTPException(400, "Invalid audio file. Please upload a valid WAV file.")

        if info.duration < 0.2:
            raise HTTPException(400, "Audio too short (< 0.2 seconds)")
        if info.duration > 15.0:
            raise HTTPException(400, "Audio too long (> 15 seconds)")

        # Score
        result = app.state.scorer.score(
            student_audio_path=tmp_path,
            audio_id=audio_id,
            expected_arabic=entry["arabic_text"],
        )

        return ScoringResponse(**result)

    finally:
        os.unlink(tmp_path)


@app.get("/api/v1/lessons", response_model=List[LessonInfo])
async def list_lessons():
    """List all 16 Qaida lessons with item counts."""
    lessons = {}
    for entry in app.state.entries:
        lnum = entry["lesson_num"]
        if lnum not in lessons:
            lessons[lnum] = {
                "lesson_id": entry["lesson_id"],
                "lesson_num": lnum,
                "type": entry["type"],
                "item_count": 0,
            }
        lessons[lnum]["item_count"] += 1

    return [LessonInfo(**v) for v in sorted(lessons.values(), key=lambda x: x["lesson_num"])]


@app.get("/api/v1/lessons/{lesson_num}/items", response_model=List[LessonItem])
async def get_lesson_items(lesson_num: int):
    """Get all items for a specific lesson."""
    items = [e for e in app.state.entries if e["lesson_num"] == lesson_num]
    if not items:
        raise HTTPException(404, f"Lesson {lesson_num} not found")

    return [
        LessonItem(
            audio_id=e["audio_id"],
            word_id=e["word_id"],
            arabic_text=e["arabic_text"],
            transliteration=e["transliteration"],
            type=e["type"],
            sub_type=e.get("sub_type"),
            duration_s=e.get("duration_s"),
        )
        for e in items
    ]


@app.get("/api/v1/audio/{audio_id}")
async def get_reference_audio(audio_id: str):
    """Stream reference audio WAV file."""
    if audio_id not in app.state.by_audio_id:
        raise HTTPException(404, f"audio_id '{audio_id}' not found")

    audio_path = app.state.by_audio_id[audio_id]["audio_path"]
    if not os.path.exists(audio_path):
        raise HTTPException(500, "Reference audio file not found on disk")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{audio_id}.wav")
