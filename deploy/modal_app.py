"""
Qaida Pronunciation Checker — Modal serverless deployment (hybrid GPU + CPU).

Two endpoints:
  GPU: https://huzaifa-karsaaz--qaida-scorer-fastapi-app.modal.run
  CPU: https://huzaifa-karsaaz--qaida-scorer-fastapi-app-cpu.modal.run

Frontend tries GPU first (8s timeout), falls back to CPU automatically.
Accuracy is identical on both — only speed differs.
"""

import os
import sys
from pathlib import Path

import modal

app = modal.App("qaida-scorer")

# This file lives at <repo>/deploy/modal_app.py. Paths below are absolute so
# `modal deploy` works from any working directory.
PKG = Path(__file__).resolve().parents[1]  # .../qaida_project

MODAL_MODELS_DIR = "/modal_models"
MODAL_AUDIO_DIR  = "/modal_audio"
MODAL_HF_CACHE   = "/hf_cache"

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg", "libsndfile1"])
    .pip_install(
        "torch>=2.6.0",
        "torchaudio>=2.6.0",
        "transformers>=4.44.0,<5.0.0",
        "librosa==0.11.0",
        "soundfile",
        "scipy",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "numpy<2.4",
        "numba",
        "scikit-learn",
    )
    .add_local_file(str(PKG / "__init__.py"),   "/project/qaida_project/__init__.py")
    .add_local_dir(str(PKG / "config"),         "/project/qaida_project/config")
    .add_local_dir(str(PKG / "embeddings"),     "/project/qaida_project/embeddings")
    .add_local_dir(str(PKG / "inference"),      "/project/qaida_project/inference")
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

models_vol = modal.Volume.from_name("qaida-models", create_if_missing=True)
audio_vol  = modal.Volume.from_name("qaida-audio",  create_if_missing=True)
hf_vol     = modal.Volume.from_name("hf-cache",     create_if_missing=True)

VOLUMES = {
    MODAL_MODELS_DIR: models_vol,
    MODAL_AUDIO_DIR:  audio_vol,
    MODAL_HF_CACHE:   hf_vol,
}

# ---------------------------------------------------------------------------
# Shared FastAPI app builder (used by both GPU and CPU functions)
# ---------------------------------------------------------------------------

def build_app():
    import json
    import tempfile
    import subprocess
    import torch
    import soundfile as sf
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    from typing import Optional, List

    sys.path.insert(0, "/project")

    DATASET_PATH    = f"{MODAL_MODELS_DIR}/unified_dataset.json"
    EMBEDDINGS_PATH = f"{MODAL_MODELS_DIR}/reference_embeddings.pkl"
    MODEL_DIR       = f"{MODAL_MODELS_DIR}/xlsr53-qaida-contrastive"

    # ── Pydantic schemas ──

    class AcousticDetail(BaseModel):
        acoustic_score: float
        cosine_similarity: float
        dtw_similarity: float = 0.0
        hybrid_similarity: float = 0.0
        margin: float = 0.0
        rank_in_lesson: int = 1
        lesson_size: int = 1
        detection_margin: float = 0.0
        detection_uncertain: bool = False
        error: Optional[str] = None

    class TopMatch(BaseModel):
        audio_id: str
        arabic_text: str
        similarity: float
        mean_sim: float = 0.0
        dtw_sim: float = 0.0

    class ScoringResponse(BaseModel):
        combined_score: float
        feedback: str
        audio_id: str
        expected_arabic: str
        detected_arabic: str
        detected_audio_id: str
        detected_similarity: float
        lesson_num: int = 0
        pass_threshold: float = 0.6
        good_threshold: float = 0.8
        compute_mode: str = "unknown"
        acoustic: AcousticDetail
        top3: List[TopMatch] = []

    class LessonInfo(BaseModel):
        lesson_id: str
        lesson_num: int
        type: str
        item_count: int

    class LessonItem(BaseModel):
        audio_id: str
        word_id: str
        arabic_text: str
        transliteration: str
        type: str
        sub_type: Optional[str] = None
        duration_s: Optional[float] = None

    class HealthResponse(BaseModel):
        status: str
        models_loaded: bool
        gpu_available: bool
        compute_mode: str
        total_entries: int

    # ── Dataset loader ──

    WIN_PREFIX = "C:/Users/PC/Desktop/Qaida_audios/"

    def load_dataset():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            p = e.get("audio_path", "").replace("\\", "/")
            if WIN_PREFIX in p:
                rel = p.split(WIN_PREFIX, 1)[-1]
                e["audio_path"] = f"{MODAL_AUDIO_DIR}/{rel}"
        return entries, {e["audio_id"]: e for e in entries}

    # ── Lifespan ──

    @asynccontextmanager
    async def lifespan(fapp: FastAPI):
        entries, by_id = load_dataset()
        fapp.state.entries = entries
        fapp.state.by_id = by_id

        os.environ["HF_HUB_CACHE"]      = MODAL_HF_CACHE
        os.environ["TRANSFORMERS_CACHE"] = MODAL_HF_CACHE
        os.environ["HF_HUB_OFFLINE"]     = "0"

        import qaida_project.config.settings as settings
        settings.REFERENCE_EMBEDDINGS_PATH = EMBEDDINGS_PATH
        settings.EMBEDDING_FINETUNED_DIR   = MODEL_DIR
        settings.EMBEDDING_MODEL_NAME      = "facebook/wav2vec2-large-xlsr-53"

        from qaida_project.inference.combined_scorer import QaidaPronunciationScorer
        fapp.state.scorer = QaidaPronunciationScorer()
        fapp.state.compute_mode = "gpu" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"Scorer ready on {device_name} [{fapp.state.compute_mode}]")
        yield
        del fapp.state.scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fapp = FastAPI(title="Qaida Pronunciation Checker API", version="3.2.0", lifespan=lifespan)
    fapp.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ── Endpoints ──

    @fapp.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            models_loaded=hasattr(fapp.state, "scorer"),
            gpu_available=torch.cuda.is_available(),
            compute_mode=getattr(fapp.state, "compute_mode", "unknown"),
            total_entries=len(fapp.state.entries),
        )

    @fapp.post("/api/v1/score", response_model=ScoringResponse)
    async def score(audio: UploadFile = File(...), audio_id: str = Form(...)):
        if audio_id not in fapp.state.by_id:
            raise HTTPException(404, f"audio_id '{audio_id}' not found")
        entry = fapp.state.by_id[audio_id]

        content_type = audio.content_type or "audio/webm"
        suffix = ".webm" if "webm" in content_type else \
                 ".ogg"  if "ogg"  in content_type else \
                 ".mp4"  if "mp4"  in content_type else ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        try:
            wav_path = tmp_path + ".wav"
            conv = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True,
            )
            if conv.returncode != 0:
                raise HTTPException(400, "Could not decode audio. Please record again.")
            os.unlink(tmp_path)
            tmp_path = wav_path

            try:
                info = sf.info(tmp_path)
                if info.duration < 0.1:
                    raise HTTPException(400, "Recording too short.")
                if info.duration > 30.0:
                    raise HTTPException(400, "Recording too long (max 30s).")
            except HTTPException:
                raise
            except Exception:
                pass

            result = fapp.state.scorer.score(
                student_audio_path=tmp_path,
                audio_id=audio_id,
                expected_arabic=entry["arabic_text"],
            )
            ac = result["acoustic"]
            return ScoringResponse(
                combined_score=result["combined_score"],
                feedback=result["feedback"],
                audio_id=audio_id,
                expected_arabic=result["expected_arabic"],
                detected_arabic=result.get("detected_arabic", ""),
                detected_audio_id=result.get("detected_audio_id", ""),
                detected_similarity=result.get("detected_similarity", 0.0),
                lesson_num=result.get("lesson_num", 0),
                pass_threshold=result.get("pass_threshold", 0.6),
                good_threshold=result.get("good_threshold", 0.8),
                compute_mode=getattr(fapp.state, "compute_mode", "unknown"),
                acoustic=AcousticDetail(
                    acoustic_score=ac.get("acoustic_score", 0.0),
                    cosine_similarity=ac.get("cosine_similarity", 0.0),
                    dtw_similarity=ac.get("dtw_similarity", 0.0),
                    hybrid_similarity=ac.get("hybrid_similarity", 0.0),
                    margin=ac.get("margin", 0.0),
                    rank_in_lesson=ac.get("rank_in_lesson", 1),
                    lesson_size=ac.get("lesson_size", 1),
                    detection_margin=ac.get("detection_margin", 0.0),
                    detection_uncertain=ac.get("detection_uncertain", False),
                ),
                top3=[
                    TopMatch(
                        audio_id=m["audio_id"],
                        arabic_text=m["arabic_text"],
                        similarity=m["similarity"],
                        mean_sim=m.get("mean_sim", 0.0),
                        dtw_sim=m.get("dtw_sim", 0.0),
                    )
                    for m in result.get("top3", [])
                ],
            )
        finally:
            os.unlink(tmp_path)

    @fapp.get("/api/v1/lessons", response_model=List[LessonInfo])
    async def list_lessons():
        seen: dict = {}
        for e in fapp.state.entries:
            n = e["lesson_num"]
            if n not in seen:
                seen[n] = {"lesson_id": e["lesson_id"], "lesson_num": n,
                           "type": e["type"], "item_count": 0}
            seen[n]["item_count"] += 1
        return [LessonInfo(**v) for v in sorted(seen.values(), key=lambda x: x["lesson_num"])]

    @fapp.get("/api/v1/lessons/{lesson_num}/items", response_model=List[LessonItem])
    async def get_items(lesson_num: int):
        items = [e for e in fapp.state.entries if e["lesson_num"] == lesson_num]
        if not items:
            raise HTTPException(404, f"Lesson {lesson_num} not found")
        return [
            LessonItem(
                audio_id=e["audio_id"], word_id=e["word_id"],
                arabic_text=e["arabic_text"], transliteration=e["transliteration"],
                type=e["type"], sub_type=e.get("sub_type"), duration_s=e.get("duration_s"),
            )
            for e in items
        ]

    @fapp.get("/api/v1/audio/{audio_id}")
    async def get_audio(audio_id: str):
        if audio_id not in fapp.state.by_id:
            raise HTTPException(404)
        path = fapp.state.by_id[audio_id]["audio_path"]
        if not os.path.exists(path):
            raise HTTPException(404, "Audio file not found")
        return FileResponse(path, media_type="audio/wav", filename=f"{audio_id}.wav")

    return fapp


# ---------------------------------------------------------------------------
# GPU function — fast (~15-25s), schedules when GPU available
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=["T4", "L4", "A10G"],
    volumes=VOLUMES,
    timeout=120,
    scaledown_window=600,
    memory=2048,
)
@modal.asgi_app()
def fastapi_app():
    return build_app()


# ---------------------------------------------------------------------------
# CPU function — always available (~30-60s), never fails to schedule
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=4.0,
    volumes=VOLUMES,
    timeout=180,
    scaledown_window=600,
    memory=4096,
)
@modal.asgi_app()
def fastapi_app_cpu():
    return build_app()
