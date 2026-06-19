"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel
from typing import Optional


class STTResult(BaseModel):
    stt_score: float
    predicted_text: str
    expected_text: str
    cer: float


class AcousticResult(BaseModel):
    acoustic_score: float
    cosine_similarity: float
    error: Optional[str] = None


class ScoringResponse(BaseModel):
    combined_score: float
    feedback: str  # "excellent", "acceptable", "needs_practice"
    stt: STTResult
    acoustic: AcousticResult
    audio_id: str
    expected_arabic: str


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
    total_entries: int
