"""Pronunciation scorer using contrastive XLSR-53 acoustic embeddings (v3).

v3: per-lesson thresholds + DTW + multi-prototype handled by AcousticScorer.
"""

from qaida_project.inference.acoustic_scorer import AcousticScorer
from qaida_project.config.settings import (
    REFERENCE_EMBEDDINGS_PATH, EMBEDDING_MODEL_NAME,
    EMBEDDING_FINETUNED_DIR, PASS_THRESHOLD, GOOD_THRESHOLD,
    LESSON_THRESHOLDS,
)


class QaidaPronunciationScorer:
    """Acoustic-only scoring pipeline using contrastive wav2vec2."""

    def __init__(
        self,
        pass_threshold=PASS_THRESHOLD,
        good_threshold=GOOD_THRESHOLD,
    ):
        self.pass_threshold = pass_threshold
        self.good_threshold = good_threshold

        print("Loading acoustic scorer...")
        self.acoustic_scorer = AcousticScorer(
            ref_store_path=REFERENCE_EMBEDDINGS_PATH,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            finetuned_dir=EMBEDDING_FINETUNED_DIR,
        )
        print("Scorer ready.")

    def _thresholds_for_lesson(self, lesson_num):
        return LESSON_THRESHOLDS.get(lesson_num, (self.pass_threshold, self.good_threshold))

    def score(self, student_audio_path: str, audio_id: str, expected_arabic: str) -> dict:
        """Score student pronunciation via acoustic embedding similarity."""
        acoustic_result = self.acoustic_scorer.compute_score(student_audio_path, audio_id)

        score = acoustic_result["acoustic_score"]
        lesson_num = acoustic_result.get("lesson_num", 0)
        pass_t, good_t = self._thresholds_for_lesson(lesson_num)

        if score >= good_t:
            feedback = "excellent"
        elif score >= pass_t:
            feedback = "acceptable"
        else:
            feedback = "needs_practice"

        return {
            "combined_score": round(score, 4),
            "feedback": feedback,
            "acoustic": acoustic_result,
            "audio_id": audio_id,
            "expected_arabic": expected_arabic,
            "detected_arabic": acoustic_result.get("detected_arabic", ""),
            "detected_audio_id": acoustic_result.get("detected_audio_id", ""),
            "detected_similarity": acoustic_result.get("detected_similarity", 0.0),
            "top3": acoustic_result.get("top3", []),
            "lesson_num": lesson_num,
            "pass_threshold": pass_t,
            "good_threshold": good_t,
        }
