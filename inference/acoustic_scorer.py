"""Acoustic similarity scoring using contrastive XLSR-53 embeddings (v3.1).

v3.1 fixes vs v3:
- Per-lesson DTW config: tail_fraction, skip_fraction, dtw/mean weights, min_margin
- Tail-focused DTW for vowel-final lessons (L3/L4/L5/L9/L10):
    use only last 50% of frames where the vowel/tanween/madd distinction lives
- Skip-prefix DTW for sukoon/shaddah (L6/L7):
    skip initial identical alif/consonant frames, focus on middle+tail
- Minimum margin confidence per lesson:
    if top1-top2 hybrid gap < min_margin, flag detection as uncertain
"""

import pickle
import numpy as np
from collections import defaultdict

from qaida_project.embeddings.embedding_model import ReferenceEmbeddingExtractor
from qaida_project.config.settings import (
    USE_DTW_SCORING, USE_MULTI_PROTOTYPE,
    DTW_WEIGHT, MEAN_WEIGHT, DTW_TOP_K,
    LESSON_DTW_CONFIG,
)

# Default per-lesson config fallback
_DEFAULT_DTW_CFG = (DTW_WEIGHT, MEAN_WEIGHT, 1.0, 0.0, 0.01)


def _lesson_dtw_cfg(lesson_num):
    return LESSON_DTW_CONFIG.get(lesson_num, _DEFAULT_DTW_CFG)


def cosine_dtw_similarity(
    a: np.ndarray,
    b: np.ndarray,
    tail_fraction: float = 1.0,
    skip_fraction: float = 0.0,
) -> float:
    """Frame-level cosine DTW between two L2-normalized frame sequences.

    Args:
        a: (Ta, H) per-frame normalized embeddings (student)
        b: (Tb, H) per-frame normalized embeddings (reference)
        tail_fraction: use only the last `tail_fraction` of frames.
            1.0 = full sequence. 0.5 = last 50% (focuses on vowel tail).
        skip_fraction: skip the first `skip_fraction` of frames.
            0.0 = keep all. 0.15 = skip first 15% (identical consonant prefix).
            Applied BEFORE tail_fraction slicing.

    Returns:
        similarity in [0, 1]: 1 - normalised DTW path cost.
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    # Apply skip (drop leading identical-consonant frames)
    if skip_fraction > 0.0:
        a = a[int(len(a) * skip_fraction):]
        b = b[int(len(b) * skip_fraction):]

    # Apply tail (focus on vowel-distinguishing region)
    if tail_fraction < 1.0:
        a = a[int(len(a) * (1.0 - tail_fraction)):]
        b = b[int(len(b) * (1.0 - tail_fraction)):]

    Ta, Tb = a.shape[0], b.shape[0]
    if Ta == 0 or Tb == 0:
        return 0.0

    # Cost matrix: cosine distance (vectors are L2-normalized so dot = cos_sim)
    cost = 1.0 - np.clip(a @ b.T, -1.0, 1.0)

    # DTW with Sakoe-Chiba band
    band = max(6, abs(Ta - Tb) + 3)
    INF = 1e9
    D = np.full((Ta + 1, Tb + 1), INF, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        j_lo = max(1, i - band)
        j_hi = min(Tb, i + band)
        for j in range(j_lo, j_hi + 1):
            c = cost[i - 1, j - 1]
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack path length for normalization
    i, j = Ta, Tb
    path_len = 0
    while i > 0 and j > 0:
        path_len += 1
        m = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if m == 0:
            i -= 1; j -= 1
        elif m == 1:
            i -= 1
        else:
            j -= 1
    path_len = max(path_len, 1)

    avg_cost = D[Ta, Tb] / path_len
    return float(max(0.0, 1.0 - avg_cost))


class AcousticScorer:
    """Score pronunciation by comparing audio embeddings with reference."""

    def __init__(self, ref_store_path, embedding_model_name="facebook/wav2vec2-base", finetuned_dir=None):
        self.extractor = ReferenceEmbeddingExtractor(
            model_name=embedding_model_name,
            finetuned_dir=finetuned_dir,
        )
        self.use_finetuned = self.extractor.use_finetuned

        with open(ref_store_path, "rb") as f:
            self.ref_store = pickle.load(f)

        self._all_ids = list(self.ref_store.keys())
        self._all_embs = np.stack([self.ref_store[a]["embedding"] for a in self._all_ids])
        self._all_arabic = [self.ref_store[a]["arabic_text"] for a in self._all_ids]

        # Multi-prototype matrix (audio_id -> (n_protos, D))
        self._proto_embs = {}
        if USE_MULTI_PROTOTYPE:
            for aid in self._all_ids:
                protos = self.ref_store[aid].get("prototypes")
                if protos is not None:
                    self._proto_embs[aid] = np.asarray(protos, dtype=np.float32)
        self._has_protos = bool(self._proto_embs)

        # Frame embeddings (audio_id -> (T, H) fp32)
        self._frame_embs = {}
        if USE_DTW_SCORING:
            for aid in self._all_ids:
                fe = self.ref_store[aid].get("frame_embedding")
                if fe is not None:
                    self._frame_embs[aid] = np.asarray(fe, dtype=np.float32)
        self._has_frames = bool(self._frame_embs)

        self._lesson_items = defaultdict(list)
        for i, aid in enumerate(self._all_ids):
            lnum = self.ref_store[aid].get("lesson_num", 0)
            self._lesson_items[lnum].append(i)

        self._id_to_idx = {aid: i for i, aid in enumerate(self._all_ids)}

        print(
            f"AcousticScorer v3.1: {len(self._all_ids)} refs | "
            f"multi-proto={self._has_protos} | frames={self._has_frames}"
        )

    def _mean_sim(self, student_emb: np.ndarray, audio_id: str) -> float:
        if self._has_protos and audio_id in self._proto_embs:
            return float(np.max(self._proto_embs[audio_id] @ student_emb))
        return float(np.dot(student_emb, self.ref_store[audio_id]["embedding"]))

    def _dtw_sim(
        self,
        student_frames: np.ndarray,
        audio_id: str,
        tail_fraction: float,
        skip_fraction: float,
    ) -> float:
        ref_frames = self._frame_embs.get(audio_id)
        if ref_frames is None or student_frames is None or student_frames.shape[0] == 0:
            return 0.0
        return cosine_dtw_similarity(
            student_frames, ref_frames,
            tail_fraction=tail_fraction,
            skip_fraction=skip_fraction,
        )

    def _hybrid_sim(
        self,
        student_emb,
        student_frames,
        audio_id,
        dtw_w,
        mean_w,
        tail_fraction,
        skip_fraction,
    ) -> tuple:
        mean_s = self._mean_sim(student_emb, audio_id)
        if USE_DTW_SCORING and self._has_frames and student_frames is not None:
            dtw_s = self._dtw_sim(student_frames, audio_id, tail_fraction, skip_fraction)
            blend = mean_w * mean_s + dtw_w * dtw_s
            return blend, mean_s, dtw_s
        return mean_s, mean_s, 0.0

    def compute_score(self, student_audio_path: str, audio_id: str) -> dict:
        if audio_id not in self.ref_store:
            return {
                "acoustic_score": 0.0,
                "cosine_similarity": 0.0,
                "detected_audio_id": None,
                "detected_arabic": None,
                "error": f"audio_id '{audio_id}' not found in reference store",
            }

        # Single forward pass for both mean-pool embedding and frame embeddings
        if USE_DTW_SCORING and self._has_frames:
            try:
                student_emb, student_frames = self.extractor.extract_both(student_audio_path)
                student_frames = student_frames.astype(np.float32)
            except Exception as e:
                print(f"[scorer] extract_both failed: {e}; falling back to extract()")
                student_emb = self.extractor.extract(student_audio_path)
                student_frames = None
        else:
            student_emb = self.extractor.extract(student_audio_path)
            student_frames = None

        target_lesson = self.ref_store[audio_id].get("lesson_num", 0)
        lesson_indices = self._lesson_items.get(target_lesson, [])

        # Per-lesson DTW config
        dtw_w, mean_w, tail_frac, skip_frac, min_margin = _lesson_dtw_cfg(target_lesson)

        # --- Stage 1: mean-pool sim for initial ranking ---
        if self._has_protos:
            mean_sims = {}
            for i in lesson_indices:
                aid = self._all_ids[i]
                protos = self._proto_embs.get(aid)
                mean_sims[i] = float(np.max(protos @ student_emb)) if protos is not None \
                    else float(np.dot(student_emb, self._all_embs[i]))
        else:
            mean_sims = {i: float(np.dot(student_emb, self._all_embs[i]))
                         for i in lesson_indices}

        if not mean_sims:
            all_sims = self._all_embs @ student_emb
            best_idx = int(np.argmax(all_sims))
            detected_id = self._all_ids[best_idx]
            return {
                "acoustic_score": float(max(0.0, min(1.0, float(all_sims[best_idx]) + 0.2))),
                "cosine_similarity": float(all_sims[self._id_to_idx[audio_id]]),
                "detected_audio_id": detected_id,
                "detected_arabic": self._all_arabic[best_idx],
                "detected_similarity": float(all_sims[best_idx]),
                "top3": [],
            }

        # --- Stage 2: DTW rerank top-K with per-lesson tail/skip config ---
        ranked = sorted(mean_sims.items(), key=lambda kv: kv[1], reverse=True)
        top_k = ranked[: max(DTW_TOP_K, 3)]

        hybrid_per_idx = {}
        dtw_per_idx = {}
        for i, mean_s in top_k:
            aid = self._all_ids[i]
            blend, _, dtw_s = self._hybrid_sim(
                student_emb, student_frames, aid,
                dtw_w, mean_w, tail_frac, skip_frac,
            )
            hybrid_per_idx[i] = blend
            dtw_per_idx[i] = dtw_s

        for i, mean_s in ranked[len(top_k):]:
            hybrid_per_idx[i] = mean_s
            dtw_per_idx[i] = 0.0

        # --- Detection: best hybrid in lesson ---
        ranked_hybrid = sorted(hybrid_per_idx.items(), key=lambda kv: kv[1], reverse=True)
        best_idx, best_sim = ranked_hybrid[0]
        detected_id = self._all_ids[best_idx]
        detected_arabic = self._all_arabic[best_idx]

        # Margin between top1 and top2
        second_sim = ranked_hybrid[1][1] if len(ranked_hybrid) > 1 else best_sim
        detection_margin = best_sim - second_sim
        detection_uncertain = detection_margin < min_margin

        top3 = []
        for i, sim in ranked_hybrid[:3]:
            top3.append({
                "audio_id": self._all_ids[i],
                "arabic_text": self._all_arabic[i],
                "similarity": round(float(sim), 4),
                "mean_sim": round(float(mean_sims.get(i, 0.0)), 4),
                "dtw_sim": round(float(dtw_per_idx.get(i, 0.0)), 4),
            })

        # --- Score for the *target* item ---
        target_idx = self._id_to_idx[audio_id]
        target_hybrid = hybrid_per_idx.get(target_idx, mean_sims.get(target_idx, 0.0))
        target_mean = mean_sims.get(target_idx, 0.0)
        target_dtw = dtw_per_idx.get(target_idx, 0.0)

        other_hybrids = [s for i, s in hybrid_per_idx.items() if i != target_idx]
        best_other = max(other_hybrids) if other_hybrids else 0.0
        margin = target_hybrid - best_other
        rank = 1 + sum(1 for s in other_hybrids if s > target_hybrid)
        total_in_lesson = len(lesson_indices)

        abs_score = max(0.0, min(1.0, target_hybrid + 0.2))
        if rank == 1:
            rel_score = min(1.0, 0.8 + margin * 2)
        elif rank <= 3:
            rel_score = max(0.3, 0.7 - (rank - 1) * 0.15)
        else:
            rel_score = max(0.0, 0.4 - rank * 0.05)

        if total_in_lesson > 1:
            acoustic_score = 0.4 * abs_score + 0.6 * rel_score
        else:
            acoustic_score = abs_score

        return {
            "acoustic_score": round(float(acoustic_score), 4),
            "cosine_similarity": round(float(target_mean), 4),
            "dtw_similarity": round(float(target_dtw), 4),
            "hybrid_similarity": round(float(target_hybrid), 4),
            "margin": round(float(margin), 4),
            "rank_in_lesson": rank,
            "lesson_size": total_in_lesson,
            "lesson_num": target_lesson,
            "detected_audio_id": detected_id,
            "detected_arabic": detected_arabic,
            "detected_similarity": round(float(best_sim), 4),
            "detection_margin": round(float(detection_margin), 4),
            "detection_uncertain": detection_uncertain,
            "top3": top3,
        }
