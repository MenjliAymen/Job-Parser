"""
Generalized Job Category Clustering + Per‑Job Soft Scoring

Usage (example):
    python generalized_category_pipeline.py \
        --jobs-csv cleaned_job_db.csv \
        --categories-col cleaned_key_categories \
        --text-as-list True \
        --model all-MiniLM-L6-v2 \
        --thresholds 0.45 0.50 0.54 0.58 0.62 \
        --output-dir artifacts

Produces:
    artifacts/category_vocab.csv                 # all unique raw category strings
    artifacts/category_embeddings.parquet        # embeddings for each category string
    artifacts/cluster_map.csv                    # title,cluster_id,normalized_title (canonical)
    artifacts/cluster_centroids.npy              # (n_clusters, dim) float32
    artifacts/experiment_report.csv              # metrics per threshold
    artifacts/job_scores.parquet                 # per job: primary + soft scores
    artifacts/normalizing_mapper.csv             # compatibility output for existing pipeline

Notes:
- Soft scoring preserves dominance: a job stays in the cluster with highest score; secondary weights are kept.
- Unseen category handling: falls back to nearest cluster by cosine.
- Canonical label selection: most frequent title per cluster with medoid tiebreaker.

"""
from __future__ import annotations
import argparse
import ast
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

# -------------------------
# Utils
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_jobs_csv(path: str, categories_col: str, text_as_list: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_as_list:
        # values are stringified Python lists
        df[categories_col] = df[categories_col].apply(lambda s: list(ast.literal_eval(s)))
    # coerce to list
    df[categories_col] = df[categories_col].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [str(v)]))
    return df


def build_category_vocab(df: pd.DataFrame, categories_col: str) -> List[str]:
    # flatten & unique while preserving insertion order
    seen = set()
    vocab = []
    for lst in df[categories_col].tolist():
        for x in lst:
            x = (x or "").strip()
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                vocab.append(x)
    return vocab


def embed_texts(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


@dataclass
class ClusterArtifacts:
    labels: np.ndarray               # (n_items,)
    centroids: np.ndarray            # (n_clusters, dim)
    id_to_indices: Dict[int, np.ndarray]
    canonical_title: Dict[int, str]


# -------------------------
# Clustering
# -------------------------

def cluster_with_threshold(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    # Agglomerative with cosine distance threshold (no preset n_clusters)
    clus = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clus.fit_predict(embeddings)
    return labels


def compute_cluster_centroids(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    id_to_indices: Dict[int, np.ndarray] = {}
    uniq = np.unique(labels)
    dim = embeddings.shape[1]
    centroids = np.zeros((uniq.size, dim), dtype=np.float32)
    id_map = {cid: i for i, cid in enumerate(uniq)}
    for cid in uniq:
        idx = np.where(labels == cid)[0]
        id_to_indices[cid] = idx
        centroids[id_map[cid]] = embeddings[idx].mean(axis=0)
    # map back from compact index to original cluster id
    # We'll keep a helper to go from cluster id -> centroid row index
    return centroids, id_to_indices


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (m, d), B: (n, d) -> (m, n)
    # embeddings are normalized, so cosine = dot product
    return A @ B.T


def pick_canonical_labels(
    titles: List[str],
    labels: np.ndarray,
    id_to_indices: Dict[int, np.ndarray],
    title_counts: Counter,
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> Dict[int, str]:
    canonical = {}
    uniq = np.unique(labels)
    # helper to convert original cluster id to centroid row
    row_of = {cid: i for i, cid in enumerate(uniq)}
    for cid in uniq:
        idx = id_to_indices[cid]
        cluster_titles = [titles[i] for i in idx]
        # Most frequent title wins; tiebreaker by medoid (closest to centroid)
        counts = Counter({t: title_counts[t] for t in cluster_titles})
        max_c = max(counts.values())
        candidates = [t for t, c in counts.items() if c == max_c]
        if len(candidates) == 1:
            canonical[cid] = candidates[0]
            continue
        # medoid among ties
        centroid = centroids[row_of[cid]].reshape(1, -1)
        sims = cosine_sim(embeddings[idx], centroid).ravel()
        best_i = idx[int(np.argmax(sims))]
        canonical[cid] = titles[best_i]
    return canonical


# -------------------------
# Scoring (soft assignment per job)
# -------------------------

def score_job_categories(
    job_cats: List[str],
    vocab: List[str],
    vocab_index: Dict[str, int],
    vocab_emb: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_centroids: np.ndarray,
    k_top_centroids: int = 10,
) -> List[Tuple[int, float]]:
    """Return list of (cluster_id, score in [0,1]) sorted desc.
    We aggregate similarity of each job category to the top-k centroids and normalize.
    """
    if not job_cats:
        return []
    # Precompute centroid similarities for each category vector
    sims_sum: Dict[int, float] = defaultdict(float)
    # map original cluster id -> row index in centroids
    uniq = np.unique(cluster_ids)
    row_of = {cid: i for i, cid in enumerate(uniq)}
    for cat in job_cats:
        # If unseen, infer its vector by encoding on the fly (or fallback: nearest vocab title)
        if cat in vocab_index:
            v = vocab_emb[vocab_index[cat]].reshape(1, -1)
        else:
            # Encode directly to avoid dropping information
            # (Light overhead since unseen categories are rare per job.)
            model = SentenceTransformer("all-MiniLM-L6-v2")
            v = model.encode([cat], normalize_embeddings=True)
        sims = cosine_sim(v, cluster_centroids).ravel()
        # take top-k centroids for sparsity
        top_idx = np.argpartition(-sims, min(k_top_centroids, sims.size - 1))[:k_top_centroids]
        for j in top_idx:
            # convert centroid row back to original cluster id
            cid = uniq[j]
            sims_sum[cid] += float(sims[j])
    if not sims_sum:
        return []
    # normalize to 1.0
    total = sum(max(x, 0.0) for x in sims_sum.values())
    if total <= 1e-9:
        # fallback uniform if all sims negative (unlikely with normalized embeddings)
        s = 1.0 / len(sims_sum)
        return sorted([(cid, s) for cid in sims_sum.keys()], key=lambda x: -x[1])
    scores = [(cid, max(0.0, val) / total) for cid, val in sims_sum.items()]
    scores.sort(key=lambda x: -x[1])
    return scores


# -------------------------
# Experiment sweep + export
# -------------------------

def run_experiments(
    titles: List[str],
    title_counts: Counter,
    embeddings: np.ndarray,
    thresholds: List[float],
    out_dir: str,
) -> Tuple[pd.DataFrame, Dict[float, ClusterArtifacts]]:
    rows = []
    artifacts_by_thr: Dict[float, ClusterArtifacts] = {}
    for thr in thresholds:
        labels = cluster_with_threshold(embeddings, thr)
        uniq = np.unique(labels)
        n_clusters = uniq.size
        centroids, id_to_indices = compute_cluster_centroids(embeddings, labels)
        # intra-cluster quality: mean cosine(sim) to centroid
        row_of = {cid: i for i, cid in enumerate(uniq)}
        sims = []
        for cid in uniq:
            idx = id_to_indices[cid]
            if idx.size == 0:
                continue
            c = centroids[row_of[cid]].reshape(1, -1)
            s = cosine_sim(embeddings[idx], c).ravel()
            sims.extend(s.tolist())
        mean_intra = float(np.mean(sims)) if sims else float("nan")
        singletons = int(sum(1 for cid in uniq if id_to_indices[cid].size == 1))
        rows.append({
            "threshold": thr,
            "n_clusters": n_clusters,
            "mean_intra_cosine": round(mean_intra, 4),
            "pct_singletons": round(100.0 * singletons / max(1, n_clusters), 2),
        })
        artifacts_by_thr[thr] = ClusterArtifacts(
            labels=labels,
            centroids=centroids,
            id_to_indices=id_to_indices,
            canonical_title={},  # filled later when we pick best threshold
        )
    report = pd.DataFrame(rows).sort_values("threshold")
    report.to_csv(os.path.join(out_dir, "experiment_report.csv"), index=False)
    return report, artifacts_by_thr


def select_threshold(report: pd.DataFrame, target_max_clusters: int = 400) -> float:
    # Heuristic: pick highest mean_intra subject to n_clusters <= target
    cand = report[report["n_clusters"] <= target_max_clusters]
    if cand.empty:
        # else take row with n_clusters closest above target
        i = (report["n_clusters"] - target_max_clusters).abs().idxmin()
        return float(report.loc[i, "threshold"])
    i = cand.sort_values(["mean_intra_cosine", "threshold"], ascending=[False, True]).index[0]
    return float(report.loc[i, "threshold"])


# -------------------------
# Main pipeline
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-csv", required=True)
    ap.add_argument("--categories-col", default="cleaned_key_categories")
    ap.add_argument("--text-as-list", type=lambda s: str(s).lower() == "true", default=True)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.54])
    ap.add_argument("--target-max-clusters", type=int, default=400)
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--topk-centroids", type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    # 1) Load jobs and build vocab
    df = load_jobs_csv(args.jobs_csv, args.categories_col, args.text_as_list)
    vocab = build_category_vocab(df, args.categories_col)
    pd.DataFrame({"title": vocab}).to_csv(os.path.join(args.output_dir, "category_vocab.csv"), index=False)

    # 2) Embed all unique category titles
    emb = embed_texts(vocab, args.model)
    cols = [f"dim_{i}" for i in range(emb.shape[1])]
    pd.DataFrame(emb, columns=cols).to_parquet(
        os.path.join(args.output_dir, "category_embeddings.parquet")
    )

    # 3) Run clustering experiments
    report, artifacts_by_thr = run_experiments(vocab, Counter(x for lst in df[args.categories_col] for x in lst), emb, args.thresholds, args.output_dir)

    # 4) Pick best threshold and finalize artifacts
    best_thr = select_threshold(report, args.target_max_clusters)
    art = artifacts_by_thr[best_thr]

    # 5) Canonical labels
    title_counts = Counter(x for lst in df[args.categories_col] for x in lst)
    uniq = np.unique(art.labels)
    art.canonical_title = pick_canonical_labels(
        titles=vocab,
        labels=art.labels,
        id_to_indices=art.id_to_indices,
        title_counts=title_counts,
        embeddings=emb,
        centroids=art.centroids,
    )

    # 6) Export cluster_map (title→cluster_id→normalized_title)
    rows = []
    for i, title in enumerate(vocab):
        cid = int(art.labels[i])
        rows.append({
            "title": title,
            "cluster_id": cid,
            "normalized_title": art.canonical_title[cid],
        })
    cluster_map = pd.DataFrame(rows)
    cluster_map.to_csv(os.path.join(args.output_dir, "cluster_map.csv"), index=False)

    # 6b) Compatibility: normalizing_mapper.csv (title, cluster, normalized_title)
    compat = cluster_map.rename(columns={"cluster_id": "cluster"})[["title", "cluster", "normalized_title"]]
    compat.to_csv(os.path.join(args.output_dir, "normalizing_mapper.csv"), index=False)

    # 7) Save centroids
    np.save(os.path.join(args.output_dir, "cluster_centroids.npy"), art.centroids)

    # 8) Per-job soft scoring
    vocab_index = {t: i for i, t in enumerate(vocab)}
    job_rows = []
    for idx, row in df.iterrows():
        job_id = row.get("id", idx)  # fallback
        cats: List[str] = row[args.categories_col]
        scores = score_job_categories(
            job_cats=cats,
            vocab=vocab,
            vocab_index=vocab_index,
            vocab_emb=emb,
            cluster_ids=art.labels,
            cluster_centroids=art.centroids,
            k_top_centroids=args.topk_centroids,
        )
        if scores:
            primary_cluster, primary_score = scores[0]
            primary_label = art.canonical_title[primary_cluster]
        else:
            primary_cluster, primary_score, primary_label = (None, 0.0, None)
        job_rows.append({
            "job_id": job_id,
            "primary_cluster": primary_cluster,
            "primary_normalized_title": primary_label,
            "primary_score": round(float(primary_score), 4),
            "cluster_scores": json.dumps([(int(cid), art.canonical_title[int(cid)], round(float(sc), 4)) for cid, sc in scores]),
            "raw_categories": json.dumps(cats),
        })
    df_scores = pd.DataFrame(job_rows)
    out_scores = os.path.join(args.output_dir, "job_scores.parquet")
    df_scores.to_parquet(out_scores, index=False)

    # 9) (Optional) Construct a normalized list per job (dedup, preserve dominance)
    def build_general_list(scores_json: str, alpha: float = 0.15) -> List[str]:
        # keep any label with score >= alpha, always include primary
        items = json.loads(scores_json)
        if not items:
            return []
        primary_label = items[0][1]
        keep = [lbl for _, lbl, sc in items if sc >= alpha]
        if primary_label not in keep:
            keep.insert(0, primary_label)
        # dedupe preserving order
        seen = set()
        out = []
        for t in keep:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    df_scores["general_category_list"] = df_scores["cluster_scores"].apply(build_general_list)

    # 10) Export an enriched CSV joinable by job_id
    df_scores.to_csv(os.path.join(args.output_dir, "job_scores.csv"), index=False)

    print("\n=== Summary ===")
    print(report.to_string(index=False))
    print(f"\nSelected threshold: {best_thr}")
    print(f"Clusters: {len(np.unique(art.labels))}")
    print("Artifacts written to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
