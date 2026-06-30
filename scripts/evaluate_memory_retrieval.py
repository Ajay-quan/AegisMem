#!/usr/bin/env python3
"""Run a deterministic local stateful.ai retrieval benchmark.

The benchmark uses no paid services and no network calls. It seeds synthetic
memories, runs retrieval queries with expected relevant keys, measures ranking
quality and latency, and writes JSON/CSV/SVG artifacts for the README and docs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.flask_memory_service import FlaskMemoryService

CORE_DATASET = [
    ("python-pref", "Alice prefers Python, Flask, and FAISS for local vector search demos."),
    ("aws-free-tier", "Alice deploys demos on AWS Free Tier EC2 with EBS persistence and no managed databases."),
    ("hash-index", "Exact lookup in stateful.ai uses a SHA-256 hash map for constant-time key retrieval."),
    ("graph-bfs", "Graph retrieval uses weighted BFS traversal across related memory nodes."),
    ("docker-gunicorn", "The Flask API is containerized with Docker and served by Gunicorn on port 8000."),
    ("priority-cache", "Hot memories are tracked with a priority queue, hash map, and recency tree."),
    ("semantic-pipeline", "The semantic retrieval pipeline embeds text, stores vectors, searches FAISS, and hydrates records."),
    ("teardown", "The AWS runbook includes teardown commands for EC2, EBS, security groups, and key pairs."),
    ("openapi", "The REST API is documented with an OpenAPI YAML specification and curl examples."),
    ("single-node", "The project is microservices-style internally but deployed as a single-node demo to avoid cost."),
]

NOISE_DATASET = [
    (f"noise-{idx}", f"Unrelated memory {idx} about calendars, recipes, campus events, music practice, or grocery planning.")
    for idx in range(1, 61)
]

DATASET = CORE_DATASET + NOISE_DATASET

QUERIES = [
    ("python faiss local search", {"python-pref", "semantic-pipeline"}),
    ("aws free tier ec2 ebs", {"aws-free-tier", "single-node", "teardown"}),
    ("constant time hash lookup", {"hash-index"}),
    ("weighted graph bfs traversal", {"graph-bfs"}),
    ("docker gunicorn flask port", {"docker-gunicorn"}),
    ("priority queue recency tree cache", {"priority-cache"}),
    ("embedding vector faiss pipeline", {"semantic-pipeline", "python-pref"}),
    ("delete aws resources teardown", {"teardown", "aws-free-tier"}),
    ("openapi rest documentation", {"openapi"}),
    ("single node no cost deployment", {"single-node", "aws-free-tier"}),
]


def precision_at_k(keys: list[str], expected: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for key in keys[:k] if key in expected) / k


def recall_at_k(keys: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    return sum(1 for key in keys[:k] if key in expected) / len(expected)


def reciprocal_rank(keys: list[str], expected: set[str]) -> float:
    for idx, key in enumerate(keys, start=1):
        if key in expected:
            return 1.0 / idx
    return 0.0


def write_svg_bar_chart(path: Path, metrics: dict[str, float]) -> None:
    width, height = 720, 320
    bars = [("P@1", metrics["precision_at_1"]), ("P@3", metrics["precision_at_3"]), ("R@5", metrics["recall_at_5"]), ("MRR", metrics["mrr"])]
    chart_left, chart_bottom = 80, 250
    bar_width, gap = 95, 45
    max_h = 180
    rects = []
    labels = []
    for i, (label, value) in enumerate(bars):
        x = chart_left + i * (bar_width + gap)
        h = value * max_h
        y = chart_bottom - h
        rects.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="#3563e9"/>')
        labels.append(f'<text x="{x + bar_width / 2}" y="{chart_bottom + 24}" text-anchor="middle" font-size="14">{label}</text>')
        labels.append(f'<text x="{x + bar_width / 2}" y="{y - 8:.1f}" text-anchor="middle" font-size="13">{value:.2f}</text>')
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="30" y="38" font-size="22" font-family="Arial" font-weight="700">stateful.ai Retrieval Quality</text>
  <line x1="{chart_left}" y1="{chart_bottom}" x2="650" y2="{chart_bottom}" stroke="#333"/>
  <line x1="{chart_left}" y1="70" x2="{chart_left}" y2="{chart_bottom}" stroke="#333"/>
  {''.join(rects)}
  {''.join(labels)}
</svg>'''
    )


def write_latency_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 720, 320
    max_latency = max(row["latency_ms"] for row in rows) or 1.0
    points = []
    for i, row in enumerate(rows):
        x = 70 + i * (580 / max(1, len(rows) - 1))
        y = 250 - (row["latency_ms"] / max_latency) * 170
        points.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#e95f35"/>' for x, y in points)
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="30" y="38" font-size="22" font-family="Arial" font-weight="700">stateful.ai Retrieval Latency</text>
  <line x1="70" y1="250" x2="660" y2="250" stroke="#333"/>
  <line x1="70" y1="70" x2="70" y2="250" stroke="#333"/>
  <polyline points="{poly}" fill="none" stroke="#e95f35" stroke-width="3"/>
  {circles}
  <text x="70" y="280" font-size="13">queries</text>
  <text x="18" y="75" font-size="13">{max_latency:.2f} ms</text>
</svg>'''
    )


def run(output_dir: Path, asset_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as data_dir:
        service = FlaskMemoryService(data_dir=data_dir, embedding_backend="mock")
        memory_ids: dict[str, str] = {}
        previous_id = None
        for key, content in DATASET:
            related = [previous_id] if previous_id else []
            memory = service.ingest(content=content, user_id="bench", key=key, related_memory_ids=related, importance_score=0.8)
            memory_ids[key] = memory.memory_id
            previous_id = memory.memory_id

        rows = []
        p1 = []
        p3 = []
        r5 = []
        rr = []
        for query, expected in QUERIES:
            start = time.perf_counter()
            results = service.retrieve(query=query, user_id="bench", top_k=5)
            latency_ms = (time.perf_counter() - start) * 1000
            keys = [result["key"] for result in results]
            row = {
                "query": query,
                "expected_keys": sorted(expected),
                "returned_keys": keys,
                "latency_ms": round(latency_ms, 4),
                "precision_at_1": precision_at_k(keys, expected, 1),
                "precision_at_3": precision_at_k(keys, expected, 3),
                "recall_at_5": recall_at_k(keys, expected, 5),
                "reciprocal_rank": reciprocal_rank(keys, expected),
            }
            rows.append(row)
            p1.append(row["precision_at_1"])
            p3.append(row["precision_at_3"])
            r5.append(row["recall_at_5"])
            rr.append(row["reciprocal_rank"])

    metrics = {
        "query_count": len(QUERIES),
        "memory_count": len(DATASET),
        "precision_at_1": round(statistics.mean(p1), 4),
        "precision_at_3": round(statistics.mean(p3), 4),
        "recall_at_5": round(statistics.mean(r5), 4),
        "mrr": round(statistics.mean(rr), 4),
        "avg_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 4),
        "p95_latency_ms": round(sorted(row["latency_ms"] for row in rows)[int(len(rows) * 0.95) - 1], 4),
    }
    report = {"metrics": metrics, "queries": rows}
    (output_dir / "retrieval_metrics.json").write_text(json.dumps(report, indent=2))
    with (output_dir / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "latency_ms", "precision_at_1", "precision_at_3", "recall_at_5", "reciprocal_rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    write_svg_bar_chart(asset_dir / "retrieval_quality.svg", metrics)
    write_latency_svg(asset_dir / "retrieval_latency.svg", rows)
    (output_dir / "README.md").write_text(
        "# Benchmark Results\n\n"
        f"- Memories: {metrics['memory_count']}\n"
        f"- Queries: {metrics['query_count']}\n"
        f"- Precision@1: {metrics['precision_at_1']}\n"
        f"- Precision@3: {metrics['precision_at_3']}\n"
        f"- Recall@5: {metrics['recall_at_5']}\n"
        f"- MRR: {metrics['mrr']}\n"
        f"- Average latency: {metrics['avg_latency_ms']} ms\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--asset-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    report = run(args.output_dir, args.asset_dir)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
