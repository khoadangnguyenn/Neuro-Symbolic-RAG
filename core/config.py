"""Runtime configuration for the EXACT pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All paths and knobs used by the pipeline.

    LLM variables intentionally point to a self-hosted OpenAI-compatible
    endpoint, usually vLLM. They are optional; the deterministic baseline still
    runs without an LLM.
    """

    project_root: Path
    logic_path: Path
    physics_path: Path
    llm_base_url: str
    llm_model: str
    expansion_llm_base_url: str
    expansion_llm_model: str
    llm_timeout_s: float
    retrieval_k: int
    high_match_threshold: float
    low_match_threshold: float
    max_retries: int
    code_timeout_s: float
    alpha: float
    enable_gpu_protection: bool
    w_acc: float
    w_lat: float
    w_cost: float
    graph_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.getenv("EXACT_PROJECT_ROOT", PROJECT_ROOT)).resolve()
        return cls(
            project_root=root,
            logic_path=Path(
                os.getenv(
                    "EXACT_LOGIC_DATA",
                    root / "dataset" / "Logic_Based_Educational_Queries.json",
                )
            ).resolve(),
            physics_path=Path(
                os.getenv(
                    "EXACT_PHYSICS_DATA",
                    root / "dataset" / "Physics_Problems_Text_Only.csv",
                )
            ).resolve(),
            llm_base_url=os.getenv("EXACT_LLM_BASE_URL", "").rstrip("/"),
            llm_model=os.getenv("EXACT_LLM_MODEL", ""),
            expansion_llm_base_url=os.getenv("EXACT_EXPANSION_LLM_BASE_URL", os.getenv("EXACT_LLM_BASE_URL", "")).rstrip("/"),
            expansion_llm_model=os.getenv("EXACT_EXPANSION_LLM_MODEL", os.getenv("EXACT_LLM_MODEL", "")),
            llm_timeout_s=_env_float("EXACT_LLM_TIMEOUT", 300.0),
            retrieval_k=_env_int("EXACT_RETRIEVAL_K", 5),
            high_match_threshold=_env_float("EXACT_HIGH_MATCH_THRESHOLD", 18.0),
            low_match_threshold=_env_float("EXACT_LOW_MATCH_THRESHOLD", 5.0),
            max_retries=_env_int("EXACT_MAX_RETRIES", 2),
            code_timeout_s=_env_float("EXACT_CODE_TIMEOUT", 4.0),
            alpha=_env_float("EXACT_ALPHA", 0.85),
            enable_gpu_protection=os.getenv("EXACT_GPU_PROTECTION", "true").lower() == "true",
            w_acc=_env_float("EXACT_W_ACC", 0.6),
            w_lat=_env_float("EXACT_W_LAT", 0.25),
            w_cost=_env_float("EXACT_W_COST", 0.15),
            graph_dir=Path(os.getenv("EXACT_GRAPH_DIR", root / "dataset" / "graph_data")).resolve()
        )
