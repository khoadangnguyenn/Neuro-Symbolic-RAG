"""SymRAG Adaptive Router."""

from __future__ import annotations

import logging
import psutil
from typing import Dict

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU = True
except Exception:
    HAS_GPU = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logging.warning("spaCy model 'en_core_web_sm' not loaded. NER heuristic will be degraded.")

from exact_pipeline.core.config import Settings


class AdaptiveRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.r_smooth = 0.0
        self.alpha_smooth = 0.3
        self.max_tokens_approx = 512.0
        
        # Weights for kappa
        self.w_A = 1.0
        self.w_L = 1.0
        self.w_sh1 = 0.05
        self.w_sh2 = 0.1

    def compute_kappa(self, query: str) -> float:
        """Compute Query Complexity \kappa(q)."""
        tokens = query.split()
        q_len = len(tokens)
        if q_len == 0:
            return 0.0

        # Heuristic for Attention A(q): fraction of non-stop words (or 1.0)
        a_q = 1.0

        # Normalized length L(q)
        l_q = min(q_len / self.max_tokens_approx, 1.0)

        # Structure Index S_H(q)
        n_ents = 0
        n_hops = sum(1 for word in tokens if word.lower() in {"and", "then", "because", "if", "so", "therefore"})
        
        if nlp is not None:
            doc = nlp(query)
            n_ents = len(doc.ents)
            
        s_h = self.w_sh1 * (n_ents / q_len) + self.w_sh2 * (n_hops / q_len)

        kappa = (self.w_A * a_q + self.w_L * l_q) * (1.0 + s_h)
        return kappa

    def compute_resource_pressure(self) -> float:
        """Compute R_p(t) = max{CPU, GPU, MEM} and apply EMA."""
        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        gpu = 0.0
        
        if HAS_GPU:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu = mem_info.used / mem_info.total
            except Exception:
                pass
                
        r_t = max(cpu, mem, gpu)
        self.r_smooth = self.alpha_smooth * r_t + (1.0 - self.alpha_smooth) * self.r_smooth
        return self.r_smooth

    def compute_utility(self, kappa: float, r_p: float, path: str) -> float:
        """Compute U(P | \kappa_{eff}, R)."""
        # path = "fast" or "hybrid"
        # We estimate Acc, Lat, Cost
        if path == "fast":
            acc = max(0.1, 1.0 - 0.8 * kappa)  # Fast path fails on complex queries
            lat = 0.05
            cost = 0.05
        else:
            acc = 0.95  # Hybrid path is very accurate
            lat = 0.8
            cost = 0.2 + 0.8 * r_p  # Cost scales with resource pressure
            
        # U = w_acc * Acc - w_lat * Lat - w_cost * Cost
        u = self.settings.w_acc * acc - self.settings.w_lat * lat - self.settings.w_cost * cost
        return u

    def route(self, query: str) -> Dict[str, Any]:
        """Decide whether to use Fast Path or Hybrid Path."""
        kappa = self.compute_kappa(query)
        r_p = self.compute_resource_pressure()
        
        u_fast = self.compute_utility(kappa, r_p, "fast")
        u_hybrid = self.compute_utility(kappa, r_p, "hybrid")
        
        decision = "fast"
        if u_hybrid > u_fast:
            decision = "hybrid"
            
        if self.settings.enable_gpu_protection and r_p > 0.9:
            decision = "fast"
            
        return {
            "path": decision,
            "kappa": kappa,
            "r_p": r_p,
            "u_fast": u_fast,
            "u_hybrid": u_hybrid
        }
