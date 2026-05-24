from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
#   python -m uvicorn src.service.api:app --reload
# ---------------------------------------------------------------------------
# ПУТИ
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()

# project/
PROJECT_ROOT = THIS_FILE.parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"

CONFIG_PATH = Path(str(CONFIG_PATH))
ARTIFACTS_PATH = Path(str(ARTIFACTS_PATH))

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_MODEL_CONFIGS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, -1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, -1),
}


class EmbeddingModel(nn.Module):
    def __init__(self, name="resnet50"):
        super().__init__()

        if name not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {name}")

        factory, weights, cut = _MODEL_CONFIGS[name]
        backbone = factory(weights=weights)

        self.model = nn.Sequential(*list(backbone.children())[:cut])

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return nn.functional.normalize(x, p=2, dim=1)


def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class AppState:
    config: dict
    model: EmbeddingModel
    transform: T.Compose
    device: torch.device
    index: faiss.Index
    metadata: pd.DataFrame
    loaded_at: float


_state: Optional[AppState] = None


def load_state():
    st = AppState()
    st.loaded_at = time.time()

    st.config = load_config()

    # DEVICE
    device_str = st.config.get("model", {}).get("device", "cpu")
    st.device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    # MODEL
    model_name = st.config["model"]["name"]

    st.model = EmbeddingModel(model_name).to(st.device)
    st.model.eval()

    st.transform = build_transform()

    # INDEX
    index_path = ARTIFACTS_PATH / "faiss_index.index"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    # ⚠️ ключевой фикс
    st.index = faiss.read_index(str(index_path.resolve()))

    # METADATA
    meta_path = ARTIFACTS_PATH / "metadata.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    st.metadata = pd.read_csv(meta_path)

    return st


# ---------------------------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Visual Search API")


@app.on_event("startup")
def startup():
    global _state
    _state = load_state()


def get_state():
    if _state is None:
        raise HTTPException(503, "Service not ready")
    return _state


# ---------------------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------------------
def embed_image(img: Image.Image, st: AppState):
    tensor = st.transform(img).unsqueeze(0).to(st.device)

    with torch.no_grad():
        emb = st.model(tensor).cpu().numpy()

    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    st = get_state()
    return {
        "status": "ok",
        "index_size": st.index.ntotal,
        "device": str(st.device),
        "uptime": round(time.time() - st.loaded_at, 1),
    }


@app.get("/config")
def config():
    st = get_state()
    return st.config


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50),
):
    st = get_state()
    t0 = time.perf_counter()

    # IMAGE
    raw = await file.read()

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(422, "Invalid image")

    # EMBEDDING
    emb = embed_image(img, st)

    # SEARCH
    k = min(top_k, st.index.ntotal)
    D, I = st.index.search(emb, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        row = st.metadata.iloc[idx]

        sim = 1 / (1 + float(dist))

        results.append({
            "filename": row["filename"],
            "distance": float(dist),
            "similarity": sim,
        })

    latency = (time.perf_counter() - t0) * 1000

    return {
        "results": results,
        "latency_ms": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )