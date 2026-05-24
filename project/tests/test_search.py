"""
tests/test_search.py
====================
Тесты для VisualSearch: модель, FAISS-индекс, API-эндпоинты.

Запуск всех тестов:
    pytest tests/ -v

Запуск только быстрых тестов (без загрузки модели):
    pytest tests/ -v -m "not slow"

Зависимости:
    pip install pytest httpx
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS    = PROJECT_ROOT / "artifacts"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "project.yaml"


# ===========================================================================
# Вспомогательные фикстуры
# ===========================================================================

def _make_dummy_image(width: int = 224, height: int = 224) -> Image.Image:
    """Создаёт случайное RGB-изображение для тестов."""
    rng = np.random.default_rng(42)
    arr = (rng.integers(0, 256, (height, width, 3))).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ===========================================================================
# 1. Конфигурация
# ===========================================================================

class TestConfig:
    def test_config_file_exists(self):
        """configs/project.yaml должен существовать."""
        assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

    def test_config_has_required_keys(self):
        """Конфиг должен содержать все обязательные секции."""
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "paths"     in cfg, "Missing 'paths' section"
        assert "model"     in cfg, "Missing 'model' section"
        assert "inference" in cfg, "Missing 'inference' section"
        assert "index"     in cfg, "Missing 'index' section"

    def test_config_model_name_supported(self):
        """Имя модели из конфига должно быть поддержанным."""
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        supported = {"resnet50", "resnet18", "efficientnet_b0", "clip_vit_b32"}
        assert cfg["model"]["name"] in supported, (
            f"Model '{cfg['model']['name']}' not in {supported}"
        )

    def test_config_top_k_positive(self):
        """top_k должен быть положительным числом."""
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["inference"]["top_k"] > 0


# ===========================================================================
# 2. Артефакты
# ===========================================================================

class TestArtifacts:
    def test_artifacts_dir_exists(self):
        """Директория artifacts/ должна существовать."""
        assert ARTIFACTS.exists(), (
            "artifacts/ not found. Run 01_train_embeddings_optimized.ipynb first."
        )

    @pytest.mark.skipif(
        not (ARTIFACTS / "embeddings.npy").exists(),
        reason="embeddings.npy not found — run notebook first",
    )
    def test_embeddings_shape(self):
        """embeddings.npy должен быть 2D-матрицей float32."""
        emb = np.load(ARTIFACTS / "embeddings.npy")
        assert emb.ndim == 2,         f"Expected 2D array, got shape {emb.shape}"
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
        assert emb.shape[0] > 0,     "Embeddings matrix is empty"
        assert emb.shape[1] > 0,     "Embedding dimension is 0"

    @pytest.mark.skipif(
        not (ARTIFACTS / "embeddings.npy").exists(),
        reason="embeddings.npy not found",
    )
    def test_embeddings_are_l2_normalized(self):
        """Эмбеддинги должны быть L2-нормализованы (норма ≈ 1.0)."""
        emb = np.load(ARTIFACTS / "embeddings.npy")
        sample = emb[:100]
        norms = np.linalg.norm(sample, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"Embeddings are not L2-normalized. Mean norm: {norms.mean():.4f}"
        )

    @pytest.mark.skipif(
        not (ARTIFACTS / "metadata.csv").exists(),
        reason="metadata.csv not found — run notebook first",
    )
    def test_metadata_columns(self):
        """metadata.csv должен содержать колонки index, filename, filepath."""
        import pandas as pd
        df = pd.read_csv(ARTIFACTS / "metadata.csv")
        for col in ("index", "filename", "filepath"):
            assert col in df.columns, f"Column '{col}' missing from metadata.csv"
        assert len(df) > 0, "metadata.csv is empty"

    @pytest.mark.skipif(
        not (ARTIFACTS / "metadata.csv").exists(),
        reason="metadata.csv not found",
    )
    def test_metadata_no_duplicate_filenames(self):
        """В метаданных не должно быть дублирующихся имён файлов."""
        import pandas as pd
        df = pd.read_csv(ARTIFACTS / "metadata.csv")
        duplicates = df["filename"].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate filenames in metadata"

    @pytest.mark.skipif(
        not (ARTIFACTS / "run_report.json").exists(),
        reason="run_report.json not found",
    )
    def test_run_report_is_valid_json(self):
        """run_report.json должен быть корректным JSON с ключами config и stats."""
        with open(ARTIFACTS / "run_report.json", encoding="utf-8") as f:
            report = json.load(f)
        assert "config" in report, "run_report.json missing 'config'"
        assert "stats"  in report, "run_report.json missing 'stats'"


# ===========================================================================
# 3. FAISS-индекс
# ===========================================================================

class TestFaissIndex:
    @pytest.fixture(scope="class")
    def faiss_index(self):
        pytest.importorskip("faiss")
        import faiss
        index_path = ARTIFACTS / "faiss_index.index"
        if not index_path.exists():
            pytest.skip("faiss_index.index not found — run notebook first")
        return faiss.read_index(str(index_path))

    def test_index_not_empty(self, faiss_index):
        """Индекс должен содержать векторы."""
        assert faiss_index.ntotal > 0, "FAISS index is empty"

    def test_index_dimension_matches_embeddings(self, faiss_index):
        """Размерность индекса должна совпадать с embeddings.npy."""
        emb_path = ARTIFACTS / "embeddings.npy"
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")
        emb = np.load(emb_path)
        assert faiss_index.d == emb.shape[1], (
            f"Index dim {faiss_index.d} != embeddings dim {emb.shape[1]}"
        )

    def test_search_returns_correct_shape(self, faiss_index):
        """search() должен возвращать массивы правильной формы."""
        dim   = faiss_index.d
        query = np.random.randn(1, dim).astype(np.float32)
        query /= np.linalg.norm(query)
        k = 5
        distances, indices = faiss_index.search(query, k)
        assert distances.shape == (1, k)
        assert indices.shape   == (1, k)

    def test_sanity_self_retrieval(self, faiss_index):
        """
        Sanity-check: изображение, запрошенное само по себе, должно быть
        на первом месте (rank-1 accuracy = 100% при Flat-индексе).
        """
        emb_path = ARTIFACTS / "embeddings.npy"
        if not emb_path.exists():
            pytest.skip("embeddings.npy not found")

        emb = np.load(emb_path)
        rng = np.random.default_rng(0)
        sample_indices = rng.choice(len(emb), size=min(50, len(emb)), replace=False)

        failures = 0
        for idx in sample_indices:
            query = emb[idx : idx + 1].astype(np.float32)
            _, result_indices = faiss_index.search(query, 1)
            if result_indices[0][0] != idx:
                failures += 1

        assert failures == 0, (
            f"Self-retrieval failed for {failures}/{len(sample_indices)} queries. "
            "Check index type (Flat должен давать точный поиск)."
        )

    def test_search_latency(self, faiss_index):
        """Среднее время одного запроса к индексу должно быть < 500 мс."""
        dim    = faiss_index.d
        n_runs = 20
        query  = np.random.randn(1, dim).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            faiss_index.search(query, 5)
        avg_ms = (time.perf_counter() - t0) * 1000 / n_runs

        assert avg_ms < 500, f"Search too slow: {avg_ms:.1f} ms/query (limit 500 ms)"


# ===========================================================================
# 4. Модель
# ===========================================================================

class TestEmbeddingModel:
    @pytest.fixture(scope="class")
    def model_and_transform(self):
        torch = pytest.importorskip("torch")
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.service.api import EmbeddingModel, _build_transform
        model     = EmbeddingModel("resnet50").eval()
        transform = _build_transform()
        return model, transform

    def test_model_output_shape(self, model_and_transform):
        """Модель должна возвращать тензор [batch, 2048]."""
        import torch
        model, transform = model_and_transform
        img    = _make_dummy_image()
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        assert out.shape == (1, 2048), f"Unexpected output shape: {out.shape}"

    def test_model_output_l2_normalized(self, model_and_transform):
        """Выходные эмбеддинги должны быть L2-нормализованы."""
        import torch
        model, transform = model_and_transform
        img    = _make_dummy_image()
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        norm = out.norm(dim=1).item()
        assert abs(norm - 1.0) < 1e-5, f"Output not L2-normalized, norm={norm:.6f}"

    def test_model_deterministic(self, model_and_transform):
        """Одно и то же изображение должно давать одинаковый эмбеддинг."""
        import torch
        model, transform = model_and_transform
        img    = _make_dummy_image()
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out1 = model(tensor).numpy()
            out2 = model(tensor).numpy()
        assert np.allclose(out1, out2), "Model output is not deterministic"

    def test_different_images_different_embeddings(self, model_and_transform):
        """Разные изображения должны давать разные эмбеддинги."""
        import torch
        model, transform = model_and_transform
        rng  = np.random.default_rng(1)
        img1 = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        img2 = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        with torch.no_grad():
            e1 = model(transform(img1).unsqueeze(0)).numpy()
            e2 = model(transform(img2).unsqueeze(0)).numpy()
        assert not np.allclose(e1, e2), "Different images produced identical embeddings"

    @pytest.mark.slow
    def test_unsupported_model_raises(self):
        """Неизвестное имя модели должно вызывать ValueError."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.service.api import EmbeddingModel
        with pytest.raises(ValueError, match="Unknown model"):
            EmbeddingModel("resnet999")


# ===========================================================================
# 5. API (через TestClient, без реального индекса — мокаем _state)
# ===========================================================================

@pytest.fixture(scope="module")
def mock_state():
    """Подготавливает мок AppState с фиктивным индексом и метаданными."""
    import faiss
    import pandas as pd
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.service.api import AppState

    DIM = 2048
    N   = 10

    # Создаём маленький Flat-индекс с синтетическими данными
    vecs = np.random.randn(N, DIM).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx  = faiss.IndexFlatL2(DIM)
    idx.add(vecs)

    meta = pd.DataFrame({
        "index"   : range(N),
        "filename": [f"item_{i:04d}.jpg" for i in range(N)],
        "filepath": [f"/fake/path/item_{i:04d}.jpg" for i in range(N)],
    })

    st = AppState()
    st.loaded_at = time.time()
    st.config    = {
        "model"    : {"name": "resnet50", "device": "cpu"},
        "inference": {"top_k": 5, "batch_size": 64},
        "index"    : {"type": "Flat", "nlist": 500, "nprobe": 5},
    }
    st.index    = idx
    st.metadata = meta
    st.device   = "cpu"
    return st, vecs


@pytest.fixture(scope="module")
def client(mock_state):
    """TestClient с замоканным _state и моделью (чтобы не грузить веса)."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    import src.service.api as api_module

    st, vecs = mock_state

    # Мокаем _embed_image: возвращает первый вектор из синтетического набора
    def fake_embed(pil_img, state):
        return vecs[0:1].copy()

    with patch.object(api_module, "_state", st), \
         patch.object(api_module, "_embed_image", fake_embed):
        yield TestClient(api_module.app)


class TestApiHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_index_size(self, client):
        data = client.get("/health").json()
        assert "index_size" in data
        assert data["index_size"] > 0

    def test_health_has_model_field(self, client):
        data = client.get("/health").json()
        assert "model" in data


class TestApiConfig:
    def test_config_returns_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_config_has_model_section(self, client):
        data = client.get("/config").json()
        assert "model" in data

    def test_config_has_inference_section(self, client):
        data = client.get("/config").json()
        assert "inference" in data


class TestApiSearch:
    def _post_search(self, client, img: Image.Image, top_k: int = 5, fmt="JPEG"):
        img_bytes = _image_to_bytes(img, fmt)
        return client.post(
            "/search",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            params={"top_k": top_k},
        )

    def test_search_returns_200(self, client):
        resp = self._post_search(client, _make_dummy_image())
        assert resp.status_code == 200

    def test_search_results_count(self, client):
        top_k = 3
        data  = self._post_search(client, _make_dummy_image(), top_k=top_k).json()
        assert len(data["results"]) == top_k

    def test_search_result_fields(self, client):
        data = self._post_search(client, _make_dummy_image()).json()
        assert len(data["results"]) > 0
        result = data["results"][0]
        for field in ("rank", "filename", "similarity_score", "distance"):
            assert field in result, f"Missing field '{field}' in result"

    def test_search_results_ranked_correctly(self, client):
        """Rank должен возрастать от 1."""
        data  = self._post_search(client, _make_dummy_image()).json()
        ranks = [r["rank"] for r in data["results"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_search_similarity_in_valid_range(self, client):
        """similarity_score должен быть в [0, 1]."""
        data = self._post_search(client, _make_dummy_image()).json()
        for r in data["results"]:
            assert 0.0 <= r["similarity_score"] <= 1.0, (
                f"similarity_score={r['similarity_score']} out of [0, 1]"
            )

    def test_search_has_latency_field(self, client):
        data = self._post_search(client, _make_dummy_image()).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_search_png_image(self, client):
        """PNG-изображение тоже должно обрабатываться."""
        resp = self._post_search(client, _make_dummy_image(), fmt="PNG")
        assert resp.status_code == 200

    def test_search_invalid_file_returns_422(self, client):
        """Отправка не-изображения должна вернуть 422."""
        resp = client.post(
            "/search",
            files={"file": ("bad.jpg", b"this is not an image", "image/jpeg")},
            params={"top_k": 5},
        )
        assert resp.status_code == 422

    def test_search_top_k_too_large_clamps(self, client):
        """top_k больше размера индекса — сервис не должен падать."""
        resp = self._post_search(client, _make_dummy_image(), top_k=50)
        assert resp.status_code == 200

    def test_search_top_k_zero_returns_422(self, client):
        """top_k=0 должен вернуть 422 (валидация Query ge=1)."""
        resp = self._post_search(client, _make_dummy_image(), top_k=0)
        assert resp.status_code == 422


# ===========================================================================
# 6. parse_item_id (юнит-тест вспомогательной функции из ноутбука)
# ===========================================================================

class TestParseItemId:
    """
    parse_item_id используется в метриках ноутбука.
    Тестируем здесь, чтобы убедиться в корректности логики.
    """

    def _parse(self, filename: str) -> str:
        from pathlib import Path
        stem = Path(filename).stem
        for part in stem.split("-"):
            if part.startswith("id_"):
                return part
        return stem

    def test_standard_deepfashion_filename(self):
        assert self._parse("MEN-Denim-id_00000089-02_7_additional.jpg") == "id_00000089"

    def test_women_category(self):
        assert self._parse("WOMEN-Blouses_Shirts-id_00001234-01_1_front.jpg") == "id_00001234"

    def test_same_item_different_shots(self):
        id1 = self._parse("MEN-Denim-id_00000089-01_7.jpg")
        id2 = self._parse("MEN-Denim-id_00000089-04_7.jpg")
        assert id1 == id2, "Same item, different shot → должен быть одинаковый id"

    def test_different_items(self):
        id1 = self._parse("MEN-Denim-id_00000080-01_7.jpg")
        id2 = self._parse("MEN-Denim-id_00000089-01_7.jpg")
        assert id1 != id2, "Разные товары → разные id"

    def test_no_id_fallback(self):
        result = self._parse("unknown_image.jpg")
        assert isinstance(result, str)
        assert len(result) > 0
