import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from graphify_search.embed import LocalEmbeddingClient

REPO = Path(__file__).resolve().parents[1]
# inv: the weights the MiniLM cell's harness.yaml pins, so what is embedded below is that cell's
CELL = REPO / "benchmark" / "systems" / "graphify-search-minilm"
HUB = CELL / "models"
NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384
# inv: the floor the [local] extra declares; below it the installed package is not the one the
# extra would have installed, so its bytes say nothing about what this extra embeds with
FLOOR = 6
TEXTS = ["def render_invoice(scores):", "where is the invoice score calculated",
         "README: how the billing is built"]

# inv: the child prints raw float32 bytes, so two runs are compared as the vectors themselves
# rather than as a rounded rendering of them
SCRIPT = """
import json, sys
from graphify_search.embed import LocalEmbeddingClient
client = LocalEmbeddingClient(sys.argv[1], "", "")
sys.stdout.buffer.write(client.embed_documents(json.loads(sys.argv[2])).tobytes())
"""


def _env(cache):
    return {**os.environ, "HF_HUB_CACHE": str(cache), "HF_HOME": str(cache.parent / "hf-home"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TQDM_DISABLE": "1"}


def _installed_major():
    # inv: the weights can sit on disk in a tree whose environment lacks the [local] extra, and an
    # absent loader is the same "cannot run" state as absent weights
    sentence_transformers = pytest.importorskip("sentence_transformers")
    return int(sentence_transformers.__version__.split(".")[0])


@pytest.fixture
def offline_hub(tmp_path, monkeypatch):
    src = HUB / f"models--{NAME.replace('/', '--')}"
    if not src.is_dir():
        pytest.skip(f"the frozen weights are not here: {src}")
    major = _installed_major()
    if major < FLOOR:
        pytest.skip(f"sentence-transformers {major}.x is below the [local] extra's floor of {FLOOR}")
    cache = tmp_path / "hub"
    # why: the frozen copy is the read-only record of what this cell ran, so the model is loaded
    # from a copy and a loader that writes a marker cannot touch it
    shutil.copytree(HUB, cache, symlinks=True)
    for key, value in _env(cache).items():
        monkeypatch.setenv(key, value)
    return cache


@pytest.mark.slow
def test_the_real_weights_embed_to_unit_rows_of_the_models_width(offline_hub):
    vecs = LocalEmbeddingClient(NAME, "", "").embed_documents(TEXTS)
    assert vecs.shape == (len(TEXTS), DIMS)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_two_separate_processes_embed_the_same_text_to_the_same_bytes(offline_hub):
    argv = [sys.executable, "-c", SCRIPT, NAME, json.dumps(TEXTS)]
    runs = [subprocess.run(argv, env=_env(offline_hub), capture_output=True, check=True) for _ in range(2)]
    first, second = runs[0].stdout, runs[1].stdout
    assert len(first) == len(TEXTS) * DIMS * 4
    # inv: the vectors are compared as bytes, not as a rounded rendering, because a rendering
    # hides exactly the last-bit drift this test exists to catch
    assert first == second
