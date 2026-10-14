from pathlib import Path

import pytest

from oagbench.config import get_settings
from oagbench.harness import run_benchmark
from oagbench.scoring import score_run

APP = Path(__file__).parents[1] / "src" / "oagbench" / "dashboard" / "app.py"


@pytest.fixture
def scored_fake_run(settings, corpus, monkeypatch):
    out = run_benchmark(settings, corpus, None, split="test", pipelines=["A", "B", "C", "B_oracle"], repeats=1)
    score_run(out, corpus)
    monkeypatch.setenv("RESULTS_DIR", str(settings.results_dir))
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    get_settings.cache_clear()
    yield out
    get_settings.cache_clear()


def test_dashboard_renders_fake_run_with_warning(scored_fake_run) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("FAKE RESULTS" in w.value for w in at.warning)
    assert len(at.metric) == 4
