from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_model import sample
import data


def test_streamlit_research_renders_all_targets(monkeypatch):
    monkeypatch.setattr(data, 'get_prices', lambda *args: sample(430))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'legacy_app.py'), default_timeout=60)
    app.run()
    app.slider[0].set_value(12)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    labels = [metric.label for metric in app.metric]
    assert '第5日收盘 Q50' in labels
    assert '未来5日最高 Q50' in labels
    assert '未来5日最低 Q50' in labels
    assert len(app.get('plotly_chart')) == 6
    assert len(app.dataframe) == 2
