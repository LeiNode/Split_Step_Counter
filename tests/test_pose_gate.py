"""The counters must ignore frames where the lower body isn't fully visible."""

from types import SimpleNamespace

import pytest

from split_step.detector import KEY_LANDMARKS, LEFT_ANKLE, SplitStepDetector


def _landmarks(hidden=()):
    return [SimpleNamespace(x=0.0, y=0.1 * i, z=0.0, visibility=0.0 if i in hidden else 0.99)
            for i in range(33)]


@pytest.fixture(scope="module")
def lstm_cls():
    pytest.importorskip("torch")
    from split_step.lstm_counter import LSTMCounter
    return LSTMCounter


def test_rules_detector_skips_partial_pose():
    d = SplitStepDetector(clock=lambda: 0.0)
    assert d.process(_landmarks(hidden={LEFT_ANKLE})) is False
    assert d.tracking is False
    d.process(_landmarks())
    assert d.tracking is True


def test_lstm_skips_model_on_partial_pose(lstm_cls):
    c = lstm_cls(clock=lambda: 0.0)
    calls = []
    c._model.register_forward_hook(lambda *a: calls.append(1))
    for lm in KEY_LANDMARKS:
        assert c.process(_landmarks(hidden={lm})) is False
        assert c.tracking is False
    assert calls == []                 # model never ran
    c.process(_landmarks())
    assert c.tracking is True and calls == [1]


def test_lstm_gate_can_be_disabled(lstm_cls):
    c = lstm_cls(clock=lambda: 0.0, require_full_pose=False)
    c.process(_landmarks(hidden={LEFT_ANKLE}))
    assert c.tracking is True
