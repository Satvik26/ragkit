from __future__ import annotations

import pytest

from ragkit.adapters.dev.stubs import ExtractiveLLM, HashingEmbedder
from ragkit.config.container import build_answer_service
from ragkit.config.settings import Settings


def test_dev_profile_builds() -> None:
    assert build_answer_service(Settings(profile="dev")) is not None


def test_cloud_profile_requires_real_adapters() -> None:
    with pytest.raises(NotImplementedError):
        build_answer_service(Settings(profile="cloud"))


def test_stubs_are_deterministic() -> None:
    e = HashingEmbedder(32)
    assert e.embed(["same text"]) == e.embed(["same text"])
    assert ExtractiveLLM().generate("no evidence here") == "NOT_FOUND"
