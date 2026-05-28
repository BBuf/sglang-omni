# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in ``SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE`` env gate.

The Ming launchers historically pass ``disable_custom_all_reduce=True`` to
SGLang at ``tp_size > 1``. The helper in
``sglang_omni.models.ming_omni.runtime_flags`` lets users opt back into
SGLang's own selection (which then picks the custom all-reduce path when
eligible and falls back to NCCL otherwise) by setting
``SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE=1``.

Default behavior (env unset, or set to anything other than the literal ``"1"``)
must be unchanged.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.ming_omni.runtime_flags import (
    ming_custom_all_reduce_opt_in,
    ming_should_disable_custom_all_reduce,
)


@pytest.fixture(autouse=True)
def _clear_ming_custom_all_reduce_env(monkeypatch):
    """Ensure each test starts with the env var unset."""
    monkeypatch.delenv("SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE", raising=False)


def test_env_unset_is_default_off():
    assert ming_custom_all_reduce_opt_in() is False


def test_env_set_to_1_enables_opt_in(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE", "1")
    assert ming_custom_all_reduce_opt_in() is True


@pytest.mark.parametrize(
    "value", ["0", "true", "True", "TRUE", "yes", "on", "", " 1", "1 "]
)
def test_env_only_literal_1_enables_opt_in(monkeypatch, value):
    """Only the literal ``"1"`` enables; other truthy-looking strings do not."""
    monkeypatch.setenv("SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE", value)
    assert ming_custom_all_reduce_opt_in() is False


@pytest.mark.parametrize("tp_size", [None, 0, 1])
def test_should_disable_is_false_when_tp_le_1(tp_size):
    """At tp_size <= 1 (or None) custom AR is irrelevant; never set disable."""
    assert ming_should_disable_custom_all_reduce(tp_size) is False


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_should_disable_default_true_at_tp_gt_1(tp_size):
    """Historical default: at tp_size > 1 the Ming launcher disables custom AR."""
    assert ming_should_disable_custom_all_reduce(tp_size) is True


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_should_disable_false_when_opt_in_at_tp_gt_1(monkeypatch, tp_size):
    """Opt-in path: at tp_size > 1 with env=1, leave the flag unset."""
    monkeypatch.setenv("SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE", "1")
    assert ming_should_disable_custom_all_reduce(tp_size) is False
