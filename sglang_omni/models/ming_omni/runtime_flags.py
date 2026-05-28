# SPDX-License-Identifier: Apache-2.0
"""Env-gated runtime flags for the Ming-Omni launchers.

The Ming launchers (`examples/run_ming_omni_server.py`,
`examples/run_ming_omni_speech_server.py`,
`examples/run_ming_omni_speech.py`) historically pass
``disable_custom_all_reduce=True`` to SGLang at ``tp_size > 1`` so the BF16
tensor-parallel all-reduce falls back to NCCL ring-LL even when SGLang's own
``CustomAllReduceV2`` / ``CustomAllreduce`` would be eligible (full-NVLink TP
group, small message size). That blanket disable predates SGLang's NVLink-aware
selection logic and is overly conservative on H200 / H100 SXM nodes where
NVLink is fully connected.

This module provides an opt-in env switch that lets the Ming launchers leave
``disable_custom_all_reduce`` unset at ``tp_size > 1``, so SGLang's own
``should_custom_ar`` / dispatch picks the custom path when eligible and falls
back to NCCL safely otherwise. Default behavior (env unset) is unchanged.
"""

from __future__ import annotations

import os

_MING_CUSTOM_ALL_REDUCE_ENV = "SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE"


def ming_custom_all_reduce_opt_in() -> bool:
    """True iff ``SGLANG_OMNI_MING_CUSTOM_ALL_REDUCE=1`` is set (literal ``"1"``).

    Other values (``"0"``, ``"true"``, ``"yes"``, empty, unset) leave the
    default behavior in place: the Ming launchers still set
    ``disable_custom_all_reduce=True`` at ``tp_size > 1``.
    """
    return os.environ.get(_MING_CUSTOM_ALL_REDUCE_ENV) == "1"


def ming_should_disable_custom_all_reduce(tp_size: int | None) -> bool:
    """Whether a Ming launcher should set ``disable_custom_all_reduce=True``.

    Returns ``True`` when ``tp_size > 1`` AND the env opt-in is NOT set —
    i.e., the historical default. Returns ``False`` when ``tp_size <= 1``
    (custom AR irrelevant) or when the env opt-in is set (let SGLang pick).
    """
    if not tp_size or tp_size <= 1:
        return False
    return not ming_custom_all_reduce_opt_in()
