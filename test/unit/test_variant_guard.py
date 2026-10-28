#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the Mitsuba variant guard of the solvers"""

import pytest
import mitsuba as mi

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
from sionna.rt.utils.misc import check_mitsuba_variant


def _unsupported_variant():
    """Returns an available variant unsupported by Sionna RT, or None"""
    for variant in mi.variants():
        if "mono_polarized" not in variant:
            return variant
    return None


def test_check_mitsuba_variant_raises():
    """A non-polarized variant triggers a RuntimeError with a clear message"""
    original = mi.variant()
    unsupported = _unsupported_variant()
    if unsupported is None:
        pytest.skip("No non-polarized Mitsuba variant available")
    try:
        mi.set_variant(unsupported)
        with pytest.raises(RuntimeError, match="mono_polarized"):
            check_mitsuba_variant()
    finally:
        mi.set_variant(original)


def test_check_mitsuba_variant_accepts_polarized():
    """A configured polarized variant passes the guard"""
    original = mi.variant()
    polarized = [v for v in mi.variants() if "mono_polarized" in v]
    if original is None and not polarized:
        pytest.skip("No polarized Mitsuba variant available")
    try:
        if original is None:
            mi.set_variant(polarized[0])
        check_mitsuba_variant()
    finally:
        if original is not None:
            mi.set_variant(original)


def test_path_solver_clear_error_on_wrong_variant():
    """PathSolver fails fast with a clear error when the variant is wrong"""
    original = mi.variant()
    unsupported = _unsupported_variant()
    if unsupported is None or original is None:
        pytest.skip("No non-polarized variant available or no variant set")
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter("tx", position=[0.0, 0.0, 20.0]))
    scene.add(Receiver("rx", position=[30.0, 0.0, 1.5]))
    try:
        mi.set_variant(unsupported)
        with pytest.raises(RuntimeError, match="mono_polarized"):
            PathSolver()(scene, max_depth=0)
    finally:
        mi.set_variant(original)
