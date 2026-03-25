from __future__ import annotations

import numpy as np

from neural_astar.utils.residual_confidence import (
    build_residual_confidence_map,
    resolve_residual_confidence_map,
)


def test_build_residual_confidence_map_suppresses_isolated_spike():
    occ = np.zeros((7, 7), dtype=np.float32)
    residual = np.ones((7, 7), dtype=np.float32)
    residual[3, 3] = 8.0

    confidence = build_residual_confidence_map(
        residual,
        occ,
        mode="spike_suppression",
        kernel_size=3,
        strength=1.0,
        min_confidence=0.2,
    )

    assert confidence.shape == residual.shape
    assert confidence[3, 3] < confidence[2, 2]
    assert np.all(confidence >= 0.2)
    assert np.all(confidence <= 1.0)


def test_resolve_residual_confidence_map_learned_spike_multiplies_sources():
    occ = np.zeros((5, 5), dtype=np.float32)
    residual = np.ones((5, 5), dtype=np.float32)
    residual[2, 2] = 6.0
    learned = np.full((5, 5), 0.5, dtype=np.float32)

    conf = resolve_residual_confidence_map(
        mode="learned_spike",
        occ_map=occ,
        residual_map=residual,
        learned_confidence_map=learned,
        kernel_size=3,
        strength=1.0,
        min_confidence=0.2,
    )

    assert conf is not None
    assert conf.shape == residual.shape
    assert conf[2, 2] < 0.5
    assert np.all(conf <= 0.5 + 1e-6)
