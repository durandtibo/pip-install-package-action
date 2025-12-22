from __future__ import annotations

import pytest
from feu.imports import is_package_available
from feu.testing import numpy_available


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    is_package_available.cache_clear()


@numpy_available
def test_numpy() -> None:
    import numpy as np  # local import because it is an optional dependency

    assert np.array_equal(np.ones((2, 3)) + np.ones((2, 3)), np.full((2, 3), 2.0))
