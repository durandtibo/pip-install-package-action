from __future__ import annotations

import pytest
from feu.imports import is_package_available
from feu.testing import (
    torch_available,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    is_package_available.cache_clear()


@torch_available
def test_torch() -> None:
    import torch  # local import because it is an optional dependency

    torch.testing.assert_close(torch.ones(2, 3) + torch.ones(2, 3), torch.full((2, 3), 2.0))
