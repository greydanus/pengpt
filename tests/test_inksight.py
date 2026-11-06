import pytest

torch = pytest.importorskip("torch")
from tools.inksight import bucket


def test_t5_buckets():
    rel = torch.tensor([0, -1, 1, -7, -8, -127, -128, -500])
    assert bucket(rel, True).tolist() == [0, 1, 17, 7, 8, 15, 15, 15]
    rel = torch.tensor([3, 0, -15, -16, -127, -128, -500])
    assert bucket(rel, False).tolist() == [0, 0, 15, 16, 31, 31, 31]
