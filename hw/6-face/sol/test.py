import numpy as np
import torch


from sol.utils import pred2coord, coord2pred


def test_pred_and_coord():
    lst = [
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[2, 4], [1, 3]]),
            torch.tensor([[0.5, 0.5, 1.5], [4, 5/3, 6]], dtype=torch.float32) - 0.5
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[3, 2]]),
            torch.tensor([[1/3, 1, 1, 2]], dtype=torch.float32) - 0.5
        ),
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            torch.tensor([[1, 2], [2, 2], [2, 3]]),
            torch.tensor([[1, 1, 3, 2], [2.5, 3, 3.5, 4], [4.5, 10/3, 5.5, 4]]) - 0.5
        )
    ]

    for real_ans, size, pred in lst:
        assert np.allclose(pred2coord(pred, size), real_ans), f'{pred2coord(pred, size)}'
        assert np.allclose(coord2pred(real_ans, size), pred), f'{coord2pred(real_ans, size)}'
