from enum import IntEnum
from copy import deepcopy
import numpy as np
from scipy import signal


def get_bayer_masks(n_rows: int, n_cols: int) -> np.ndarray:
    row_repeat, col_repeat = (n_rows + 1) // 2, (n_cols + 1) // 2
    red = np.tile(np.array([[0, 1], [0, 0]], dtype=bool), (row_repeat, col_repeat))
    green = np.tile(np.array([[1, 0], [0, 1]], dtype=bool), (row_repeat, col_repeat))
    blue = np.tile(np.array([[0, 0], [1, 0]], dtype=bool), (row_repeat, col_repeat))
    return np.dstack([red, green, blue])[:n_rows, :n_cols, :]


def get_colored_img(raw_img: np.ndarray) -> np.ndarray:
    masks = get_bayer_masks(*raw_img.shape)
    return masks * raw_img.reshape([*raw_img.shape, 1])


def bilinear_interpolation(colored_img):
    known = get_bayer_masks(*colored_img.shape[:2])

    for index in range(known.shape[2]):
        cur_known = known[..., index]
        unknown = 1 - cur_known
        result = signal.convolve2d(colored_img[..., index] * cur_known, np.ones([3, 3]), mode='same')
        counts = signal.convolve2d(cur_known, np.ones([3, 3]), mode='same')
        result /= counts
        colored_img[..., index] = result * unknown + colored_img[..., index] * cur_known

    return colored_img


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


def improved_interpolation(raw_img):
    colored_img = get_colored_img(raw_img)
    known = get_bayer_masks(*raw_img.shape)

    masks = {
        Color.RED: {
            "same":
                np.array([
                    [0, 4, 0],
                    [4, 0, 4],
                    [0, 4, 0]
                ]),
            "green":
                np.array([
                    [0, 0, 1/2, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 0, 5, 0, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 1/2, 0, 0]
                ]),
            "opp":
                np.array([
                    [0, 0, -3/2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-3/2, 0, 6, 0, -3/2],
                    [0, 0, 0, 0, 0],
                    [0, 0, -3/2, 0, 0]
                ]),
            "opp_same":
                np.array([
                    [2, 0, 2],
                    [0, 0, 0],
                    [2, 0, 2]
                ]),
        },
        Color.GREEN: {
            "other":
                np.array([
                    [0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-1, 0, 4, 0, -1],
                    [0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0]
                ]),
            "same":
                np.array([
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0],
                ]),
        }
    }
    masks[Color.BLUE] = masks[Color.RED]

    def cut_result(arr):
        return np.minimum(np.maximum(arr, 0), 255)

    red_known = known[..., Color.RED]
    green_known = known[..., Color.GREEN]
    blue_known = known[..., Color.BLUE]

    even_rows = np.zeros_like(red_known)
    even_rows[np.arange(even_rows.shape[0]) % 2 == 0] = 1

    even_cols = np.zeros_like(red_known)
    even_cols[:, np.arange(even_cols.shape[1]) % 2 == 0] = 1

    red_masked = colored_img[..., Color.RED] * red_known
    green_masked = colored_img[..., Color.GREEN] * green_known
    blue_masked = colored_img[..., Color.BLUE] * blue_known

    all_known = [red_known, green_known, blue_known]
    all_masked = [red_masked, green_masked, blue_masked]

    # GREEN color
    green_common = signal.convolve2d(green_masked, masks[Color.GREEN]["same"], mode="same")
    green_at_red = signal.convolve2d(red_masked, masks[Color.GREEN]["other"], mode="same")
    green_at_blue = signal.convolve2d(blue_masked, masks[Color.GREEN]["other"], mode="same")

    colored_img[..., Color.GREEN] = cut_result(
        green_masked + (
                (1 - green_known) * green_common +
                red_known * green_at_red +
                blue_known * green_at_blue
        ) / 8
    )

    # RED and BLUE colors
    row_green = signal.convolve2d(green_masked, masks[Color.RED]["green"], mode="same")
    col_green = signal.convolve2d(green_masked, masks[Color.RED]["green"].T, mode="same")

    for color, rows, cols in zip([Color.RED, Color.BLUE], [even_rows, 1 - even_rows], [1 - even_cols, even_cols]):
        opp_color = 2 - color
        same_at_green = signal.convolve2d(all_masked[color], masks[color]["same"], mode="same")

        same_at_opp = signal.convolve2d(all_masked[color], masks[color]["opp_same"], mode="same")
        opp_at_opp = signal.convolve2d(all_masked[opp_color], masks[color]["opp"], mode="same")

        colored_img[..., color] = cut_result(
            all_masked[color] + (
                    green_known * rows * (row_green + same_at_green) +
                    green_known * cols * (col_green + same_at_green) +
                    all_known[opp_color] * (same_at_opp + opp_at_opp)
            ) / 8
        )

    return colored_img


def compute_psnr(img_pred, img_gt):
    assert img_pred.shape == img_gt.shape
    img_pred, img_gt = img_pred.astype(np.float64), img_gt.astype(np.float64)
    mse = np.sum((img_pred - img_gt) ** 2) / img_pred.size
    if mse == 0.:
        raise ValueError("Images are equal")
    return 10 * np.log10(np.max(img_gt) ** 2 / mse)
