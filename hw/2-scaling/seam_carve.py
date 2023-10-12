import numpy as np
from scipy import signal


def compute_energy(img):
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    m, n = Y.shape
    Ix = signal.convolve2d(Y, np.array([[-1/2, 0, 1/2]]), mode='same', boundary='symm')
    Ix[:, [0, n - 1]] *= 2
    Iy = signal.convolve2d(Y, np.array([[-1/2], [0], [1/2]]), mode='same', boundary='symm')
    Iy[[0, m - 1], :] *= 2

    return np.sqrt(Ix**2 + Iy**2)


def compute_seam_matrix(energy, mode, mask=None):
    assert mode in ['horizontal', 'vertical']
    if mode != 'horizontal':
        energy = energy.T
        mask = mask.T if mask is not None else mask

    if mask is not None:
        energy = energy.astype(np.float64)
        energy += mask * energy.size * 256

    result = np.zeros_like(energy)
    result[0] = energy[0]

    for row in range(1, result.shape[0]):
        prev_row = result[row - 1]
        for col in range(result.shape[1]):
            start_idx = max(0, col - 1)
            shift_idx = np.argmin(prev_row[start_idx:col + 2])
            idx = start_idx + shift_idx
            result[row, col] = prev_row[idx] + energy[row, col]

    return result if mode == 'horizontal' else result.T


def remove_minimal_seam(img, seam_matrix, mode, real_mask=None):
    assert mode in ['horizontal shrink', 'vertical shrink']
    if mode != 'horizontal shrink':
        img, seam_matrix = np.transpose(img, axes=(1, 0, 2)), seam_matrix.T
        real_mask = real_mask.T if real_mask is not None else real_mask

    path = [np.argmin(seam_matrix[-1])]
    for row in range(seam_matrix.shape[0] - 2, -1, -1):
        start_idx = max(0, path[-1] - 1)
        path.append(start_idx + np.argmin(seam_matrix[row][start_idx:path[-1] + 2]))

    path = np.flip(path)
    mask = np.zeros_like(seam_matrix, dtype=bool)
    mask[np.arange(mask.shape[0]), path] = 1

    result = np.zeros([img.shape[0], img.shape[1] - 1, 3], dtype=np.uint8)
    assert img.shape[2] == 3

    m, n = seam_matrix.shape
    for idx in range(img.shape[2]):
        cur_arr = img[..., idx]
        result[..., idx] = cur_arr[~mask].reshape(m, -1)

    if real_mask is not None:
        real_mask = real_mask[~mask].reshape(m, -1)

    if mode != 'horizontal shrink':
        result, mask = np.transpose(result, axes=(1, 0, 2)), mask.T
        real_mask = real_mask.T if real_mask is not None else real_mask

    return result, real_mask, mask.astype(np.uint8)


def seam_carve(img, mode, mask=None):
    assert mode in ['horizontal shrink', 'vertical shrink']

    e = compute_energy(img)
    seam_matrix = compute_seam_matrix(e, mode.split()[0], mask)

    return remove_minimal_seam(img, seam_matrix, mode, mask)
