import numpy as np


def process_borders(img, const=15):
    h, w = img.shape
    h_ten, w_ten = h // const, w // const
    return img[h_ten:h - h_ten, w_ten:w - w_ten]


def search_shift(img1, img2):
    assert img1.shape == img2.shape
    C = np.real(np.fft.ifft2(np.fft.fft2(img1) * np.conj(np.fft.fft2(img2))))
    index = np.array(np.unravel_index(np.argmax(C, axis=None), C.shape))
    for i in range(2):
        index[i] = index[i] - img1.shape[i] if index[i] > img1.shape[i] // 2 else index[i]
    return index


def crop_images(shift1, shift2, imgs):
    f0 = np.array([max([0, shift1[0], shift2[0]]), max([0, shift1[1], shift2[1]])])
    l0 = np.array([
        min([0, shift1[0], shift2[0]]) + imgs[0].shape[0],
        min([0, shift1[1], shift2[1]]) + imgs[0].shape[1]
    ])

    f1, l1 = f0 - shift1, l0 - shift1
    f2, l2 = f0 - shift2, l0 - shift2

    imgs[0] = imgs[0][f0[0]:l0[0], f0[1]:l0[1]]
    imgs[1] = imgs[1][f1[0]:l1[0], f1[1]:l1[1]]
    imgs[2] = imgs[2][f2[0]:l2[0], f2[1]:l2[1]]

    assert imgs[0].shape == imgs[1].shape == imgs[2].shape

    return imgs


def align(img, g_coord):
    # print('Shape:', img.shape)
    # print('Green coords:', g_coord)
    g_coord = np.array(g_coord)
    one_third = img.shape[0] // 3
    first, second, third = img[:one_third], img[one_third:2 * one_third], img[2 * one_third:3 * one_third]
    assert first.shape == second.shape == third.shape
    first, second, third = process_borders(first), process_borders(second), process_borders(third)

    first_search = search_shift(second, first)
    third_search = search_shift(second, third)

    # print(f'Shifts: b{third_search} and r{first_search}')

    blue_coord = g_coord - first_search + np.array([-one_third, 0])
    red_coord = g_coord - third_search + np.array([one_third, 0])

    red, green, blue = crop_images(third_search, first_search, [second, third, first])

    aligned_img = np.array([red, green, blue]).T
    # print(aligned_img.shape)

    return aligned_img, blue_coord, red_coord
