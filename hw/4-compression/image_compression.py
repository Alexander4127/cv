import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# ! Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Отцентруем каждую строчку матрицы
    mean_val = np.mean(matrix, axis=1)
    M = matrix - mean_val[:, None]
    # Найдем матрицу ковариации
    cov = M @ M.T
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    ei_val, ei_vec = np.linalg.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    # n_vec = ei_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    sort_val = np.flip(np.argsort(ei_val))
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # ! Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    ei_vec = ei_vec[:, sort_val]
    # Оставляем только p собственных векторов
    ei_vec = ei_vec[:, :p]
    # Проекция данных на новое пространство

    proj = ei_vec.T @ M

    return ei_vec, proj, mean_val


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # ! Это следует из описанного в самом начале примера!
        ei_vec, proj, mean_val = comp
        result_img.append(ei_vec @ proj + mean_val[:, None])

    ans = np.transpose(np.array(result_img), axes=(1, 2, 0))

    return np.clip(ans, a_min=0, a_max=255)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[..., j], p))
            pass

        axes[i // 3, i % 3].imshow(pca_decompression(compressed).astype(np.uint8))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    b = np.array([0, 128, 128])
    c = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.4187, -0.0813]
    ])

    return np.transpose(c @ np.transpose(img, axes=(0, 2, 1)), axes=(0, 2, 1)) + b


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    b = np.array([0, 128, 128])
    c = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -0.71414],
        [1, 1.77, 0]
    ])
    
    return np.transpose(c @ np.transpose(img - b, axes=(0, 2, 1)), axes=(0, 2, 1))


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycb_img = rgb2ycbcr(rgb_img)
    for idx in [1, 2]:
        ycb_img[..., idx] = gaussian_filter(ycb_img[..., idx], sigma=10)

    img = ycbcr2rgb(ycb_img)

    plt.imshow(img.astype(np.uint8))
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycb_img = rgb2ycbcr(rgb_img)
    ycb_img[..., 0] = gaussian_filter(ycb_img[..., 0], sigma=10)

    img = ycbcr2rgb(ycb_img)

    plt.imshow(img.astype(np.uint8))
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """

    component = component.reshape(*component.shape[:2])
    rows, cols = np.arange(component.shape[0]), np.arange(component.shape[1])
    result = gaussian_filter(component, sigma=10)[rows % 2 == 0][:, cols % 2 == 0]

    return result


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    assert block.shape == (8, 8)
    g = np.zeros_like(block, dtype=np.float64)
    xs = ys = np.pi * (2 * np.arange(8) + 1) / 16
    for u in range(8):
        for v in range(8):
            m_cos = np.outer(np.cos(xs * u), np.cos(ys * v))
            g[u, v] = np.sum(block * m_cos)

    g[0] /= np.sqrt(2)
    g[:, 0] /= np.sqrt(2)

    return g / 4


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    s = max(1, 5000 / q if q < 50 else 200 - 2 * q)
    return np.maximum(np.floor((50 + s * default_quantization_matrix) / 100), 1)


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    assert block.shape == (8, 8)
    return np.hstack([np.diagonal(np.fliplr(block), i)[::(2 * ((i + 1) % 2) - 1)] for i in range(7, -8, -1)])


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    result_list = []
    n_zeros = 0
    for el in zigzag_list:
        if el == 0:
            n_zeros += 1
            if n_zeros != 1:
                continue
        elif n_zeros > 0:
            result_list.append(n_zeros)
            n_zeros = 0
        result_list.append(el)

    if n_zeros > 0:
        result_list.append(n_zeros)

    return result_list


def jpeg_compression(img, q_matrices):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Переходим из RGB в YCbCr
    img = rgb2ycbcr(img).astype(np.int64)
    # Уменьшаем цветовые компоненты
    lst = [img[..., 0], downsampling(img[..., 1]), downsampling(img[..., 2])]
    assert np.all(lst[i].shape[0] % 8 == 0 and lst[i].shape[1] % 8 == 0 for i in range(len(lst)))
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    blocks = [[] for _ in range(len(lst))]
    for idx, el in enumerate(lst):
        for i in range(0, el.shape[0], 8):
            for j in range(0, el.shape[1], 8):
                blocks[idx].append(el[i:i + 8, j: j + 8] - 128)
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    result = [[] for _ in range(len(lst))]
    for idx, (sub_blocks, q_matrix) in enumerate(zip(
            blocks,
            [q_matrices[0], q_matrices[1], q_matrices[1]]
    )):
        for block in sub_blocks:
            result[idx].append(compression(zigzag(quantization(dct(block), q_matrix))))

    return result


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    initial_list = []
    prev_zero = False
    for el in compressed_list:
        if prev_zero:
            initial_list.extend([0] * el)
            prev_zero = False
        elif el == 0:
            prev_zero = True
        else:
            initial_list.append(el)

    return initial_list


def inverse_zigzag(list_elem):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка,
        расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    rows = np.repeat(np.arange(8), 8).reshape(8, 8)
    row_idx = np.hstack([np.diagonal(np.fliplr(rows), i)[::(2 * ((i + 1) % 2) - 1)] for i in range(7, -8, -1)])
    col_idx = np.hstack([np.diagonal(np.fliplr(rows.T), i)[::(2 * ((i + 1) % 2) - 1)] for i in range(7, -8, -1)])

    result = np.zeros([8, 8])
    result[row_idx, col_idx] = list_elem

    return result


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    assert block.shape == (8, 8)
    g = np.zeros_like(block, dtype=np.float64)
    us = vs = np.pi * np.arange(8) / 16
    for x in range(8):
        for y in range(8):
            m_cos = np.outer(np.cos((2 * x + 1) * us), np.cos((2 * y + 1) * vs))
            m_cos[0] /= np.sqrt(2)
            m_cos[:, 0] /= np.sqrt(2)
            g[x, y] = np.sum(block * m_cos)

    return np.round(g / 4)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    return np.repeat(np.repeat(component, 2, axis=1), 2, axis=0)


def jpeg_decompression(result, result_shape, q_matrices):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    assert len(result) == 3 and result_shape[2] == 3
    result_shape = np.array(result_shape)[:2]

    for sub_list, q_matrix in zip(result, [q_matrices[0], q_matrices[1], q_matrices[1]]):
        for idx, comp_list in enumerate(sub_list):
            block = inverse_zigzag(inverse_compression(comp_list))
            sub_list[idx] = inverse_dct(inverse_quantization(block, q_matrix))

    components = []
    for blocks, shape in zip(result, [result_shape, (result_shape + 1) // 2, (result_shape + 1) // 2]):
        cur_component = np.zeros((shape + 7) // 8 * 8, dtype=np.int64)
        dx, dy = (shape + 7) // 8
        assert dx * dy == len(blocks)
        for i in range(0, shape[0], 8):
            for j in range(0, shape[1], 8):
                cur_component[i:i + 8, j:j + 8] = blocks[i // 8 * dy + j // 8]
        cur_component += 128
        components.append(cur_component[:shape[0], :shape[1]])

    for idx in [1, 2]:
        components[idx] = upsampling(components[idx])[:result_shape[0], :result_shape[1]]

    assert components[0].shape == components[1].shape == components[2].shape
    img = np.transpose(np.array(components), axes=(1, 2, 0))

    return np.clip(ycbcr2rgb(img), 0, 255)


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        q_matrices = [
            own_quantization_matrix(y_quantization_matrix, p),
            own_quantization_matrix(color_quantization_matrix, p)
        ]
        comp_image = jpeg_compression(img, q_matrices)

        axes[i // 3, i % 3].imshow(jpeg_decompression(comp_image, img.shape[:2], q_matrices).astype(np.uint8))
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    else:
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), np.array(compressed, dtype=np.object_))
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")
