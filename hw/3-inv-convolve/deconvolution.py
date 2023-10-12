import numpy as np
import scipy.fft as fft
from sklearn.metrics import mean_squared_error


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * x**2 / sigma**2)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h_reshaped = np.zeros(shape)
    x, y = (shape[0] - h.shape[0] + 1) // 2, (shape[1] - h.shape[1] + 1) // 2
    h_reshaped[x:x + h.shape[0], y:y + h.shape[1]] = h
    return fft.fft2(fft.ifftshift(h_reshaped))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    mask = np.abs(H) > threshold
    H_inv = np.zeros_like(H)
    H_inv[mask] = 1 / H[mask]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    H_inv = inverse_kernel(fourier_transform(h, blurred_img.shape), threshold)
    return np.abs(fft.ifft2(fft.fft2(blurred_img) * H_inv))


def wiener_filtering(blurred_img, h, K=4.29e-5):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    return np.abs(fft.ifft2(np.conj(H) / (np.abs(H)**2 + K) * fft.fft2(blurred_img)))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    return 10 * np.log10(255**2 / mean_squared_error(img1, img2, squared=True))
