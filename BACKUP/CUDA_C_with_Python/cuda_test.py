import numpy as np
import ctypes
import cv2
from ctypes import *


def get_cuda_gray():
    dll = ctypes.CDLL('cuda/x64/Debug/cuda.dll', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_gray
    func.argtypes = [POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), c_size_t]
    return func


__cuda_gray = get_cuda_gray()


def cuda_gray(a, b, c, d, size):
    a_p = a.ctypes.data_as(POINTER(c_ubyte))
    b_p = b.ctypes.data_as(POINTER(c_ubyte))
    c_p = c.ctypes.data_as(POINTER(c_ubyte))
    d_p = d.ctypes.data_as(POINTER(c_ubyte))
    __cuda_gray(a_p, b_p, c_p, d_p, size)


if __name__ == '__main__':

    img = cv2.imread('river.jpeg')
    rows, columns = img.shape[:2]
    b, g, r = cv2.split(img)
    gray = np.zeros((rows * columns, 1))

    b = np.copy(b.reshape((rows * columns, 1))).astype('uint8')
    g = np.copy(g.reshape((rows * columns, 1))).astype('uint8')
    r = np.copy(r.reshape((rows * columns, 1))).astype('uint8')
    gray = np.copy(gray).astype('uint8')

    cuda_gray(b, g, r, gray, rows * columns)
    gray = gray.reshape(rows, columns,)
    cv2.imwrite('gray.jpg', gray)
