import cupy as cp
from cupy.fft import fft, ifft, rfft, irfft, fft2, ifft2, fftshift
from math import *
from typing import Tuple

def amplitudeSpectrum(image : cp.ndarray) -> cp.ndarray:
    boxsize = max(image.shape)
    boxsize += boxsize % 2
    spectrum = fftshift(cp.abs(fft2(image, s = (boxsize, boxsize))))
    spectrum[boxsize // 2, :] = 0
    spectrum[:, boxsize // 2] = 0
    return spectrum

def crop(image : cp.ndarray, boxsize : int) -> cp.ndarray:
    initialBoxsize, targetBoxsize = max(image.shape), boxsize
    center = slice((initialBoxsize - targetBoxsize) // 2, (initialBoxsize + targetBoxsize) // 2)
    return image[center, center]

def resize(image : cp.ndarray, boxsize : int) -> cp.ndarray:
    initialBoxsize, targetBoxsize = max(image.shape), boxsize
    assert initialBoxsize % 2 == targetBoxsize % 2 == 0

    fimage = fftshift(fft2(image, norm = 'forward'))
    if initialBoxsize >= targetBoxsize:
        fimage = crop(fimage, targetBoxsize)
    else:
        raise NotImplementedError
    return ifft2(fftshift(fimage), norm = 'forward').real

def lowpassBeta(polargram : cp.ndarray, width : int = 2) -> cp.ndarray:
    assert polargram.ndim == 2
    n = polargram.shape[0]
    freq = cp.arange(n // 2 + 1, dtype = cp.float32)
    window = cp.where(freq <= width, (cp.cos(freq / width * cp.pi) + 1) / 2, 0)
    return irfft(rfft(polargram, axis = 0) * window[:, None], axis = 0).copy()

def extractImf(polargram : cp.ndarray, m : int, alpha : float = 0.5, reflect : bool = True) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    if reflect:
        nTheta = polargram.shape[-1] - 1
        polargramExt = cp.empty((*polargram.shape[:-1], 2 * nTheta), dtype = cp.float32)
        polargramExt[..., : nTheta] = polargram[..., : nTheta]
        polargramExt[..., nTheta :] = polargram[..., nTheta : 0 : -1]
        imf, u, v = extractImf(polargramExt, 2 * m, alpha, False)
        return imf[..., : nTheta + 1].copy(), u[..., : nTheta + 1].copy(), v[..., : nTheta + 1].copy()

    nTheta = polargram.shape[-1]
    assert nTheta % 2 == 0 and m < nTheta // 2

    # Bandpass filter, centered at m.
    width = min(m * alpha, nTheta // 2 - m)
    freq = cp.abs(fftshift(cp.arange(nTheta, dtype = cp.float32) - nTheta // 2))
    window = cp.where(cp.abs(freq - m) < width, (cp.cos(cp.abs(freq - m) / width * cp.pi) + 1) / 2, 0)
    fimf = fft(polargram) * window
    imf = ifft(fimf).real

    # Compute envelope function.
    fenv = cp.zeros_like(fimf, dtype = cp.complex64)
    width = floor(width)
    for j in range(-width, width + 1):
        fenv[..., j] = fimf[..., m + j]
    env = ifft(fenv)
    v, u = 2 * env.real, -2 * env.imag

    return imf, u, v

def stripAmplitudeSpectrum(
    image : cp.ndarray,
    boxsize : int,
    oversampling : int,
    pixelsize : float,
    tiltangle : float
) -> Tuple[cp.ndarray, cp.ndarray]:

    nx, ny = image.shape
    step = boxsize // 2 ** oversampling
    lxRange = range(ceil(-(nx - boxsize) / 2 / step), floor(((nx - boxsize) / 2 - 1) / step) + 1)
    lyRange = range(ceil(-(ny - boxsize) / 2 / step), floor(((ny - boxsize) / 2 - 1) / step) + 1)

    spectrum = cp.zeros((len(lxRange), boxsize, boxsize), dtype = cp.float32)
    dz = cp.empty(len(lxRange), dtype = cp.float32)
    for i, lx in enumerate(lxRange):
        for ly in lyRange:
            x = nx // 2 - boxsize // 2 + lx * step
            y = ny // 2 - boxsize // 2 + ly * step
            patch = image[y : y + boxsize, x : x + boxsize]
            spectrum[i] += amplitudeSpectrum(patch)
        dz[i] = lx * step * pixelsize * tan(radians(tiltangle))
    return spectrum, dz
