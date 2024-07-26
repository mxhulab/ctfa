import argparse
import mrcfile
import os
import sys
import numpy as np
import cupy as cp
from math import *
from .utility import amplitudeSpectrum, resize, crop, lowpassBeta, extractImf
from .utility import stripAmplitudeSpectrum
from .kernels import sample, equiphaseAverage, normalEquation, computeCC

def parse():
    global args

    parser = argparse.ArgumentParser(description = 'CTFA: Cryo-EM CTF (Contrast Transfer Function) parameter estimation software by TFA (Time-Frequency Analysis).')

    basicGroup = parser.add_argument_group('Basic arguments')
    basicGroup.add_argument('micrograph',   type = str,                    help = 'Input micrograph. It should be like `[n@]fpath`, denoting the (n-th slice of) MRC file fpath.')
    basicGroup.add_argument('--pixelsize',  type = float, required = True, help = 'Pixelsize in Angstrom.')
    basicGroup.add_argument('--voltage',    type = float, required = True, help = 'Voltage in kV.')
    basicGroup.add_argument('--cs',         type = float, required = True, help = 'Cs in mm.')
    basicGroup.add_argument('--amplitude',  type = float, required = True, help = 'Amplitude contrast.')
    basicGroup.add_argument('--boxsize',    type = int,   default = 512,   help = 'Boxsize for amplitude spectrum, 512 by default.')
    basicGroup.add_argument('--minres',     type = float, default = 50,    help = 'Minimum resolution for fitting, 50 Angstrom by default.')
    basicGroup.add_argument('--maxres',     type = float, default = 4,     help = 'Maximum resolution for fitting, 4 Angstrom by default.')
    basicGroup.add_argument('--maxiter',    type = int,   default = 20,    help = 'Maximum number of iteration, 10 by default.')
    basicGroup.add_argument('--vppdata',    action = 'store_true',         help = 'Estimate phase plate for VPP data.')

    advancedGroup = parser.add_argument_group('Advanced arguments')
    advancedGroup.add_argument('--defocus',    type = str,   default = "0.8,1.6,3.2", help = 'Minimal guess(es) of defocus in um, seperated by ",", "0.8,1.6,3.2" by default.')
    advancedGroup.add_argument('--noresample', action = 'store_true',      help = 'Do not resample if pixelsize is too small (< 1.4 Angstrom).')
    advancedGroup.add_argument('--debug',      action = 'store_true',      help = 'Write extra files for debugging.')

    tiltGroup = parser.add_argument_group('Tilt arguments')
    tiltGroup.add_argument('--tilt',         action = 'store_true',         help = 'Enable tilt image mode.')
    tiltGroup.add_argument('--tiltangle',    type = float, default = '0',   help = 'Tilt angle in degree, 0 by default.')
    # tiltGroup.add_argument('--tiltaxis',     type = float, default = '180', help = 'Stage tilt axis angle in degree (relative to negative direction of the y-axis), 180 by default.')
    tiltGroup.add_argument('--oversampling', type = int,   default = 3,     help = 'Oversampling of patch.')

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    args = parser.parse_args()

def readMicrograph(fpath):
    if (pos := fpath.find('@')) >= 0:
        n, micrographPath = int(fpath[:pos]) - 1, os.path.abspath(fpath[pos + 1:])
        print(f'Load {n + 1}-th slice from MRC file "{micrographPath}"')
    else:
        n, micrographPath = None, os.path.abspath(fpath)
        print(f'Load micrograph from MRC file "{micrographPath}"')

    with mrcfile.mmap(micrographPath, permissive = True) as mrc:
        data = mrc.data
        shape = data.shape
        ndim = len(shape)

    if n is None and ndim == 3 and shape[0] > 1:
        raise NotImplementedError('Input MRC file contains a movie. Please do motion correction to get a micrograph, or use prefix `n@` to designate a slice.')
    elif n is None and ndim == 3 and shape[0] == 1:
        return cp.array(data[0], dtype = cp.float32)
    elif n is None and ndim == 2:
        return cp.array(data, dtype = cp.float32)
    elif n is not None and ndim == 3 and not (0 <= n < shape[0]):
        raise RuntimeError(f'{n + 1}-th slice does not exist.')
    elif n is not None and ndim == 3:
        return cp.array(data[n], dtype = cp.float32)
    elif n == 1 and ndim == 2:
        return cp.array(data, dtype = cp.float32)
    elif n > 1 and ndim == 2:
        raise RuntimeError(f'{n + 1}-th slice does not exist.')
    else:
        raise RuntimeError('Unknown error happened when reading the micrograph.')

def computeAmplitudeSpectrum(micrograph):
    global args

    if args.tilt:
        print(f'ShiCTF for tilt image. Resample disabled.')
        pixelsize = args.pixelsize
        spectrum, zs = stripAmplitudeSpectrum(micrograph, args.boxsize, args.oversampling, pixelsize, args.tiltangle)
    elif not args.noresample and args.pixelsize < 1.4:
        boxsize = round(args.boxsize * 1.4 / args.pixelsize)
        boxsize += boxsize % 2
        pixelsize = args.pixelsize * boxsize / args.boxsize
        print(f'Resample pixelsize from {args.pixelsize:.3f}A to {pixelsize:.3f}A.')
        spectrum = amplitudeSpectrum(micrograph)
        spectrum = resize(spectrum, boxsize)
        spectrum = crop(spectrum, args.boxsize)
        zs = None
    else:
        pixelsize = args.pixelsize
        spectrum = amplitudeSpectrum(micrograph)
        spectrum = resize(spectrum, args.boxsize)
        zs = None

    return spectrum / cp.std(spectrum), zs, pixelsize

def prepare():
    global args, spectrum, da, a, b, psi, df0, df1
    global debug, minf, maxf, maxiter, vppdata

    # Load micrograph and compute amplitude spectrum.
    micrograph = readMicrograph(args.micrograph)
    spectrum, dz, pixelsize = computeAmplitudeSpectrum(micrograph)
    assert spectrum.dtype == cp.float32 and spectrum.flags['OWNDATA'] and spectrum.flags['C_CONTIGUOUS']

    # Config debug.
    debug = args.debug
    if debug:
        os.makedirs('ShiCTF_Debug', exist_ok = True)
        assert os.path.isdir('ShiCTF_Debug')
        mrcfile.write('ShiCTF_Debug/spectrum.mrc', spectrum.get(), True)

    # Setup parameters.
    voltage = args.voltage * 1e3
    cs = args.cs * 1e7
    amplitude = args.amplitude
    waveLength = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6))
    a = 2 * pi * waveLength / pixelsize ** 2
    da = a * dz if dz is not None else None
    b = -pi * cs * waveLength ** 3 / pixelsize ** 4
    psi = 2 * asin(amplitude) - 0.5 * pi
    minf = pixelsize / args.minres
    maxf = pixelsize / args.maxres
    maxiter = args.maxiter
    vppdata = args.vppdata

def estimate():
    '''
    Estimate: df0, df1, ast, phaseshift
    Equivalen to estimate: a0, a1, ast, phi
    Minimize: | spectrum - amp(f) * sin(a * f ** 2 + b * f ** 4 + phi) |^2
    '''
    global spectrum, da, a, b, psi, df0, df1, ast, phi, phaseshift, cc
    global debug, minf, maxf, maxiter

    n = spectrum.shape[1]
    nBeta = 128
    nTheta = n // 2
    a0 = a * df0
    a1 = a * df1
    ast = 0.
    phi = psi
    phaseshift = 0.
    cc = -inf
    dfTol = 10.
    angTol = 1e-2

    print('|-----------------------------------------------------------------------------------------------------|')
    print('| round |    defocus1 (A)    |    defocus2 (A)    |  astigmatism (deg) |  phaseshift (deg)  |  score  |')
    print('|-----------------------------------------------------------------------------------------------------|')
    print(f'|{0:^7d}|{df0:^20.1f}|{df1:^20.1f}|{degrees(ast):^20.1f}|{degrees(phaseshift):^20.1f}|{cc:^9.2f}|')
    for rd in range(maxiter):
        # Sampling in grid (beta, theta).
        theta0 = (a0 + a1) / 2 * minf ** 2 + b * minf ** 4 + phi
        theta1 = (a0 + a1) / 2 * maxf ** 2 + b * maxf ** 4 + phi
        m = ceil((theta1 - theta0) / (2 * pi))
        theta1 = theta0 + 2 * pi * m
        if da is None:
            polargram = sample(spectrum, nBeta, nTheta, theta0, theta1, a0, a1, ast, b, phi)
        else:
            polargram = equiphaseAverage(spectrum, da, nBeta, nTheta, theta0, theta1, minf, a0, a1, ast, b, phi)
        if debug:
            mrcfile.write(f'ShiCTF_Debug/it{rd:03d}_polargram.mrc', polargram.get(), True)
        if m < 4:
            print('|-----------------------------------------------------------------------------------------------------|')
            print('Oops... estimated defocus values are too small to extract 2D-IMF and detect Thon rings!')
            print('Please check if this micrograph has tiny defocus, or adjust initial defocus / minres / maxres.')
            cc = -inf
            break

        # Lowpass in beta and extractImf in theta.
        polargram2 = lowpassBeta(polargram)
        imf, u, v = extractImf(polargram2, m)
        signal = polargram - polargram2 + imf
        amp = cp.hypot(u, v)
        theta = cp.remainder(cp.linspace(0, 2 * pi * m, nTheta + 1, dtype = cp.float32) + cp.arctan2(v, u), 2 * pi)
        theta = cp.unwrap(theta, axis = 1)
        theta = cp.unwrap(theta, axis = 0)
        if debug:
            mrcfile.write(f'ShiCTF_Debug/it{rd:03d}_polargram2.mrc', polargram2.get(), True)
            mrcfile.write(f'ShiCTF_Debug/it{rd:03d}_imf.mrc', imf.get(), True)
            mrcfile.write(f'ShiCTF_Debug/it{rd:03d}_amp.mrc', amp.get(), True)
            mrcfile.write(f'ShiCTF_Debug/it{rd:03d}_theta.mrc', theta.get(), True)
            mrcfile.write(f'ShiCTF_Debug/it{rd:03d}_sintheta.mrc', cp.sin(theta).get(), True)

        # Store old parameters.
        df0Old = df0
        df1Old = df1
        astOld = ast

        # Fit new parameters.
        lft = round(2 / m * nTheta)
        A, B, f2 = normalEquation(theta, amp, lft, theta0, theta1, maxf, a0, a1, ast, b, phi)
        try:
            S = cp.linalg.solve(A, B).get()
            assert not np.any(np.isnan(S))
        except:
            print('|-----------------------------------------------------------------------------------------------------|')
            print('Oops... Something wrong happened in fitting parameters!')
            print('Please check if this micrograph has tiny defocus, or adjust initial defocus / minres / maxres.')
            break
        D = hypot(S[1], S[2])
        a0 = S[0] + D
        a1 = S[0] - D
        ast = atan2(S[2], S[1]) / 2 % pi
        ast = ast if ast <= 0.5 * pi else ast - pi
        phi = S[3] % (2 * pi) if vppdata else psi
        df0 = a0 / a
        df1 = a1 / a
        phaseshift = (phi - psi) / 2 % pi
        cc = computeCC(f2, signal, lft, maxf, a0, a1, ast, b, phi)
        print(f'|{rd + 1:^7d}|{df0:^20.1f}|{df1:^20.1f}|{degrees(ast):^20.1f}|{degrees(phaseshift):^20.1f}|{cc:^9.3f}|')

        if abs(df0Old - df0) < dfTol and abs(df1Old - df1) < dfTol and 1 - cos(2 * (ast - astOld)) < angTol:
            print('|-----------------------------------------------------------------------------------------------------|')
            print('Converged!')
            break

    else:
        print('|-----------------------------------------------------------------------------------------------------|')
        print('Not converged! But the results could be reliable if the score is high.')

    print()

def compute():
    global df0, df1, ast, phaseshift, cc
    ccMax, paras = -inf, None
    dfs = [float(df) * 10000 for df in args.defocus.split(',')]
    for df in dfs:
        print(f'Estimate CTF parameters with initial defocus {df:.1f}A.')
        df0 = df1 = df
        estimate()
        if ccMax < cc:
            ccMax = cc
            paras = (df0, df1, ast, phaseshift)
    print('Final results:')
    print('    defocus1 (A)         defocus2 (A)       astigmatism (deg)    phaseshift (deg)     score   ')
    print(f'{paras[0]:^20.1f} {paras[1]:^20.1f} {degrees(paras[2]):^20.1f} {degrees(paras[3]):^20.1f} {ccMax:^9.3f}')

def main():
    parse()
    prepare()
    compute()
