# -*- coding: utf-8 -*-

# Copyright (C) 2018-2024 Jacob Hollebon
# Copyright (C) 2018-2023 University of Southampton

# Higher-Order Stereophony
# Author: Jacob Hollebon
# Project page: https://github.com/jacobhollebon/hos

# This code is provided under the ISC (Internet Systems Consortium) license
# https://www.isc.org/downloads/software-support-policy/isc-license/ :

# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


# We kindly ask to acknowledge the use of this software in publications or software.
# Paper citation:
# Jacob Hollebon and Filippo Maria Fazi,
# “Higher-order stereophony”
# IEEE/ACM Transactions on Audio, Speech, and Language Processing,
# vol. 31, pp. 2872–2885, 2023
# doi: 10.1109/TASLP.2023.3297953.

# This implementation uses the VISR framework. Information about the VISR,
# including download, setup and usage instructions, can be found on the VISR project page
# http://cvssp.org/data/s3a/public/VISR .


### Functions for performing ciruclar harmonic related operations and transforms ###


import numpy as np
import matplotlib.pyplot as plt


def circHarm(az, N, kind="real", plot=False):
    """
    Calculate a matrix of circular harmonics up to order N
    sampled at azimuthal positions in `pos`.

    Parameters
    ----------
    az : array-like, shape (Q,)
        Q azimuthal positions in radians (0 to 2*pi).
    N : int
        Maximum order of circular harmonics.
    kind : str
        'real' or 'exp' harmonic basis.
    plot : bool
        If true, plot the sampling positions.

    Returns
    -------
    Y : ndarray, shape (Q, 2N+1)
        Circular harmonic matrix
    """
    kind = kind.lower()
    if kind not in ["real", "exp"]:
        raise ValueError("Invalid kind: Choose 'real' or 'exp'")

    az = np.asarray(az)
    if az.ndim != 1:
        raise ValueError(f"Expected 1D input for `az`, got shape {az.shape}")

    if kind == "real":
        Y = circHarmReal(az, N)
    else:
        Y = circHarmExp(az, N)

    if plot:
        plt.figure()
        plt.polar(az, np.ones_like(az), "o")
        plt.title("Circular Harmonic Sampling Positions")

    return Y


def circHarmReal(az, N):
    """Real-valued circular harmonics up to order N"""
    Q = len(az)
    Y = np.zeros((Q, 2 * N + 1), dtype=float)

    # Order: [0, -1, 1, -2, 2, ..., -N, N]
    channels = [0] + [i * (-1) ** (i + 1) for i in range(1, N + 1) for _ in (0, 1)]

    norm = 1 / np.sqrt(2 * np.pi)
    for i, n in enumerate(channels):
        if n == 0:
            Y[:, i] = 1
        elif n < 0:
            Y[:, i] = np.sqrt(2) * np.sin(abs(n) * az)
        else:
            Y[:, i] = np.sqrt(2) * np.cos(n * az)
    return norm * Y


def circHarmExp(az, N):
    """Complex exponential circular harmonics up to order N"""
    Q = len(az)
    Y = np.zeros((Q, 2 * N + 1), dtype=complex)

    # Order: [0, -1, 1, -2, 2, ..., -N, N]
    channels = [0] + [i * (-1) ** (i + 1) for i in range(1, N + 1) for _ in (0, 1)]

    for i, n in enumerate(channels):
        Y[:, i] = np.exp(-1j * n * az)
    return Y


def iCHT(data, az, N, kind="real", beta=1e-15):
    """
    Inverse Circular Harmonic Transform

    Projects sampled data at azimuthal positions into the circular harmonic domain.

    Parameters
    ----------
    data : ndarray, shape (Q, time/freq) or (Q, chs, time/freq)
        Input data sampled at Q azimuthal positions.
    az : ndarray, shape (Q,)
        Azimuthal angles in radians [0, 2pi].
    N : int
        Maximum harmonic order.
    kind : str
        'real' or 'exp'.
    beta : float
        Tikhonov regularization value.

    Returns
    -------
    Y : ndarray, shape (Q, 2N+1)
        Circular harmonic matrix.
    Yinv : ndarray, shape (2N+1, Q)
        Regularized pseudoinverse of Y.
    data_nm : ndarray, shape (2N+1, time/freq) or (2N+1, chs, time/freq)
        Harmonic coefficients.
    """
    kind = kind.lower()
    if kind not in ["real", "exp"]:
        raise ValueError("Invalid kind: Choose 'real' or 'exp'")

    data = np.asarray(data)
    az = np.asarray(az)

    if data.ndim not in (2, 3):
        raise ValueError("Data should have 2 or 3 dimensions")

    Q = len(az)
    Y = circHarm(az, N, kind=kind, plot=False)
    numCoeffs = 2 * N + 1

    YH = Y.conj().T
    gram = np.linalg.inv(YH @ Y + beta * np.identity(numCoeffs))
    Yinv = gram @ YH

    if data.ndim == 2:
        data_nm = Yinv @ data
    else:
        dtype = (Yinv @ data[:, 0, :]).dtype
        data_nm = np.zeros((numCoeffs, data.shape[1], data.shape[2]), dtype=dtype)
        for ch in range(data.shape[1]):
            data_nm[:, ch, :] = Yinv @ data[:, ch, :]

    return Y, Yinv, data_nm


def CHT(data_nm, az, N, kind="real"):
    """
    Circular Harmonic Transform

    Synthesizes spatial data from circular harmonic coefficients.

    Parameters
    ----------
    data_nm : ndarray, shape (2N+1, time/freq) or (2N+1, chs, time/freq)
        Harmonic domain data.
    az : ndarray, shape (Q,)
        Azimuthal angles (radians).
    N : int
        Maximum order.
    kind : str
        'real' or 'exp'.

    Returns
    -------
    Y : ndarray, shape (Q, 2N+1)
        Circular harmonic matrix.
    data : ndarray, shape (Q, time/freq) or (Q, chs, time/freq)
        Synthesized spatial data.
    """
    kind = kind.lower()
    if kind not in ["real", "exp"]:
        raise ValueError("Invalid kind: Choose 'real' or 'exp'")

    data_nm = np.asarray(data_nm)
    az = np.asarray(az)

    if data_nm.ndim not in (2, 3):
        raise ValueError("data_nm should have 2 or 3 dimensions")

    Q = len(az)
    Y = circHarm(az, N, kind=kind, plot=False)

    if data_nm.ndim == 2:
        data = Y @ data_nm
    else:
        dtype = (Y @ data_nm[:, 0, :]).dtype
        data = np.zeros((Q, data_nm.shape[1], data_nm.shape[2]), dtype=dtype)
        for ch in range(data.shape[1]):
            data[:, ch, :] = Y @ data_nm[:, ch, :]

    return Y, data
