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


### Functions for performing spherical harmonic related operations and transforms ###
### Utilises the spharpy toolbox for core operations ###


import numpy as np
import matplotlib.pyplot as plt
import spharpy
import sofar
from scipy.signal import group_delay
from statistics import median


def _getSupportedSHConventions():
    """
    Helper function to return the supported types of
    spherical harmonics within this toolbox

    Parameters
    ----------
    None

    Returns
    -------
    supportedSHConventions : str
        List of the supported types of spherical harmonics

    """
    supportedSHConventions = ["complex", "realn3d", "realsn3d"]
    return supportedSHConventions


def nmACN(N):
    """
    Return set of order (n) and degree (m) indices and ACN channels up to
    a supplied truncation order N

    Parameters
    ----------
    N : int
        Order of the sampling.

    Returns
    -------
    n : Array-like , shape (N+1**2)
        Array of the order index for each channel
    m : Array-like , shape (N+1**2)
        Array of the degree index for each channel
    ACN : Array-like , shape (N+1**2)
        Array of the ACN channel index for each channel

    """
    n_list = []
    m_list = []
    for n in range(N + 1):
        for m in range(-n, n + 1):
            n_list.append(n)
            m_list.append(m)
    n = np.array(n_list)
    m = np.array(m_list)
    ACN = n**2 + n + m
    return n, m, ACN


def Nkr(N, r=0.0875, c=343):
    """
    Calculate the frequency a given order N ensures accurate reproduction
    up to for a given head/sphere radius


    Parameters
    ----------
    N : int
        Order of the sampling
    r : float, optional
        Radius of the sphere of reproduction in m
    c : float, optional
        Speed of sounds

    Returns
    -------
    None

    """
    f = ((N / r) * c) / (2 * np.pi)
    print(f"N=kr frequency is {round(f)} Hz")
    return f


def Nkr_f(N, f, c=343):
    """
    Calculate the radius a given order N ensures accurate reproduction
    up to for a given frequency

    Parameters
    ----------
    N : int
        Order of the sampling
    f : float
        Frequency of interest in Hz
    c : float, optional
        Speed of sound

    Returns
    -------
    None
    """
    k = (2 * np.pi * f) / c
    r = N / k
    print(f"N=kr radius is {round(r, 2)} m")
    return r


def sphHarm(pos, N, kind="realSN3D", plot=False):
    """
    Calculate a matrix of spherical harmonics up to order N
    sampled at pos

    For definitions of the various spherical harmonics see
    https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization
    https://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#Normalisation

    Parameters
    ----------
    pos : Array-like, shape(Q, 3)
        Q x spherical sampling positions with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
    N : int
        Order of the sampling
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation
    plot : bool, optional
        If true the sampling positions are plotted

    Returns
    -------
    Ynm : Array-like, shape(Q, (N+1)**2)
        Matrix of spherical harmonics up to order N sampled at the Q positions

    """

    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = _getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")

    # Utilise the spharpy coordinate convention
    # even though the documentation says elevation (+90 to -90), the underlying sph2cart function requires colatitude (0-180)
    az = pos[:, 0]
    el = pos[:, 1]
    co = (el - np.pi / 2) % np.pi
    r = pos[:, 2]
    coords = spharpy.samplings.Coordinates()
    coords = coords.from_spherical(r, co, az)
    # Plot the sampling
    if plot:
        plt.figure()
        spharpy.plot.scatter(coords)
        plt.title("Sampling Positions")
        plt.figure()
        plt.plot(np.rad2deg(az), ls="-", label="Az, Deg")
        plt.plot(np.rad2deg(el), ls="--", label="El, Deg")
        plt.plot(np.rad2deg(co), ls="-.", label="Co, Deg")
        plt.xlabel("Sampling Pos Index")
        plt.ylabel("Angle, Deg")
        plt.legend()
        plt.grid("on")
        plt.title("Angular Sampling Positions")
        plt.tight_layout()

    # Build the SHs
    if kind == "complex":
        Ynm = spharpy.spherical.spherical_harmonic_basis(N, coords)
        return Ynm
    elif kind == "realn3d":
        Ynm = spharpy.spherical.spherical_harmonic_basis_real(N, coords)
        return Ynm
    elif kind == "realsn3d":
        Ynm = spharpy.spherical.spherical_harmonic_basis_real(N, coords)
        # Convert N3D to SN3D
        n = np.repeat(np.arange(N + 1), np.arange(N + 1) * 2 + 1)
        normFactor = 1 / np.sqrt(2 * n + 1)
        Ynm /= normFactor
        return Ynm


def iSHT(data, pos, N, kind="realsn3d", beta=1e-15):
    """
    Calculate the inverse spherical harmonic transform of a set of data

    Transform to the spherical harmonic domain

    The transform is performed using the regularised pseudoinverse of the
    SH matrix


    Parameters
    ----------
    data : Array-like, shape (Q, time/freq) or (Q, chs, time/freq)
        The data to undergo the inverse spherical harmonic transform
        May be supplied in the time or frequency domain
    pos : Array-like, shape(Q, 3)
        Q x spherical sampling positions with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
    N : int
        Order of the sampling
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation
    beta : float, optional
        Tikhonov regularisation value

    Returns
    -------
    Ynm : Array-like, shape(Q, (N+1)**2)
        Matrix of spherical harmonics up to order N sampled at the Q positions
    YnmInv : Array-like, shape((N+1)**2, Q)
        Regularised pseudoinversion of the spherical harmonics up to order N sampled at the Q positions
    data_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The data in the spherical harmonic domain
        The spherical harmonic coefficients which represent the supplied data

    """

    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = _getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    # check the dimensions of the supplied data
    dims = data.ndim
    if dims > 3:
        raise ValueError("Data larger than 3 dimensions is not supported")
    elif dims < 2:
        raise ValueError(
            "Data should have at minimum 2 dimensions, positions x (time/freq)"
        )
    
    if beta is None:
        beta = 0
    
    # Compute the SH matrix
    Ynm = sphHarm(pos, N, kind=kind)
    numCoeffs = (N + 1) ** 2  # number of SHs
    Q = pos.shape[0]  # number of sampling positions

    # Regularised pseuodinversion of the SH matrix
    YnmH = Ynm.conj().T
    grammian = np.linalg.inv((YnmH @ Ynm) + beta * np.identity(numCoeffs))
    YnmInv = grammian @ YnmH

    # Perform the iSHT
    if dims == 2:
        data_nm = YnmInv @ data
    elif dims == 3:
        dtype = (YnmInv @ data[:, 0, :]).dtype
        data_nm = np.zeros((numCoeffs, data.shape[1], data.shape[2]), dtype=dtype)
        for ch in range(data.shape[1]):
            data_nm[:, ch, :] = YnmInv @ data[:, ch, :]

    return Ynm, YnmInv, data_nm


def iSHTmagls(data,
                pos,
                N,
                kind="realsn3d",
                beta=1e-15,
                fmagls=None,
                fade=None,
                removeGD=False,
                fs=48000,
                NFFT=None,
                ):
    """
    Calculate the inverse spherical harmonic transform of a set of data

    Transform to the spherical harmonic domain

    The transform is performed using the regularised pseudoinverse of the
    SH matrix

    Above a certain frequency the magnitude least-squares algorithm is applied
    which priorities the reproduction of the magntiude at the cost of an erranous
    phase representations

    ### TODO
    Add delay management
    add NFFT arg for high f resoltuion when designing

    Parameters
    ----------
    data : Array-like, shape (Q, time/freq) or (Q, chs, time/freq)
        The data to undergo the inverse spherical harmonic transform
        May be supplied in the time or frequency domain
        If all data is real assumed to be time domain
    pos : Array-like, shape(Q, 3)
        Q x spherical sampling positions with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
    N : int
        Order of the sampling
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation
    beta : float, optional
        Tikhonov regularisation value
    fmagls : None or float, optional
        Frequency in Hz above which to apply magls
        If left as None, the N=kr frequency is used
    fade : None or float, optional
        If a float is supplied the LS and magLS solutions are faded between
        over this many Hz starting from frequency fmagls Hz
    removeGD : Bool, optional
        If True, the group delay is estimated by synthesising the omni component
        and removed on an ear-by-ear basis
        This may help with the phase reconstruction
    fs : float, optional
        Sampling rate of the data in Hz
    NFFT : None or float, optional
        Value used for the fft dictating the frequency resolution
        If None then the length of the time series (hrir in) or
        current NFFT value (hrtf in) is used
        May not be smaller either of this two values


    Returns
    -------
    Ynm : Array-like, shape(Q, (N+1)**2)
        Matrix of spherical harmonics up to order N sampled at the Q positions
    YnmInv : Array-like, shape((N+1)**2, Q)
        Regularised pseudoinversion of the spherical harmonics up to order N sampled at the Q positions
    data_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The data in the spherical harmonic domain
        The spherical harmonic coefficients which represent the supplied data
        With magls applied

    """

    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = _getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    # check the dimensions of the supplied data
    dims = data.ndim
    if dims > 3:
        raise ValueError("Data larger than 3 dimensions is not supported")
    elif dims < 2:
        raise ValueError(
            "Data should have at minimum 2 dimensions, positions x (time/freq)"
        )

    # Test if the data is in time or freq domain
    if not np.sum(np.imag(data)):
        isReal = True
        NFFTin = data.shape[-1]
    else:
        isReal = False
        NFFTin = (data.shape[-1] - 1) * 2
    if NFFT is None:
        NFFT = NFFTin
    else:
        if NFFT < NFFTin:
            if isReal:
                err = f"The NFFT supplied ({NFFT}) is smaller than the time series of the  supplied HRIR s({NFFTin} samples)"
            else:
                err = f"The NFFT supplied ({NFFT}) is smaller than the current NFFT of the supplied HRTFs ({NFFTin})"
            raise ValueError(err)

    # Set up frequency variables and ensure data is in frequency domain
    if isReal:
        data = np.fft.rfft(data, NFFT, axis=-1)  # magls requires freq domain
    else:
        if NFFT != NFFTin:
            data = np.fft.irfft(data, NFFTin, axis=-1)  # to time domain
            data = np.fft.rfft(data, NFFT, axis=-1)  # back to freq domain with new NFFT
    f = np.linspace(0, fs / 2, (NFFT // 2) + 1)
    f_len = len(f)
    
    if beta is None:
        beta = 0
        
    # Set up mag ls
    if fmagls is None:
        fmagls = Nkr(N)
    magls_start = np.argmin(np.abs(f - fmagls))  # apply magls above this freq index
    if fade is not None:
        magls_stop = np.argmin(np.abs(f - fmagls - fade))
        fade_len = magls_stop - magls_start
        if not (fade_len % 2):
            fade_len += 1  # ensure the window is odd to get a value of 1 sampled
        fade = np.hanning(fade_len)
        fade_in = fade[: (fade_len // 2) + 1]
        fade_out = fade[(fade_len // 2) + 1 :]
    if magls_start > f_len:
        raise ValueError(
            f"The requested magls start frequency {fmagls} Hz is larger than the nyquist frequencys {fs/2}!"
        )

    # Compute the SH matrix
    Ynm = sphHarm(pos, N, kind=kind)
    numCoeffs = (N + 1) ** 2  # number of SHs
    Q = pos.shape[0]  # number of sampling positions

    # Regularised pseuodinversion of the SH matrix
    YnmH = Ynm.conj().T
    grammian = np.linalg.inv((YnmH @ Ynm) + beta * np.identity(numCoeffs))
    YnmInv = grammian @ YnmH

    # Remove any group delay
    if removeGD:
        # Remove any delay from the HRIRs. This helps with the phase reconstrution in the magls section
        # Synthesise the W SH channel for each ear, find the group delay based on this omni response
        # Take the overall delay as the median of the group delay (should be consistent under diff Nfft sizes)
        # Then rmove the delay
        data = np.fft.irfft(data, NFFT, axis=-1)  # needs time domain data
        if dims == 2:
            W = YnmInv[0, :] @ data
            _, gd = group_delay((W, 1), w=f, fs=fs)
            data = np.roll(data, -round(median(gd)), axis=-1)
        elif dims == 3:
            for ch in range(data.shape[1]):
                W = YnmInv[0, :] @ data[:, ch, :]
                _, gd = group_delay((W, 1), w=f, fs=fs)
                data[:, ch, :] = np.roll(data[:, ch, :], -round(median(gd)), axis=-1)
        data = np.fft.rfft(data, NFFT, axis=-1)

    # Perform the iSHT
    if dims == 2:
        data_nm = YnmInv @ data
    elif dims == 3:
        dtype = (YnmInv[0, :] @ data[:, 0, :]).dtype
        data_nm = np.zeros((numCoeffs, data.shape[1], data.shape[2]), dtype=dtype)
        for ch in range(data.shape[1]):
            data_nm[:, ch, :] = YnmInv @ data[:, ch, :]

    # Peform mag ls
    data_nm_magls = data_nm.copy()
    for findx in range(magls_start, f_len):
        if dims == 2:
            phi = np.angle(Ynm @ data_nm[:, findx - 1])
            data_nm_magls[:, findx] = YnmInv @ (
                np.abs(data[:, findx]) * np.exp(1j * phi)
            )
        elif dims == 3:
            for ch in range(data.shape[1]):
                phi = np.angle(Ynm @ data_nm[:, ch, findx - 1])
                data_nm_magls[:, ch, findx] = YnmInv @ (
                    np.abs(data[:, ch, findx]) * np.exp(1j * phi)
                )

    # Join together to create the output
    data_nm_out = data_nm.copy()
    data_nm_out[..., magls_start:] = data_nm_magls[..., magls_start:]
    if fade is not None:
        data_out = data_nm[..., magls_start : magls_start + fade_len] * fade_out
        data_in = data_nm_magls[..., magls_start : magls_start + fade_len] * fade_in
        data_nm_out[..., magls_start : magls_start + fade_len] = data_out + data_in

    if isReal:
        data_nm_out = np.fft.irfft(data_nm_out, NFFT, axis=-1)  # return to time domain

    return Ynm, YnmInv, data_nm_out


def SHT(data_nm, pos, N, kind="realsn3d"):
    """
    Calculate the spherical harmonic transform using a supplied set of spherical harmonic coefficients

    Transform from the spherical harmonic domain to the set of requested positions


    Parameters
    ----------
    data_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The spherical harmonic coefficients
    pos : Array-like, shape(Q, 3)
        Q x spherical resynthesis positions with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
    N : int
        Order of the sampling
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation

    Returns
    -------
    Ynm : Array-like, shape(Q, (N+1)**2)
        Matrix of spherical harmonics up to order N sampled at the Q positions
    data_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The data in the spherical harmonic domain

    """

    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = _getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    # check the dimensions of the supplied data
    dims = data_nm.ndim
    if dims > 3:
        raise ValueError("Data larger than 3 dimensions is not supported")
    elif dims < 2:
        raise ValueError(
            "Data should have at minimum 2 dimensions, coefficients x (time/freq)"
        )

    # Compute the SH matrix
    Ynm = sphHarm(pos, N, kind=kind)
    numCoeffs = (N + 1) ** 2  # number of SHs
    Q = pos.shape[0]  # number of sampling positions

    # Perform the SHT
    if dims == 2:
        data = Ynm @ data_nm
    elif dims == 3:
        dtype = (Ynm @ data_nm[:, 0, :]).dtype
        data = np.zeros((Q, data_nm.shape[1], data_nm.shape[2]), dtype=dtype)
        for indx in range(data.shape[1]):
            data[:, indx, :] = Ynm @ data_nm[:, indx, :]

    return Ynm, data


def sofa2sh(sofaPath, N=None, kind="realsn3d", beta=1e-15):
    """
    Load a sofa file and perform the iSHT returning the
    hrir SH coefficients


    Parameters
    ----------
    sofaPath : str
        File path to the sofa file
    N : int or None
        Order of the sampling
        If None, the highest maximum order as per the number of
        sampling positions in the sofa file will be used
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation
    beta : float, optional
        Tikhonov regularisation value for the iSHT psueodinversion

    Returns
    -------
    Ynm : Array-like, shape(Q, (N+1)**2)
        Matrix of spherical harmonics up to order N sampled at the Q positions
    YnmInv : Array-like, shape((N+1)**2, Q)
        Regularised pseudoinversion of the spherical harmonics up to order N sampled at the Q positions
    hrir_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The hrir spherical harmonic coefficients
    pos : Array-like, shape(Q, 3)
        Q x spherical sampling positions of the hrir with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
    fs : float
        Sampling frequency of the hrir in Hz

    """

    # Read in the sofa file and extract the relevation data
    file = sofar.read_sofa(sofaPath)
    hrir = file.Data_IR
    fs = file.Data_SamplingRate
    pos = file.SourcePosition  # spherical positions [az,el,r] in [deg,deg,m]

    az = np.deg2rad(pos[:, 0])
    el = np.deg2rad(pos[:, 1])
    r = pos[:, 2]
    # new sampling position array
    pos = np.stack([az, el, r], axis=1)

    if beta is None:
        beta = 0
        
    Nmax = int(np.floor(np.sqrt(pos.shape[0]) - 1))
    if N is None:
        print(f"No truncation order supplied, using Nmax={Nmax}")
        N = Nmax
    elif N > Nmax:
        print(
            "You have requested an order higher than possible with this sampling regime"
        )
        print(f"Truncating to maximum order allowed: {Nmax}")
        N = Nmax

    Ynm, YnmInv, hrir_nm = iSHT(hrir, pos, N, kind=kind, beta=beta)

    return Ynm, YnmInv, hrir_nm, pos, fs


def sofa2shmagls(
    sofaPath,
    N=None,
    kind="realsn3d",
    beta=1e-15,
    fmagls=None,
    fade=None,
    removeGD=False,
    NFFT=None,
):
    """
    Load a sofa file and perform the iSHT returning the
    hrir SH coefficients


    Parameters
    ----------
    sofaPath : str
        File path to the sofa file
    N : int or None
        Order of the sampling
        If None, the highest maximum order as per the number of
        sampling positions in the sofa file will be used
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation
    beta : float, optional
        Tikhonov regularisation value for the iSHT psueodinversion
    fmagls : None or float, optional
        Frequency in Hz above which to apply magls
        If left as None, the N=kr frequency is used
    fade : None or float, optional
        If a float is supplied the LS and magLS solutions are faded between
        over this many Hz starting from frequency fmagls Hz
    removeGD : Bool, optional
        If True, the group delay is estimated by synthesising the omni component
        and removed on an ear-by-ear basis
        This may help with the phase reconstruction
    NFFT : None or float, optional
        Value used for the fft dictating the frequency resolution
        If None then the length of the time series (hrir in) or
        current NFFT value (hrtf in) is used
        May not be smaller either of this two values


    Returns
    -------
    Ynm : Array-like, shape(Q, (N+1)**2)
        Matrix of spherical harmonics up to order N sampled at the Q positions
    YnmInv : Array-like, shape((N+1)**2, Q)
        Regularised pseudoinversion of the spherical harmonics up to order N sampled at the Q positions
    hrir_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The hrir spherical harmonic coefficients
    pos : Array-like, shape(Q, 3)
        Q x spherical sampling positions of the hrir with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
    fs : float
        Sampling frequency of the hrir in Hz

    """

    # Read in the sofa file and extract the relevation data
    file = sofar.read_sofa(sofaPath)
    hrir = file.Data_IR
    fs = file.Data_SamplingRate
    pos = file.SourcePosition  # spherical positions [az,el,r] in [deg,deg,m]

    az = np.deg2rad(pos[:, 0])
    el = np.deg2rad(pos[:, 1])
    r = pos[:, 2]
    # new sampling position array
    pos = np.stack([az, el, r], axis=1)

    if beta is None:
        beta = 0
        
    Nmax = int(np.floor(np.sqrt(pos.shape[0]) - 1))
    if N is None:
        print(f"No truncation order supplied, using Nmax={Nmax}")
        N = Nmax
    elif N > Nmax:
        print(
            "You have requested an order higher than possible with this sampling regime"
        )
        print(f"Truncating to maximum order allowed: {Nmax}")
        N = Nmax

    Ynm, YnmInv, hrir_nm = iSHTmagls(
        hrir, pos, N, kind=kind, beta=beta, fmagls=fmagls, fade=fade, fs=fs, NFFT=NFFT
    )

    return Ynm, YnmInv, hrir_nm, pos, fs


def rotateCoefficients(data_nm, angles, seq="zyx", kind="realsn3d", isDegrees=False):
    """
    Rotate a set of spherical harmonic cofficients

    This function is a wrapper for the spharpy.transform.RotationSH class
    limiting the functionality to perform using euler angles only

    Further input types (quaternions, rotation matrices etc) can be exploited
    using this class

    Parameters
    ----------
    data_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The spherical harmonic coefficients
    angles : Array-like, shape(3)
        Euler angles of rotation as per the specified seq
        Supplied in radians (isDegrees=False) or degrees (isDegrees=True)
    seq : str
        Specifies sequence of axes for rotations. 3 characters
        belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations,
        or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and intrinsic
        rotations cannot be mixed in one function call.
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs
        real: real SHs (realsn3d or realn3d are also accepted as normalisation is not needed for the rotations)
    isDegrees : bool, optional
        If True, the supplied ypr euler angles are assumed to be supplied in degrees, not in radians

    Returns
    -------
    data_nm_rot : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The rotated spherical harmonic coefficients.

    """

    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = _getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    # check the dimensions of the supplied data
    dims = data_nm.ndim
    if dims > 3:
        raise ValueError("Data larger than 3 dimensions is not supported")
    elif dims < 2:
        raise ValueError(
            "Data should have at minimum 2 dimensions, coefficients x (time/freq)"
        )

    if kind in ["complex"]:
        kind = "complex"
    elif kind in ["real", "realn3d", "realsn3d"]:
        kind = "real"

    N = int(np.sqrt(data_nm.shape[0]) - 1)

    rotClass = spharpy.transforms.RotationSH.from_euler(
        N, seq=seq, angles=angles, degrees=isDegrees
    )

    if dims == 2:
        data_nm_rot = rotClass.apply(data_nm, kind)
    elif dims == 3:
        data_nm_rot = np.empty_like(data_nm)
        for indx in range(data_nm.shape[1]):
            data_nm_rot[:, indx, :] = rotClass.apply(data_nm[:, indx, :], kind)

    return data_nm_rot


def decimateCoefficients(data_nm):
    """
    Reduce a full set of (N+1)**2 coefficients down to the HOS representation
    of just the (N+1) set of m=0 coefficients

    Parameters
    ----------
    data_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The spherical harmonic coefficients

    Returns
    -------
    data_n : Array-like, shape(N+1), time/freq) or (N+1), ch, time/freq))
        The decimated spherical harmonic coefficients.

    """

    # check the dimensions of the supplied data
    dims = data_nm.ndim
    if dims > 3:
        raise ValueError("Data larger than 3 dimensions is not supported")
    elif dims < 2:
        raise ValueError(
            "Data should have at minimum 2 dimensions, coefficients x (time/freq)"
        )

    N = int(np.sqrt(data_nm.shape[0]) - 1)

    D = np.zeros(((N + 1), (N + 1) ** 2))
    for n in range(N + 1):
        acn = n**2 + n  # acn formula for m=0 coefficients
        D[n, acn] = 1

    if dims == 2:
        data_nm_dec = D @ data_nm
    elif dims == 3:
        data_nm_dec = np.empty_like(data_nm)
        for ch in range(data_nm.shape[1]):
            data_nm_dec[:, ch, :] = D @ data_nm[:, ch, :]

    return data_nm_dec
