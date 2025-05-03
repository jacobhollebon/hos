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


### Decoders to render HOS to binaural or loudspeaker signals ###

import numpy as np
import warnings
from scipy.signal import oaconvolve
import hos.toolboxes.spherical as spherical
import hos.decoders as decoders


def hos2BinauralDecoder(
    hos_nm,
    hos_hrir_nm=None,
    sofaPath=None,
    kind="realsn3d",
    beta=1e-15,
    usemagls=False,
    maglsArgs={},
):
    """
    Render a hos signal to binaural

    Note the traditional hos representation uses a basis of functions made from  a sin or cos power expansion
    hos may also work with a spherical harmonic expansion
    if hos_nm and hos_hrir_nm are both supplied, the basis expansion for each should match
    if sofaPath is supplied, then hos_nm should be using the spherical harmonic represnetation

    The order of the rendering is inferred from the shape of the input soundfield and must match
    hos_hrir_nm (if supplied)

    hrir coefficients are either directly supplied (hos_hrir_nm) or calculated from a sofafile

    Parameters
    ----------
    hos_nm : Array-like, shape(N+1), time/freq)
        The hos soundfield coefficients
    hos_hrir_nm : Array-like, shape(N+1), 2, time/freq),  optional
        The hos hrir coefficients
    sofaPath : str, optional
        File path to the sofa file
        if hos_hrir_nm is supplied, this is ignored
    kind : str, optional
        The type of spherical harmonic for calculating hos_hrir_nm, options are
        complex: complex SHs
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation
    beta : float, optional
        Tikhonov regularisation value for the iSHT psueodinversion for calculating hos_hrir_nm
    usemagls : bool, optional
        WHhether to use magls when for calculating hos_hrir_nm
    maglsArgs : dict, optional
        Dictionairy of function arguments to pass to the magls calculation
        when calculating hos_hrir_nm
        see spherical.sofa2shmagls for more information


    Returns
    -------
    binaural : Array-like, shape(2, time/freq)
        The binaural signals
        The domain (freq or time) matches that of the input
        soundfield coefficients

    """

    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = spherical._getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    if hos_nm.ndim != 2:
        raise ValueError(
            "Data input should have 2 dimensions, coefficients x (time/freq)"
        )
    if (hos_hrir_nm is None) and (sofaPath is None):
        raise ValueError("One of hos_hrir_nm or sofaPath must be supplied")

    # Infer the domain of the supplied hos signals
    if np.sum(np.imag(hos_nm)):
        domain = "time"
    else:
        domain = "freq"

    # Order of the rendering is inferred from the soundfield input
    N = int(hos_nm.shape[0] - 1)

    # Safety checks of the supplied hrir
    if hos_hrir_nm is not None:
        N_hrir = int(hos_hrir_nm.shape[0] - 1)
        if N != N_hrir:
            raise ValueError(
                "The supplied hos signals have a different order ({N} to the supplied hrtf coefficients {N_hrir}"
            )
        if hos_hrir_nm.ndim != 3:
            raise ValueError(
                "HRIR input should have 3 dimensions, coefficients x 2 (ears) x (time/freq)"
            )
        if not np.sum(np.imag(hos_hrir_nm)):
            warnings.warn(
                "You have supplied a complex valued hos_hrir_nm, this should be time domain data... performing an irfft"
            )
            hos_hrir_nm = np.fft.irfft(hos_hrir_nm, axis=-1)

    # Generate the hrir coefficients
    if hos_hrir_nm is None:
        if usemagls:
            hoa_hrir_nm = spherical.sofa2shmagls(
                sofaPath, N=N, kind=kind, beta=beta, **maglsArgs
            )
        else:
            hoa_hrir_nm = spherical.sofa2sh(sofaPath, N=N, kind=kind, beta=beta)
        hos_hrir_nm = decoders.hoa2hosDecoder(hoa_hrir_nm, kind=kind)

    # Render the binaural signals
    if domain == "freq":
        hos_nm = np.fft.irfft(hos_nm)
    binaural = np.zeros((2, hos_nm.shape[-1] + hos_hrir_nm.shape[-1] - 1))
    for ear in range(2):
        binaural[ear, :] = oaconvolve(hos_nm, hos_hrir_nm[:, ear, :], mode="full")
    if domain == "freq":
        binaural = np.fft.rfft(binaural, axis=-1)

    return binaural
