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


### Decoder to render HOA signals using HOS, to loudspeakers or binaural###

import numpy as np
import warnings
import hos.toolboxes.spherical as spherical
import hos.decoders as decoders

def hoa2hosDecoder( hoa_nm, kind='realsn3d' ):
    '''
    Decode a set of hoa format signals (coefficients of a spherical harmonic 
    expansion) to hos format signals
    Decoding consists of
    1) Rotating frame of reference such that yhat is rotated to zhat
    2) Decimating (removing) all channels with m not equal to 0
    
    May be used to convert soundfield or hrtf hoa coefficients

    Parameters
    ----------
    hoa_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The hoa spherical harmonic coefficients 
    kind : str, optional
        The type of spherical harmonic, options are
        complex: complex SHs 
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation 
        
    Returns
    -------
    hos_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The hos format spherical harmonic coefficients.

    '''
    
    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = spherical._getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    # check the dimensions of the supplied data
    dims = hoa_nm.ndim
    if dims > 3:
        raise ValueError('Data input larger than 3 dimensions is not supported')
    elif dims < 2:
        raise ValueError('Data input should have at minimum 2 dimensions, coefficients x (time/freq)')
        
    # Perform the HOS rotation (y axis rotated to the z axis)
    ypr = [0, 0, np.pi/2]  # extrensic rotation ordered zyx following ypr
    seq = 'zyx'
    hos_nm = spherical.rotateCoefficients(hoa_nm.copy(), ypr, seq=seq, kind=kind)

    # Perform the decimation (removal of all channels with m not equal to 0)
    hos_nm = spherical.decimateCoefficients( hos_nm )
    
    return hos_nm


def hoa2hosBinauralDecoder( hoa_nm, hoa_hrir_nm=None, sofaPath=None, kind='realsn3d', beta=1e-15, usemagls=False, maglsArgs={}):
    '''
    Decode a set of hoa format signals (coefficients of a spherical harmonic 
    expansion) to binaural using OS rendering
    Decoding consists of 
    1) Rotating frame of reference such that yhat is rotated to zhat
    2) Decimating (removing) all channels with m not equal to 0
    3) Rendering in the SH domain the m=0 hrtf and soundfield coefficients
    
    spherical harmonic hrtf coefficients may be supplied or are calculated from
    a sofafile
    

    Parameters
    ----------
    hoa_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The hoa spherical harmonic coefficients 
    hoa_hrir_nm : Array-like, shape(N+1)**2, 2, time/freq),  optional
        The hoa hrir coefficients 
    sofaPath : str, optional
        File path to the sofa file
        if hoa_hrir_nm is supplied, this is ignored
    kind : str, optional
        The type of spherical harmonic for calculating hos_hrir_nm, options are
        complex: complex SHs 
        realn3d: real with n3d normalisation
        realsn3d: real with sn3d normalisation 
    beta : float, optional
        Tikhonov regularisation value for the iSHT psueodinversion for calculating hoa_hrir_nm
    usemagls : bool, optional
        WHhether to use magls when for calculating hoa_hrir_nm
    maglsArgs : dict, optional
        Dictionairy of function arguments to pass to the magls calculation
        when calculating hoa_hrir_nm
        see spherical.sofa2shmagls for more information
        
    Returns
    -------
    hos_nm : Array-like, shape(N+1)**2, time/freq) or (N+1)**2, ch, time/freq))
        The hos format spherical harmonic coefficients.

    '''
    
    # Safety Checks
    # check the kind of SH requested
    kind = kind.lower()
    supportedSHConventions = spherical._getSupportedSHConventions()
    if kind not in supportedSHConventions:
        raise ValueError(f"Invalid kind: Choose from {supportedSHConventions}")
    if hoa_nm.ndim != 2:
        raise ValueError('Data input should have 2 dimensions, coefficients x (time/freq)')
    if (hoa_hrir_nm is None) and (sofaPath is None):
        raise ValueError('One of hoa_hrir_nm or sofaPath must be supplied')
        
    # Order of the rendering is inferred from the soundfield input
    N = int(np.sqrt(hoa_nm.shape[0])-1)
        
    # Safety checks of the supplied hrir
    if hoa_hrir_nm is not None:
        N_hrir = int(np.sqrt(hoa_hrir_nm.shape[0])-1)
        if N != N_hrir:
            raise ValueError('The supplied hos signals have a different order ({N} to the supplied hrtf coefficients {N_hrir}')
        if hoa_hrir_nm.ndim != 3:
            raise ValueError('HRIR input should have 3 dimensions, coefficients x 2 (ears) x (time/freq)')
        if not np.sum(np.imag(hoa_hrir_nm)):
            warnings.warn('You have supplied a complex valued hoa_hrir_nm, this should be time domain data... performing an irfft')
            hoa_hrir_nm = np.fft.irfft(hoa_hrir_nm, axis=-1)
    
    # convert the soundfield coefficients to HOS
    hos_nm = decoders.hoa2hosDecoder( hoa_nm, kind=kind )
    
    # convert the hrir coefficients to HOS
    hos_hrir_nm = decoders.hoa2hosDecoder( hoa_hrir_nm, kind=kind )
    
    # render binaural using hos
    binaural = decoders.hos2BinauralDecoder( hos_nm, 
                                              hos_hrir_nm=hos_hrir_nm, 
                                              sofaPath=sofaPath, 
                                              kind=kind, 
                                              beta=beta, 
                                              usemagls=usemagls, 
                                              maglsArgs=maglsArgs
                                              )
    
    return binaural


