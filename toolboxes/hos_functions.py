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


### Functions for performing core Higher-Order Stereo operations ###

    
import numpy as np
import geometry

def _quadrantMapping( vec, hhat ):
    '''
    Returns a factor to multiply the HOSAngle by to ensure it is in the correct 
    quadrant, as the great-circle distance/dot product only gives the angle between
    vectors not the relative direction

    Parameters
    ----------
    xyz : Array-like, shape (3)
        Cartesian coordinates of a single source positions
    hhat : Array-like, shape (3)
        Unit norm cartesian vector pointing in the look direction of
        the listeners head (listener orientation)
        
    Returns
    -------
    factor : int
        1 or -1 depending on the quadrant.

    '''
    vec_rot = np.squeeze(geometry.estimateAndApplyRotation(vec, hhat))

    sph = geometry.cart2sph(vec_rot[0], vec_rot[1], vec_rot[2])
    # Quadrant of each azimuthal position as int 0,1,2,3 going clockwise w.r.t azimuth
    aziQuad = int(np.floor((np.round(sph[0],10) % (2*np.pi))/(np.pi/2)))
    
    if aziQuad in [0,1]:
        factor = 1
    elif aziQuad in [2,3]:
        factor = -1
    else:
        raise ValueError(f'Something went wrong calculating aziQuad, returned value: {aziQuad}')
    return factor

    
    
def calculateHOSAngle( xyz, hhat=np.array([1,0,0]) ):
    '''
    Calculates the HOS angle 
    This is the angle between a source at position xyz and the x axis in the listeners rotated
    frame of reference defined by hhat
    
    The HOSAngle maps any given source position to a cone of confusion of possible source positions
    which for a plane wave source all create the same soundfield across the interaural axis
    
    This means only one angle is required to define the source position and in any HOS encoding/decoding
    
    Equivalently the HOSAngle may be considered as the horizontal only source position which maps an 
    elevated position through the cone of confusion

    Parameters
    ----------
    xyz : Array-like, shape (L, 3) or (3)
        Cartesian coordinates of the source positions
        May be an array of L different coordinates or a single set
    hhat : Array-like, 3
        Unit norm cartesian vector pointing in the look direction of
        the listeners head (listener orientation)

    Returns
    -------
    angle : Array-like, shape (L)
        HOS angle defining the source positons w.r.t the listeners frame of reference
    '''
    
    
    xyz = np.asarray(xyz)
    if xyz.shape[-1] != 3:
        raise ValueError('You must supply all three cartesian coordinates to variable xyz')
    if xyz.ndim == 1:
        xyz = xyz[np.newaxis,:]
    
    hhat = np.asarray(hhat)
    if hhat.shape[-1] != 3:
        raise ValueError('You must supply all three cartesian coordinates to variable hhat')
    if hhat.ndim != 1:
        hhat = np.squeeze(hhat)
    if hhat.shape[0] != 3:
        raise ValueError(f'hhat must be single dimension length 3, you have supplied shape {hhat.shape}')
        
    hhat = hhat / np.linalg.norm(hhat) # In case nhat wasnt normed already
    
    angle = np.zeros(xyz.shape[0])
    for idx in range(xyz.shape[0]):
        vec = xyz[idx,:]
        vec = vec / np.linalg.norm(vec)
        # Get the correct quadrant 
        factor = _quadrantMapping( vec, hhat )
        angle[idx] = factor * abs(np.arccos( np.dot(vec, hhat) ))
        
    return angle

def calculateHOSPlant( HOSAngles, order, orderMatrix=None, HOSType='sine'):
    '''
    This funciton calculates a HOS plant matrix
    This is matrix of encoding coefficients to transform a set of coordinates
    into HOS format
    
    It may be used to create encoding coefficients which are need to transform a 
    set of L plane wave objects positioned at HOSAngles into a HOS format stream
    
    It may also be used to encode a plant matrix of N loudspeaker positions to then
    be inverted for use in a HOS loudspeaker decoder to transform HOS format to
    loudspeaker signals

    Parameters
    ----------
    HOSAngles : Array-like, shape(L)
        HOS angles of the L objects
        Angles the object source positions make w.r.t the x axis in the listeners rotated frame
    order : int
        Order of the encoding
        Order N requires N+1 coefficients
    orderMatrix : Array-like, shape(order+1, L)
        Optional, may be supplied to speed up calculations in realtime
        Array of encoding order coefficients
        The ith and jth element is equal to i
    HOSType : str
        Optional, the encoding type for the HOS format which changes which axis 
        the soundfield reproduction occurs over
        If sine then the interaural axis is assumed to be along the y axis
        If cosine then the interaural axis is assumed to be along the x axis
        Note if HOSAngle has been calculated using a listener rotation compensated 
        interaural axis then the HOSType MUST be sine
    Returns
    -------
    plant : Array-like, shape(order+1, L)
        Plant matrix of HOSAngles calculated for all orders of up to order+1
        for L objects

    '''
    
    HOSAngles = np.asarray(HOSAngles)
    if HOSAngles.ndim != 1:
        raise ValueError('HOSAngle should be 1D array')
        
    if order < 0:
        raise ValueError('Order must be positive')
    order = int(order)
    
    # Check the HOS representation type
    if HOSType.lower() not in ['sine', 'sin', 'cosine', 'cos']:
        raise ValueError(f'Invalid type of HOS representation requested. Must be sine or cosine, supplied {HOSType}')
    
    numObjects = HOSAngles.shape[0]
    numCoeffs  = order+1
    
    # Create matrix of size HOSOrder+1 x num object positions
    HOSAnglesMatrix = np.tile(HOSAngles,(order+1,1))    
    
    # orderMatrix may be supplied to speed up calculation
    if orderMatrix is None:
        orderMatrix = np.tile(np.arange(0,order+1,1),(numObjects,1)).T      # Create matrix of powers
    
    # Create encoding matrix
    if HOSType.lower() in ['sine', 'sin']:
        plant = np.sin(HOSAnglesMatrix)**orderMatrix     
    elif HOSType.lower() in ['cosine', 'cos']:
        plant = np.cos(HOSAnglesMatrix)**orderMatrix           
    
    return plant
        
    

def calculateHOSDecoder( plant, order, beta=None ):
    '''
    Calculates a HOS decoding matrix by performing a pseudoinversion of a HOS plant
    
    Optional regularisation may be included in the inversion
    
    Can be used to invert a plant matrix of loudspeaker positions. Applying the inverted plant
    to a HOS format audio stream will create a set of loudspeaker signals
    

    Parameters
    ----------
    plant : Array-like, shape(order+1, L)
        Plant matrix of HOSAngles calculated for all orders of up to order+1
        for L objects
    order : int
        Order of the encoding
        Order N requires N+1 coefficients
    beta : float, optional
        Tikhonov regularisation parameter

    Returns
    -------
    plantInv : Array-like, shape(L, order+1)
        Pseudoinvese of the plant matrix

    '''
    
    numObjects = plant.shape[1]
    numCoeffs  = order+1
    if plant.shape[0] != numCoeffs:
        raise ValueError(f'The supplied plant matrix is of order: {plant.shape[1]-1} which doesnt match the supplied order: {order}')
    
    plantH = plant.conj().T # hermitian transpose
    if beta is None:
        gram = np.linalg.inv( plant @ plantH ) # no regularisation
    else: 
        betaMatrix = beta * np.identity(numCoeffs)
        gram = np.linalg.inv( (plant @ plantH) + betaMatrix ) # tikhonov regularisation
    plantInv = plantH @ gram
    
    return plantInv
        

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from geometry import sph2cart, applyRotation
    
    # Create a set of source positions ranging from -90 to 90 azimuth
    res = 1 # angular res in deg
    az = np.deg2rad( np.arange(-90,90,res) )
    el = np.zeros(az.shape)
    r  = np.ones(az.shape)
    srcPos_sph = np.stack([az, el, r], axis=-1)
    srcPos_xyz = np.asarray( sph2cart( srcPos_sph ))
    numSrcs = srcPos_xyz.shape[0]

    # listener orientation axis
    hhat = np.array([1,0,0]) # listener head orientation
    # ypr = [np.pi/4, 0, 0]
    ypr = [0,0,0]
    hhat = applyRotation( hhat, ypr )

    # HOS order
    order = 4

    # Loudspeaker Positions
    numSpkrs = order+1 # miniumum required
    az = np.deg2rad( np.linspace(-90,90,numSpkrs) ) # optimal spkr array pos, semicircle in front of listener
    el = np.zeros(az.shape)
    r  = np.ones(az.shape) # assume radially equidistant loudspeakers
    spkrPos_sph = np.stack([az, el, r], axis=-1)
    spkrPos_xyz = np.asarray( sph2cart( spkrPos_sph ))


    # Source angles w.r.t interaural axis
    HOSAngles_src = calculateHOSAngle(srcPos_xyz, hhat)
    # Plant Matrix (encoding coefficents for each source from each given direction)
    srcEncoder = calculateHOSPlant(HOSAngles_src, order, HOSType='sine')

    # Spkr angles w.r.t interaural axis
    HOSAngles_spkrs = calculateHOSAngle(spkrPos_xyz, hhat)
    # Plant Matrix (coeifficents for spkr position)
    spkrPlant = calculateHOSPlant(HOSAngles_spkrs, order, HOSType='sine')
    # Decoding matrix for spkr plant
    spkrDecoder = calculateHOSDecoder(spkrPlant, order, beta=0.00001)

    # Calculate the actual loudspeaker gains for each source position
    gains = np.zeros((numSpkrs, numSrcs))
    for idx, HOSSignals in enumerate(srcEncoder.T):
        gains[:,idx] = spkrDecoder @ HOSSignals
        
        


    clip = [-1.2, 1.2]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twiny()
    ls = ['-', '--', '-.', ':','--']
    for l in range(numSpkrs):
        label = '$g_{'+str(l+1)+'}$'
        ax1.plot(np.rad2deg(HOSAngles_src), gains[l,:], ls=ls[l], label=label)
       
    ax1.plot(np.rad2deg(HOSAngles_src), np.sum(gains, axis=0), label=r'$\sum_{l=1}^L g_l$')

    ax2.vlines(np.rad2deg(spkrPos_sph[:,0]), clip[0], clip[1], color='r', ls='--')

    ax1.set_ylabel(r'Amplitude')
    ax1.set_xlabel(r'HOS Angle (Deg)')
    ax1.grid('on')
    ax1.minorticks_on()
    ax1.set_ylim(clip[0],clip[1])
    ax1.set_xlim(-90,90)
    ax1.set_xticks([-90, -45, 0, 45,90])
    # ax2.set_xticks([0,0.25,0.5,0.75,1])
    # ax2.set_xticklabels([-90,-45,0,45,90])
    ax2.set_xticks([-90,-45,0,45,90])
    ax2.set_xlim(-90,90)
    ax2.set_xlabel('Azimuth (Deg)')

    #        ax1.legend(loc=4, ncol=numcol)
    if numSpkrs > 6: # Double column legend for many loudspeakers
        numcol=2
    elif numSpkrs > 8: # Double column legend for many loudspeakers
        numcol=3
    elif numSpkrs > 10: # Double column legend for many loudspeakers
        numcol=4
    else:
        numcol=1
    ax1.legend(loc=4, ncol=numcol, fontsize=16) #Fontsize change in legend to make it smaller
    plt.tight_layout()
    plt.show()


        
    
    