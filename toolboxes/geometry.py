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


### Functions for handling geometry operations ###


import math
import numpy as np
from scipy.spatial.transform import Rotation as rot

# note: the input format is inconsistent with the return value and with sph2cart
# (individual values vs single matrix)
def cart2sph(x,y,z):
    ''' 
    port from visr geoemtry functions
    visr uses a listener centric coordinate system as per the SOFA conventions
    cart: +x pointing front of listener, +y to the left ear, +z up
    sph: azimuth angle from x axis in x-y plane (0 to 2pi)
         elevation from x-y plane (-pi/2 to pi/2)
    '''
    radius = np.sqrt( x*x + y*y + z*z )
    az = np.arctan2( y, x )
    el = np.arcsin( z / radius )
    sph = np.stack( (az, el, radius) )
    return sph

def sph2cart( sph ):
    ''' 
    port from visr geoemtry functions
    visr uses a listener centric coordinate system as per the SOFA conventions
    cart: +x pointing front of listener, +y to the left ear, +z up
    sph: azimuth angle from x axis in x-y plane (0 to 2pi)
         elevation from x-y plane (-pi/2 to pi/2)
    '''
    elFactor = np.cos( sph[...,1] )
    x = np.cos( sph[...,0] ) * elFactor * sph[...,2]
    y = np.sin( sph[...,0] ) * elFactor * sph[...,2]
    z = np.sin( sph[...,1] ) * sph[...,2]
    cart = np.stack( (x,y,z), axis=-1 )
    return cart

def applyRotation( xyz, ypr, isInverseRotation=False, isDegrees=False):
    '''
    Applies a rotation as per ypr euler angles to a set of cartesian vectors
    Wrapper for scipy.spatial.transform.Rotation object

    Parameters
    ----------
    xyz : Array-like, shape (N, 3) or (3)
        Cartesian coordinates to be rotated
        May be an array of N different coordinates or a single set
    ypr : Array-like, shape (3)
        Array of euler angles (yaw, pitch, roll) that characeterise the rotation
        In rads, unless inDegrees=True
    isInverseRotation : Bool, optional
        If True the inverse of the ypr is applied
        This should be True if compensating for a listener head rotation 
    isDegrees : Bool, optional
        If True assumes the ypr vector is supplied in degrees
        The default is False.

    Returns
    -------
    xyz_rot : Array-like, shape (N, 3) or (3)
        Rotated cartesian coordinates
    '''
    
    xyz = np.asarray(xyz)
    if xyz.shape[-1] != 3:
        raise ValueError('You must supply all three cartesian coordinates to variable xyz')
    
    ypr = np.asarray(ypr)
    if ypr.shape[-1] != 3:
        raise ValueError('You must supply all three euler angles to variable ypr')
    
    r = rot.from_euler('zxy', ypr, degrees=isDegrees)
    xyz_rot = r.apply(xyz, inverse=isInverseRotation)
    return xyz_rot

def estimateAndApplyRotation( xyz, hhat ):
    '''
    Estimates the rotation vector to rotate the global frame of reference
    to the listener frame of reference defined by hhat
    
    Then apply rotation to the supplied vectors xyz
    
    Parameters
    ----------
    xyz : Array-like, shape (N, 3) or (3)
        Cartesian coordinates to be rotated
        May be an array of N different coordinates or a single set
    hhat : Array-like, shape (3)
        Cartesian coordinates in the world frame of reference defining 
        the look direction of the listener (which is in the x axis in the 
        listener frame of reference)

    Returns
    -------
    xyz_rot : Array-like, shape (N, 3) or (3)
        Rotated cartesian coordinates, now in the listener frame of reference
    '''
    
    xyz = np.asarray(xyz)
    if xyz.shape[-1] != 3:
        raise ValueError('You must supply all three cartesian coordinates to variable xyz')
    
    hhat = np.asarray(hhat)
    if hhat.shape[-1] != 3:
        raise ValueError('You must supply all three cartesian coordinates to variable hhat')
    if hhat.ndim != 1:
        raise ValueError(f'hhat must be single dimension length 3, you have supplied shape {hhat.shape}')
     
    r, _= rot.align_vectors([1,0,0], hhat) # calculates rotation from global to listener frame
    isDegrees = False
    ypr = r.as_euler('zxy', degrees=isDegrees)
    xyz_rot = applyRotation(xyz, ypr, isInverseRotation=False, isDegrees=isDegrees)
    
    return xyz_rot

    
def calcRotationMatrix( ypr ):
    ''' Port of visr rot calculator '''
    if ypr.shape[-1] != 3:
        raise ValueError( "Trailing dimension of ypr argument must be 3.")

    phi = ypr[...,0]
    the = ypr[...,1]
    psi = ypr[...,2]

    a11 = np.cos(the) * np.cos(phi)
    a12 = np.cos(the) * np.sin(phi)
    a13 = -np.sin(the)

    a21 = np.sin(psi) * np.sin(the) * np.cos(phi) - np.cos(psi) * np.sin(phi)
    a22 = np.sin(psi) * np.sin(the) * np.sin(phi) + np.cos(psi) * np.cos(phi)
    a23 = np.cos(the) * np.sin(psi)

    a31 = np.cos(psi) * np.sin(the) * np.cos(phi) + np.sin(psi) * np.sin(phi)
    a32 = np.cos(psi) * np.sin(the) * np.sin(phi) - np.sin(psi) * np.cos(phi)
    a33 = np.cos(the) * np.cos(psi)

    rotation = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    return rotation

def calcRotationMatrixYawOnly( y ):
    ''' Port of visr rot calculator for rotation matrix yaw only '''
#    if y.shape[-1] != 1:
#        raise ValueError( "Trailing dimension of y argument must be 1.")

    phi = y

    a11 =  np.cos(phi)
    a12 = -np.sin(phi)
    a13 = 0

    a21 = np.sin(phi)
    a22 = np.cos(phi)
    a23 = 0

    a31 = 0
    a32 = 0
    a33 = 1

    rotation = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    return rotation
