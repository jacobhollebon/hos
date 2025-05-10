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


### Functions for plotting ###


import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
import matplotlib as mpl; mpl.rcParams['figure.dpi'] = 200;  mpl.rcParams['savefig.dpi'] = 300

   
def pointsOnSphere(pos, includeRadius=False, includeSphere=True):
    """
    Plot a set of spherical points in 3D

    Parameters
    ----------
    pos : Array-like or a list of arrays, each shape(Q, 3) 
        Q x spherical resynthesis positions with trailing dimension ordered
        azimuth (rads, 0 to 2pi)
        elevation (rads, pi/2 to -pi/2)
        radius (m, 0 to inf)
        If a list is supplied, multiple sets of points will be plotted
        Each array may be a different number of positions
    includeRadius : bool, optional
        If False the radius entry of pos is ignored and the points
        are plotted over a unit sphere
    includeSphere : bool, optional
        If True, the outline of a unit sphere is also plotted.

    Returns
    -------
    None.

    """

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if includeSphere:
        # Create sphere
        r = 1
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(phi)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3)
    
    # If not a list, make it a list
    if not isinstance(pos, list):
        pos = [pos]
    
    # Cycle over all sets of points
    for currPos in pos:
        if not includeRadius:
            currPos[:,2] = 1 # nullify radial coordinate
        az = currPos[:, 0]
        el = currPos[:, 1]
        r = currPos[:, 2]
        
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        
        ax.scatter(x,y,z)
    
    if includeSphere:
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()