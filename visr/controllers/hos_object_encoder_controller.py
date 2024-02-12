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

import numpy as np
import warnings

# core VISR packages
import visr
import pml
import objectmodel as om

from geometry import sph2cart, cart2sph, applyRotation
import hos_functions as hos

class HOSObjectEncoderController( visr.AtomicComponent ):
    """
    Component to calculate the encoding coefficients for a number of plane wave
    objects into HOS format (analogous to HOA B Format)
    
    Optional 3DOF headtracking to perform Dynamic HOS
    The headtracking may be reduced to 1DOF (yaw only) by an optional command
    e.g useful for horizontal loudspeaker rendering
    """
    def __init__( self,
                  context, name, parent,    # Standard visr component constructor arguments
                  numberOfObjects,          # The number of plane wave objects rendered.                
                  objectPos = None,         # Initial positions of the objects
                  HOSOrder = 0,             # Order of the HOS encoding
                  HOSType = 'Sine' ,        # What type of HOS encoding
                  useHeadTracking = False,  # Whether head tracking data is provided via a self.headOrientation port.
                  useYawOnly = True, 
                  initialOrientation = None):
        """
        Constructor.

        Parameters
        ----------
        context : visr.SignalFlowContext
            Standard visr.Component construction argument, a structure holding the block size and the sampling frequency
        name : string
            Name of the component, Standard visr.Component construction argument
        parent : visr.CompositeComponent
            Containing component if there is one, None if this is a top-level component of the signal flow.
        numberOfObjects: int
            The number of plane wave objects rendered.
            This must be supplied to initialise signal flow sizes before runtime
            If a larger number is supplied via a scene decoder an error will be raised
        objectPos: array-like, size (numberOfObjects, 2)
            Starting positions of the objects. 
            Second dimension containing (azimuth, elevation) of the sources in rads
        HOSOrder: int
            Order of the HOS encoding
        HOSType: string
           Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
           Ensures angles are correctly identified depending on which axis the reproduction is along.
        useHeadTracking: bool
            Whether the orientation is updated at runtime. If True, a parmater input
            "orientation" is instantiated that receivers pml.ListenerPositions
        initialOrientation: array-like (length 3) or NoneType
            The initial head rotation or the static head orientation if dynamic updates are deactivated. 
            Given as yaw, pitch, roll in radians
        useYawOnly: bool
            If False listener head orientation is tracked w.r.t 3DOF
            If True the pitch and roll of the listener orientation is ignored
        """
        # Call base class (AtomicComponent) constructor
        super( HOSObjectEncoderController, self ).__init__( context, name, parent )
        
        
        self.numberOfObjects = numberOfObjects 
        self.numHOSCoeffs = (HOSOrder+1)
        self.HOSOrder        = HOSOrder
        self.HOSType         = HOSType
        self.useYawOnly      = useYawOnly

        # Encoding coefficient output ports
        self.coeffOutput = visr.ParameterOutput( "coefficientOutput", self,
                                                pml.MatrixParameterFloat.staticType,
                                                pml.DoubleBufferingProtocol.staticType,
                                                pml.MatrixParameterConfig(self.numberOfObjects, self.numHOSCoeffs))
        self.coeffOutputProtocol = self.coeffOutput.protocolOutput()

        # Object metadata input port
        self.objectInput = visr.ParameterInput( "objectVector", self, pml.ObjectVector.staticType,
                                              pml.DoubleBufferingProtocol.staticType,
                                              pml.EmptyParameterConfig() )
        self.objectInputProtocol = self.objectInput.protocolInput()

        # Headtracking input ports
        if useHeadTracking:
            self.orientationInput = visr.ParameterInput( "orientation", self, pml.ListenerPosition.staticType,
                                                      pml.DoubleBufferingProtocol.staticType,
                                                      pml.EmptyParameterConfig() )
            self.orientationInputProtocol = self.orientationInput.protocolInput()
        else:
            self.orientationInputProtocol = None
            
            
        # Perform one process loop offline to create initial coefficients
        if initialOrientation is None:
            initialOrientation = np.zeros( (3), np.float32 )
        else:
            initialOrientation = np.asarray( initialOrientation, dtype = np.float32 ) 
        if initialOrientation.shape[0] != 3:
            raise ValueError(f'Invalid initialOrientation, should be size 3 [y,p,r], supplied {initialOrientation}')
        
        
        # Encoding setup
        # objectPos is size numObjects x 2 (azi and ele)
        if objectPos is None:
            objectPos = np.zeros((self.numberOfObjects,2))
        if objectPos.shape != (numberOfObjects,2):
            raise ValueError(f'Specified starting object positions the wrong shape, should be numObjects x 2 (az, el), supplied {objectPos.shape}' )
        # Find the cartesian coordinates of all the source positions
        self.objectPos_xyz = np.zeros((self.numberOfObjects,3), dtype=np.float32)
        for indx, obj in enumerate(objectPos):
            az = obj[0]
            el = obj[1]
            sph = np.asarray([az,el,1]) # Plane wave source, assume distance of 1m
            xyz = np.asarray( sph2cart( sph ))
            xyzNormed = 1.0/np.sqrt(np.sum(np.square(xyz))) * xyz
            self.objectPos_xyz[indx,:] = xyzNormed
        
        # Calculate hhat, vector pointing in direction of listener look / x axis in listenter frame
        if self.useYawOnly:  # yaw only orientation handling
            initialOrientation[1:] = 0
        self.hhat = applyRotation( [1,0,0], initialOrientation ) # listener orientation axis
        
        # Calculate Plant Matrix (encoding coefficents for each source from each given direction)
        self.HOSAngles = hos.calculateHOSAngle(self.objectPos_xyz, self.hhat) # Source angles w.r.t listener frame
        srcEncoder = hos.calculateHOSPlant(self.HOSAngles, self.HOSOrder, HOSType=self.HOSType) # encoder
        
        # Initiate unitary volume at start up
        self.levels = np.ones( self.numberOfObjects, dtype = np.float32 )
        self.srcEncoder = srcEncoder * self.levels[np.newaxis,:]
        
    def process( self ):
        """
        Processing function, called from the runtime system in every iteration of the signal flow.
        """
        
        recalculateFlag = False
        
        # If orientation data recieved calculate new hhat
        if (self.orientationInputProtocol is not None ) and self.orientationInputProtocol.changed():
            head = self.orientationInputProtocol.data()
            ypr = np.array(head.orientation, dtype = np.float32 ) # Euler angles YPR orientation
            
            # Calculate hhat, vector pointing in direction of listener look / x axis in listenter frame
            if self.useYawOnly:  # yaw only orientation handling
                ypr[1:] = 0
            self.hhat = applyRotation( [1,0,0], ypr ) # listener orientation axis
            
            recalculateFlag = True
            self.orientationInputProtocol.resetChanged()
            
        # If new object vector recieved, recalculate the source positions
        if self.objectInputProtocol.changed():
            ov = self.objectInputProtocol.data();
            objIndicesRaw = [x.objectId for x in ov
                          if isinstance( x, (om.PointSource, om.PlaneWave) ) ]
        
            self.levels = np.zeros( self.numberOfObjects, dtype = np.float32 )
            sphPos = np.zeros( (self.numberOfObjects,3), dtype = np.float32 )
            sphPos[:,2] = 1 # Set unused sources to a valid direction. (az=0, el=0, r=1)
            for index, src in enumerate(ov):
                if index < self.numberOfObjects :
                    if isinstance( src, om.PointSource ):
                        p = np.asarray(src.position, dtype=np.float32 )
                        p = 1.0/np.sqrt(np.sum(np.square(p))) * p # Normalise the distance
                        sph = cart2sph( p[0], p[1], p[2] )
#                            print("\r" + "Object " + str(np.rad2deg(sph)), end="")
                    elif isinstance( src, om.PlaneWave ):
                        AZ = deg2rad( src.azimuth )
                        EL = deg2rad( src.elevation )
                        R = src.referenceDistance
                        sph = np.asarray([AZ,EL,R])
#                            print("\r" + "PW " + str(np.rad2deg(sph)), end="")
#                        print("\r" + "ch " + str(ch), end="")
                    xyz = np.asarray( sph2cart( sph ))
                    xyzNormed = 1.0/np.sqrt(np.sum(np.square(xyz))) * xyz
                    self.objectPos_xyz[index,:] = xyzNormed
                
                    ch = src.channels[0]
                    self.levels[ch]   = src.level
                else:
                    print("\r" + "Breaking channel is ch " + str(index), end="")
                    warnings.warn('The number of dynamically instantiated sound objects is more than the maximum number specified')
                    break
            recalculateFlag = True
            self.objectInputProtocol.resetChanged()
            
        if recalculateFlag:
            # Calculate Plant Matrix (encoding coefficents for each source from each given direction)
            self.HOSAngles = hos.calculateHOSAngle(self.objectPos_xyz, self.hhat) # Source angles w.r.t interaural axis
            srcEncoder = hos.calculateHOSPlant(self.HOSAngles, self.HOSOrder, HOSType=self.HOSType) # encoder
            self.srcEncoder = srcEncoder * self.levels[np.newaxis,:]
            
        
        # Output the encoding coefficients 
        coeffOut = np.array(self.coeffOutputProtocol.data(), copy=False)
        coeffOut[:] = self.srcEncoder
        self.coeffOutputProtocol.swapBuffers()
        
        
    
        
             
        
