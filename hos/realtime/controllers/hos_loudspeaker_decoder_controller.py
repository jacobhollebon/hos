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

# core VISR packages
import visr
import pml

import hos
from hos.toolboxes.geometry import sph2cart, cart2sph, applyRotation

class HOSLoudspeakerDecoderController( visr.AtomicComponent ):
    """
    Component to calculate the loudspeaker gains for a given loudspeaker array 
    decoding an input HOS format (analogous to HOA B format) to loudspeaker signals
    
    Optional 6DOF, 4DOF, 3DOF or 1DOF headtracking to perform Dynamic HOS 
    The orientation headtracking may be reduced to 1DOF (yaw only) e.g useful for 
    horizontal loudspeaker rendering
    6DOF: usePositionTracking = True,  useOrientationTracking = True, useYawOnly = False
    4DOF: usePositionTracking = True,  useOrientationTracking = False, useYawOnly = False
    3DOF: usePositionTracking = False, useOrientationTracking = True, useYawOnly = False
    1DOF: usePositionTracking = False, useOrientationTracking = True, useYawOnly = True
    
    Optional array compensation (dynamic loudspeaker delay alignment and/or 
    dynamic gain compensation w.r.t current listener position)
    
    """
    def __init__( self,
                  context, name, parent,    # Standard visr component constructor arguments
                  numberOfLoudspeakers,     # The number of loudspeakers               
                  loudspeakerPos,           # Initial positions of the loudspeakers
                  HOSOrder = 0,             # Order of the HOS encoding
                  HOSType = 'Sine' ,        # What type of HOS encoding
                  beta = None,              # Regularisation parameter for the pseudoinversion
                  useOrientationTracking = False,  # Whether head tracking data is provided via a self.headOrientation port.
                  initialOrientation = None,
                  useYawOnly = True, 
                  usePositionTracking = False,
                  initialPosition = None,     
                  useDelayCompensation = False,
                  useGainCompensation = False,
                  ):
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
        numberOfLoudspeakers: int
            The number of plane wave objects rendered.
            This must be supplied to initialise signal flow sizes before runtime
            If a larger number is supplied via a scene decoder an error will be raised
        loudspeakerPos: array-like, size (numberOfObjects,3)
            Positions of the loudspeakers in global frame of reference, with the origin at the array center
            Array ordered [az, el, radius] in [rads, rads, m]
        HOSOrder: int
            Order of the HOS encoding
        HOSType: string
            Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
            Ensures angles are correctly identified depending on which axis the reproduction is along.
        beta: float
            Regularisation parameter for inversion of plant matrix. If left None, then no regularisation will be used.
        useOrientationTracking: bool
            Whether the orientation is updated at runtime. If True, a parmater input
            "tracking" is instantiated that receivers pml.ListenerPositions
        initialOrientation: array-like (length 3) or NoneType
            The initial head rotation or the static head orientation if dynamic updates are deactivated. 
            Given as yaw, pitch, roll in radians
            If None supplied, listener assumed to be facing forwards
        useYawOnly: bool
            If False listener head orientation is tracked w.r.t 3DOF
            If True the pitch and roll of the listener orientation is ignored
        usePositionTracking: bool
            Whether the position is updated at runtime. If True, a parmater input
            "tracking" is instantiated that receivers pml.ListenerPositions
        initialPosition: array-like (length 3) or NoneType
            The initial listener position or the static listener position if dynamic updates are deactivated. 
            Given as [x, y, z] in m
            If None supplied, listener assumed to be at the origin
        useDelayCompensation: bool
            If True an output port named "delayOutput" is inistitated that
            outputs a delay per loudspeaker to time align them w.r.t the current
            listener position, such that the array is radially acousticall equidistant
            Note the delays are minimised such that the furthest loudspeaker has 0 delay
            Should be connected to a delay vector component at a higher level
        useGainCompensation: bool
            If True an output port named "gainOutput" is inistitated that
            outputs a linear gain value per loudspeaker to account for different spherical spreading
            attenuation, such that all loudspeakers are volume normalised at the the current 
            listener position.
            Note the gains are normalised such the furthest loudspeaker has a gain of 1
            and all other gains are attenuations (no amplification)
            Should be connected to a gain or delay vector component at a higher level
        """
        # Call base class (AtomicComponent) constructor
        super( HOSLoudspeakerDecoderController, self ).__init__( context, name, parent )
        
        self.numberOfLoudspeakers = numberOfLoudspeakers 
        self.loudspeakerPos       = loudspeakerPos
        self.numHOSCoeffs    = (HOSOrder+1)
        self.HOSOrder        = HOSOrder
        self.HOSType         = HOSType
        self.beta            = beta
        self.useYawOnly      = useYawOnly
        self.useDelayCompensation = useDelayCompensation
        self.useGainCompensation  = useGainCompensation
        self.c = 343 # speed of sound
        
        # Encoding coefficient output ports
        self.coeffOutput = visr.ParameterOutput( "coefficientOutput", self,
                                                pml.MatrixParameterFloat.staticType,
                                                pml.SharedDataProtocol.staticType,
                                                pml.MatrixParameterConfig(self.numberOfLoudspeakers, self.numHOSCoeffs))
        self.coeffOutputProtocol = self.coeffOutput.protocolOutput()
        if self.useDelayCompensation:
            outConfigDelays = pml.VectorParameterConfig( self.numberOfLoudspeakers )
            self.delayOutput = visr.ParameterOutput( "delayOutput", self,
                                                    pml.VectorParameterFloat.staticType,
                                                    pml.DoubleBufferingProtocol.staticType,
                                                    outConfigDelays )
            self.delayOutputProtocol = self.delayOutput.protocolOutput()
        if self.useGainCompensation:
            outConfigGains = pml.VectorParameterConfig( self.numberOfLoudspeakers )
            self.gainOutput = visr.ParameterOutput( "gainOutput", self,
                                                    pml.VectorParameterFloat.staticType,
                                                    pml.DoubleBufferingProtocol.staticType,
                                                    outConfigGains )
            self.gainOutputProtocol = self.gainOutput.protocolOutput()
        
            
        if useOrientationTracking or usePositionTracking:
            self.listenerInput = visr.ParameterInput( "tracking", self, pml.ListenerPosition.staticType,
                                                      pml.DoubleBufferingProtocol.staticType,
                                                      pml.EmptyParameterConfig() )
            self.listenerInputProtocol = self.listenerInput.protocolInput()
        else:
            self.listenerInputProtocol = None
      

            
        # Perform one process loop offline to create initial coefficients
        if initialOrientation is None:
            initialOrientation = np.zeros( (3), np.float32 )
        else:
            initialOrientation = np.asarray( initialOrientation, dtype = np.float32 ) 
        if initialOrientation.shape[0] != 3:
            raise ValueError(f'Invalid initialOrientation, should be size 3 [y,p,r], supplied {initialOrientation}')
        
        if initialPosition is None:
            initialPosition = np.zeros( (3), np.float32 )
        else:
            initialPosition = np.asarray( initialPosition, dtype = np.float32 ) 
        if initialPosition.shape[0] != 3:
            raise ValueError(f'Invalid initialPosition, should be size 3 [x,y,z], supplied {initialPosition}')
        
        # Calculate hhat, vector pointing in direction of listener look / x axis in listenter frame
        if self.useYawOnly:  # yaw only orientation handling
            initialOrientation[1:] = 0
        self.hhat = applyRotation( [1,0,0], initialOrientation ) # listener orientation axis
        
        # Loudspeaker array setup
        # loudspeakerPos is size numLoudspeakers x 3 (azi and ele and radius)
        if self.loudspeakerPos.shape != (self.numberOfLoudspeakers,3):
            raise ValueError(f'Specified loudspeaker positions is the wrong shape, should be numObjects x 3 (az, el, rad), supplied {loudspeakerPos.shape}' )
        self.loudspeakerPos_xyz = sph2cart( self.loudspeakerPos ) # in world frame of reference
        self.loudspeakerPos_xyz_rel2lis = self.loudspeakerPos_xyz - initialPosition # Spkr pos relative to listener position
        
        # Compensation step
        if self.useDelayCompensation or self.useGainCompensation:
            self.loudspeakerPos_rel2lis = cart2sph( self.loudspeakerPos_xyz_rel2lis[:,0], self.loudspeakerPos_xyz_rel2lis[:,1], self.loudspeakerPos_xyz_rel2lis[:,2] )
            self.radius = self.loudspeakerPos_rel2lis[:,2]
            if self.useDelayCompensation:
                # Delays that 'push' closer speakers acoustically back to the radius of the further speaker
                self.delays = (np.max(self.radius) - self.radius) / self.c 
            if self.useGainCompensation:
                # Furthest speaker has gain of 1
                # Attenuate closer speakers by 1/4piR with R relative difference to furthest speaker
                # So only attenuate the minimum amount necessary 
                self.gains = 1/(4*np.pi*(np.max(self.radius) - self.radius))
            
        # Calculate Plant Matrix (encoding coefficents for each loudspeaker from each given direction)
        self.HOSAngles = hos.calculateHOSAngle(self.loudspeakerPos_xyz_rel2lis, self.hhat) # Loudspeaker angles w.r.t listener frame
        plant = hos.calculateHOSPlant(self.HOSAngles, self.HOSOrder, HOSType=self.HOSType) 
        self.decoder = hos.calculateHOSDecoder( plant, self.HOSOrder, beta=self.beta )
        
        # save curr ypr and position
        self.lastYPR = initialOrientation
        self.lastPos = initialPosition
        
    def process( self ):
        """
        Processing function, called from the runtime system in every iteration of the signal flow.
        """
        
        recalculateFlag = False
        # If new listener input recalculate orientation or position as appropriate
        if (self.listenerInputProtocol is not None ) and self.listenerInputProtocol.changed():
            listener = self.listenerInputProtocol.data()
            if self.useOrientationTracking:
                ypr = np.array(listener.orientation, dtype = np.float32 ) # Euler angles YPR orientation
            else:
                ypr = self.lastYPR # use initialOrientation
            if self.usePositionTracking:
                pos = np.array(listener.position, dtype = np.float32 ) # Cartesian position
            else:
                pos = self.lastPos # use initialPosition
                
            # yaw only orientation handling
            if self.useYawOnly:  
                ypr[1:] = 0
                    
            if ypr != self.lastYPR:
                # Calculate hhat, vector pointing in direction of listener look / x axis in listenter frame
                self.hhat = applyRotation( [1,0,0], ypr ) # listener orientation axis
                self.lastYPR = ypr
                recalculateFlag = True
            
            if pos != self.lastPos:
                # Spkr positions relative to listener position
                self.loudspeakerPos_xyz_rel2lis = self.loudspeakerPos_xyz - pos 
                self.lastPos = pos
                recalculateFlag = True
                
                # Speaker compensation step
                if self.useDelayCompensation or self.useGainCompensation:
                    self.loudspeakerPos_rel2lis = cart2sph( self.loudspeakerPos_xyz_rel2lis[:,0], self.loudspeakerPos_xyz_rel2lis[:,1], self.loudspeakerPos_xyz_rel2lis[:,2] )
                    self.radius = self.loudspeakerPos_rel2lis[:,2]
                    if self.useDelayCompensation:
                        # Delays that 'push' closer speakers acoustically back to the radius of the further speaker
                        self.delays = (np.max(self.radius) - self.radius) / self.c 
                    if self.useGainCompensation:
                        # Furthest speaker has gain of 1
                        # Attenuate closer speakers by 1/4piR with R relative difference to furthest speaker
                        # So only attenuate the minimum amount necessary 
                        self.gains = 1/(4*np.pi*(np.max(self.radius) - self.radius))
            
            self.listenerInputProtocol.resetChanged()
        
      
        if recalculateFlag:
            # Calculate Plant Matrix (encoding coefficents for each loudspeaker from each given direction)
            self.HOSAngles = hos.calculateHOSAngle(self.loudspeakerPos_xyz_rel2lis, self.hhat) # Loudspeaker angles w.r.t listener frame
            plant = hos.calculateHOSPlant(self.HOSAngles, self.HOSOrder, HOSType=self.HOSType) 
            self.decoder = hos.calculateHOSDecoder( plant, self.HOSOrder, beta=self.beta )
        
        # Output the decoder coefficients 
        coeffOut = np.array(self.coeffOutputProtocol.data(), copy=False)
        coeffOut[:] = self.decoder
        
        print(self.decoder)
        
        # Output compensation coefficients
        if self.useDelayCompensation: 
             delayOut = np.array( self.delayOutputProtocol.data(), copy = False )
             delayOut[:] = self.delays
             # self.delayOutputProtocol.swapBuffers()
        if self.useGainCompensation:
             gainOut = np.array( self.gainOutputProtocol.data(), copy = False )
             gainOut[:] = self.gains
             
        
            
            

            