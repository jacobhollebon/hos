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
import visr, rbbl, pml, rcl
from hos_loudspeaker_decoder_controller import HOSLoudspeakerDecoderController
 

class HOSLoudspeakerDecoder( visr.CompositeComponent ):
    """
    Decode a HOS signal to loudspeaker signals.
    
    Takes a given loudspeaker array, creates then inverts the corresponding plant matrix (with Tikhonov regularisation, if requested).
    
    Next, the loudspeaker signals dependent on the input HOS signals are calculated and outputted.
    
    Compensation for listener yaw head rotations are included 
    
    This decoder depends on a controller to compute the loudspeaker gains
    
    
    
    
    HOS format to loudspeaker signals decoder. 
    
    Takes a given loudspeaker array, creates then inverts the corresponding plant matrix (with Tikhonov regularisation, if requested).
    Recieves a HOS format signal (analogous to HOA Bformat), and applies the inverted plant matrix coefficients (the decoder) to create the
    loudspeaker signals.
    
    Compensation for listener rotations and/or translations is optionally included for the decoder
    
    Dynamic delay and/or gain calibration of the loudspeaker array is optionally included to ensure the array is acoustical equidistant
    
    Relies on the HOSLoudspeakerDecoderController object to calculate the decoding and delay/gain compensation
    """
    def __init__( self,
                context, name, parent,     
                loudspeakerPos,           
                HOSOrder = 0,             
                HOSType = 'Sine' ,       
                beta = None,   
                interpolationSteps = None,
                headOrientation = None,
                headPosition = None,
                useHeadTracking = False,
                usePositionTracking = False
                useYawOnly = True, 
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
        loudspeakerPos: array-like, size (numberOfObjects,2) or numberOfObjects,3) 
            Positions of the loudspeakers in global frame of reference, with the origin at the array center
            Array ordered [az, el, radius] in [rads, rads, m]
            If only angular positions [az, el] are passed, the speakers are assumed to be radially equidistant
        HOSOrder: int
            Order of the HOS encoding
        HOSType: string
            Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
            Ensures angles are correctly identified depending on which axis the reproduction is along.
        beta: float
            Regularisation parameter for inversion of plant matrix. If left None, then no regularisation will be used.
        interpolationSteps: int, optional
           Number of samples to transition to new coefficients in processing blocks after an update.
        headOrientation : array-like
            Head orientation in spherical coordinates (2- or 3-element vector or list). Either a static orientation (when no tracking is used),
            or the initial view direction
        headPosition : array-like
            Head position in x y z cartesian coordinates (3-element vector or list). Either a static position (when no tracking is used),
            or the initial position 
        useHeadTracking: bool
            Whether dynamic head tracking (rotation) is active.
            Opens up top level parameter port named "orientation" to recieve a pml.ListenerPosition object
        usePositionTracking: bool
            Whether dynamic head tracking (position) is active.
            Opens up top level parameter port named "position" to recieve a pml.ListenerPosition object
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
        
        super( HOSLoudspeakerDecoder, self ).__init__( context, name, parent )     
        
        # The setup of the input and output audio channels 
        numIn = HOSOrder+1 # Number of input channels is the B Format, assume HOSOrder same for encoding and loudspeaker decoding
        numLoudspeakers = loudspeakerPos.shape[0] # Number of loudspeakers
        self.audioIn  = visr.AudioInputFloat( "audioIn", self, numIn )  
        self.audioOut = visr.AudioOutputFloat( "audioOut", self, numLoudspeakers )
        
        # Set default value for fading between interpolation
        if interpolationSteps is None:
            interpolationSteps = context.period
            
        # Check dimensions of loudspeakerPos
        if loudspeakerPos.shape[-1] != 3: # If there is no radial position applied, fix them to be at a radius of 1m
            print('Adding radius to loudspeakers')
            loudspeakerPos = np.concatenate( (loudspeakerPos,
                    np.ones( (numLoudspeakers,3-loudspeakerPos.shape[-1]), dtype=np.float32) ), axis=-1 )
        
        # Check the HOS representation type
        if HOSType.lower() not in ['sine', 'sin', 'cosine', 'cos']:
            raise ValueError(f'Invalid type of HOS representation requested. Must be sine or cosine, supplied {HOSType}')
            
            
            
         # The encoding to create the HOS format is applied through a gain matrix
         self.HOSLoudspeakerDecoder = rcl.GainMatrix(context, "HOSLoudspeakerDecodingMatrix", self,
                                                numberOfInputs = numIn,
                                                numberOfOutputs = numLoudspeakers,
                                                interpolationSteps = interpolationSteps,
                                                controlInput=True)
         
         # The decoding calculator accepts positions defining a loudspeaker array and creates coefficeints to decode HOS format to speaker signals 
         self.decodingCalculator = HOSObjectEncoderController(  context, "DecodingCalculator", self,    
                                                                numberOfLoudspeakers = numLoudspeakers,            
                                                                loudspeakerPos = loudspeakerPos,          
                                                                HOSOrder = HOSOrder,             
                                                                HOSType  = HOSType,        
                                                                beta = beta,              
                                                                useHeadTracking = useHeadTracking,  
                                                                initialOrientation = headOrientation,
                                                                useYawOnly = useYawOnly, 
                                                                usePositionTracking = usePositionTracking,
                                                                initialPosition = headPosition,     
                                                                useDelayCompensation = useDelayCompensation,
                                                                useGainCompensation  = useGainCompensation )
        # Send the coefficients of the decoder from the calculator to the deocder matrix
        self.parameterConnection( self.encodingCalculator.parameterPort("coefficientOutput"),
                                  self.HOSLoudspeakerDecoder.parameterPort( "gainInput" ) )
        # If headTracking is specified connect the tracking input to the coefficient calculator
        if headTracking:
            self.trackingInput = visr.ParameterInput( "tracking", self, pml.ListenerPosition.staticType,
                                          pml.DoubleBufferingProtocol.staticType,
                                          pml.EmptyParameterConfig() )
            self.parameterConnection( self.trackingInput,
                                      self.encodingCalculator.parameterPort("orientation") )  
         # If positionTracking is specified connect the position input to the coefficient calculator
        if positionTracking:
            self.positionInput = visr.ParameterInput( "position", self, pml.ListenerPosition.staticType,
                                          pml.DoubleBufferingProtocol.staticType,
                                          pml.EmptyParameterConfig() )
            self.parameterConnection( self.positionInput,
                                      self.decodingCalculator.parameterPort("position") )   
             
            
         
        # Optional dynamic array calibration
        initialDelays = np.zeros(numLoudspeakers)
        initialGains  = np.ones(numLoudspeakers)
        if useDelayCompensation:
            self.delayCalibration = rcl.DelayVector( context, "DelayCalibration", self, 
                                                     numberOfChannels=numLoudspeakers, 
                                                     initialDelay=initialDelays, 
                                                     initialGain=initialGains, 
                                                     controlInputs=rcl.DelayVector.ControlPortConfig.Delay)
            self.parameterConnection( self.decodingCalculator.parameterPort("delayOutput"),
                                      self.delayCalibration.parameterPort("delayInput") )  
        if useGainCompensation:
            self.gainCalibration = rcl.GainVector( context, "GainCalibration", self, 
                                                     numberOfChannels=numLoudspeakers, 
                                                     initialDelay=initialDelays, 
                                                     initialGain=initialGains, 
                                                     controlInputs=rcl.GainVector.ControlPortConfig.Gain)
            self.parameterConnection( self.decodingCalculator.parameterPort("gainOutput"),
                                      self.gainCalibration.parameterPort("gainInput") )  
            
            
        # Audio connections 
        self.audioConnection( self.audioIn, self.HOSLoudspeakerDecoder.audioPort("in") ) # Audio input (HOS format) to the decoder
        if useDelayCompensation and useGainCompensation:
            self.audioConnection( self.HOSLoudspeakerDecoder.audioPort("out"), self.delayCalibration.audioPort("in") ) # speaker signals to delay calib
            self.audioConnection( self.delayCalibration.audioPort("out"), self.gainCalibration.audioPort("in") )  # speaker signals to gain calib
            self.audioConnection( self.gainCalibration.audioPort("out"), self.audioOut ) # Delay and gain calibrated loudspeaker signals to output
        elif useDelayCompensation:
            self.audioConnection( self.HOSLoudspeakerDecoder.audioPort("out"), self.delayCalibration.audioPort("in") )  # speaker signals to delay calib
            self.audioConnection( self.delayCalibration.audioPort("out"), self.audioOut ) # Delay calibrated loudspeaker signals to output
        elif useGainCompensation:
            self.audioConnection( self.HOSLoudspeakerDecoder.audioPort("out"), self.gainCalibration.audioPort("in") ) # speaker signals to gain calib
            self.audioConnection( self.gainCalibration.audioPort("out"), self.audioOut ) # Gain calibrated loudspeaker signals to output
        else:
            self.audioConnection( self.HOSLoudspeakerDecoder.audioPort("out"), self.audioOut ) # Loudspeaker signals to output   
          
        
        
        
        
        
        
        