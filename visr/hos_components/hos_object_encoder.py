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

    
import visr, rbbl, pml, rcl
from hos_object_encoder_controller import HOSObjectEncoderController

class HOSObjectEncoder( visr.CompositeComponent ):
    """
    Object to HOS format encoder. 
    
    Recieves a number of objects (mono audio channels) and encodes then mixes them to a HOS format signal.
    
    Object metadata supplied in VISR S3A JSON format
    
    Includes head rotation compensation. yawOnly activates whether the object encoding AND head rotation compensation is for 1DOF (yaw) or 3DOF
    
    Relies on the HOSObjectEncoderController object to calculate the encoding
    """
    def __init__( self,
                 context, name, parent,
                 numObjects = 1,
                 objectPos = None,
                 HOSOrder = 0,
                 HOSType = 'Sine',
                 interpolationSteps = None,
                 headOrientation = None,
                 headTracking = False,
                 useYawOnly = False,
                 ):
        """ 
        Constructor.

        Parameters
        ----------
        context : visr.SignalFlowContext
            Standard visr.Component construction argument, a structure holding the block size and the sampling frequency.
        name : string
            Name of the component, Standard visr.Component construction argument.
        parent : visr.CompositeComponent
            Containing component if there is one, None if this is a top-level component of the signal flow.
        numObjects: int
            Number of audio channels/mono objects in.
        HOSOrder: float
            Order of the HOS encoding.
        HOSType: string
           Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
           Ensure angles are correctly identified depending on which axis the reproduction is along.
        interpolationSteps: int, optional
           Number of samples to transition to new object positions after an update.
        headOrientation : array-like
            Head orientation in spherical coordinates (2- or 3-element vector or list). Either a static orientation (when no tracking is used),
            or the initial view direction
        headTracking: bool
            Whether dynamic head tracking is active.
        useYawOnly: bool
            Whether 3DOF (false) headtracking or 1DOF (yaw only) headtracking is implemented. 
        """
        
        super( HOSObjectEncoder, self ).__init__( context, name, parent )     
        
        # The setup of the input and output audio channels 
        numIn  = numObjects
        numOut = HOSOrder+1 
        self.audioIn  = visr.AudioInputFloat( "audioIn", self, numIn )  
        self.audioOut = visr.AudioOutputFloat( "audioOut", self, numOut )
        
        # Check the HOS representation type
        if HOSType.lower() not in ['sine', 'sin', 'cosine', 'cos']:
            raise ValueError(f'Invalid type of HOS representation requested. Must be sine or cosine, supplied {HOSType}')
            
        # Set default value for fading between interpolation
        if interpolationSteps is None:
            interpolationSteps = context.period
         
            
         
        # The encoding to create the HOS format is applied through a gain matrix
        self.HOSObjectEncoder = rcl.GainMatrix(context, "HOSObjectEncodingMatrix", self,
                                               numberOfInputs = numIn,
                                               numberOfOutputs = numOut,
                                               interpolationSteps = interpolationSteps,
                                               controlInput=True)
        
        
        
        
        # The encoding calculator accepts an object vector input and calculates the encoding coefficients to create the HOS format
        self.encodingCalculator = HOSObjectEncoderController(  context, "EncodingCalculator", self,    
                                                               numberOfObjects = numObjects,
                                                               objectPos = objectPos,
                                                               HOSOrder = HOSOrder,             
                                                               HOSType = HOSType,    
                                                               useHeadTracking = headTracking,  
                                                               initialOrientation = headOrientation,
                                                               useYawOnly = useYawOnly)
        
        # Setup object metadata port and send to the Encoder
        self.objectVectorInput = visr.ParameterInput( "objectVector", self, pml.ObjectVector.staticType,
                                                     pml.DoubleBufferingProtocol.staticType,
                                                     pml.EmptyParameterConfig() )
        self.parameterConnection( self.objectVectorInput, self.encodingCalculator.parameterPort("objectVector"))
        
        # Send the coefficients of the encoding from the calculator to the encoder
        self.parameterConnection( self.encodingCalculator.parameterPort("coefficientOutput"),
                                    self.HOSObjectEncoder.parameterPort( "gainInput" ) )
        
        # If headTracking is specified connect the tracking input to the coefficient calculator
        if headTracking:
            self.trackingInput = visr.ParameterInput( "tracking", self, pml.ListenerPosition.staticType,
                                          pml.DoubleBufferingProtocol.staticType,
                                          pml.EmptyParameterConfig() )
            self.parameterConnection( self.trackingInput,
                                     self.encodingCalculator.parameterPort("orientation") )  
        
        
        # Audio connections 
        self.audioConnection( self.audioIn, self.HOSObjectEncoder.audioPort("in") ) # Audio input to the encoder
        self.audioConnection( self.HOSObjectEncoder.audioPort("out"), self.audioOut ) # Output of the encoder to audio out - this is HOS format    
          

 
