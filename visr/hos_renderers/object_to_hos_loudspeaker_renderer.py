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

import visr
import rcl
import pml

from hos_object_encoder import HOSObjectEncoder
from hos_loudspeaker_decoder import HOSLoudspeakerDecoder


class ObjectToHOSLoudspeakerRenderer(visr.CompositeComponent ):
    """
    Renderer to transform a set of audio objects into HOS loudspeaker signals for output to a real loudspeaker array

    This class extends visr_bst.VirtualLoudspeakerRenderer by a configurable head tracking receiver,
    making it suitable for realtime use.
    """
    def __init__( self,
                 context, name, parent,
                 loudspeakerPos,     
                 numObjects = 1,
                 objectPos = None,    
                 sceneReceiveUdpPort = None,  
                 HOSOrder = 0,
                 HOSType = 'Sine',     
                 headOrientation = None,
                 headPosition = None,
                 useHeadTracking = False,
                 usePositionTracking = False,
                 useYawOnly = True, 
                 beta = None,   
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
        numObjects: int
            Number of audio channels/mono objects in.
        objectPos: array-like, size (numberOfObjects, 2)
            Starting positions of the objects. 
            Second dimension containing (azimuth, elevation) of the sources in rads
        sceneReceiveUdpPort: int, optional
            A UDP port number where scene object metadata (in the S3A JSON format) is to be received).
            If not given (default), no network receiver is instantiated, and the object exposes a
            top-level parameter input port "objects"   
        HOSOrder: int
            Order of the HOS encoding
        HOSType: string
            Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
            Ensures angles are correctly identified depending on which axis the reproduction is along.
         headOrientation : array-like
             Head orientation in spherical coordinates (or 3-element vector or list). Either a static orientation (when no tracking is used),
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
        useYawOnly: bool
            If False listener head orientation is tracked w.r.t 3DOF
            If True the pitch and roll of the listener orientation is ignored
        beta: float
            Regularisation parameter for inversion of plant matrix. If left None, then no regularisation will be used.
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

        super( ObjectToHOSLoudspeakerRenderer, self ).__init__( context, name, parent )

        numLoudspeakers = loudspeakerPos.shape[0]
        self.objectSignalInput = visr.AudioInputFloat( "audioIn", self, numObjects )
        self.loudspeakerOutput = visr.AudioOutputFloat( "audioOut", self, numLoudspeakers )


        
        # Check the HOS representation type
        if HOSType.lower() not in ['sine', 'sin', 'cosine', 'cos']:
            raise ValueError(f'Invalid type of HOS representation requested. Must be sine or cosine, supplied {HOSType}')
            
            

        # Encode the objects to HOS Format
        self.HOSObjectEncoder = HOSObjectEncoder( context, "HOSObjectEncoder", self,
                                                    numObjects = numObjects,
                                                    objectPos = objectPos,
                                                    HOSOrder = HOSOrder,                 
                                                    HOSType = HOSType,
                                                    interpolationSteps = None,
                                                    headOrientation = headOrientation,
                                                    useHeadTracking = useHeadTracking,
                                                    useYawOnly = useYawOnly
                                                    )
        # Set up the object metadata port or receiver and patch it to the object encoder
        if sceneReceiveUdpPort is None:
            print('Scene metadata port opened')            
            self.objectMetadata = visr.ParameterInput("objects", self,
                                   protocolType=pml.DoubleBufferingProtocol.staticType,
                                   parameterType=pml.ObjectVector.staticType,
                                   parameterConfig=pml.EmptyParameterConfig() )
            self.parameterConnection( self.objectMetadata,
                                 self.HOSObjectEncoder.parameterPort("objectVector"))
        else:
            print('Scene decoder activated')
            self.sceneReceiver = rcl.UdpReceiver( context, "SceneReceiver", self,
                                                 port = int(sceneReceiveUdpPort) )
            self.sceneDecoder = rcl.SceneDecoder( context, "SceneDecoder", self )
            self.parameterConnection( self.sceneReceiver.parameterPort("messageOutput"),
                                 self.sceneDecoder.parameterPort("datagramInput") )
            self.parameterConnection( self.sceneDecoder.parameterPort( "objectVectorOutput"),
                                 self.HOSObjectEncoder.parameterPort("objectVector"))
               


        # Decode the HOS format to loudspeaker signals, using Dynamic (head-tracked) HOS panning
        self.HOSLoudspeakerDecoder = HOSLoudspeakerDecoder( context, "HOSLoudspeakerDecoder", self,
                                                             loudspeakerPos,           
                                                             HOSOrder = HOSOrder,        
                                                             HOSType = HOSType,
                                                             beta = beta,
                                                             interpolationSteps = None,
                                                             headOrientation = headOrientation,
                                                             headPosition = headPosition,
                                                             useHeadTracking = useHeadTracking,
                                                             usePositionTracking = usePositionTracking,
                                                             useYawOnly = useYawOnly, 
                                                             useDelayCompensation = useDelayCompensation,
                                                             useGainCompensation = useGainCompensation,
                                                             )
        
        # Orientation and position tracking ports
        if useHeadTracking:
            self.trackingInput = visr.ParameterInput( "orientation", self, pml.ListenerPosition.staticType,
                                          pml.DoubleBufferingProtocol.staticType,
                                          pml.EmptyParameterConfig() )
            self.parameterConnection( self.trackingInput, self.HOSObjectEncoder.parameterPort("tracking"))
            self.parameterConnection( self.trackingInput, self.HOSLoudspeakerDecoder.parameterPort("tracking"))
        if usePositionTracking:
            self.positionInput = visr.ParameterInput( "position", self, pml.ListenerPosition.staticType,
                                          pml.DoubleBufferingProtocol.staticType,
                                          pml.EmptyParameterConfig() )
            self.parameterConnection( self.positionInput, self.HOSLoudspeakerDecoder.parameterPort("position"))

        
            
        # Audio Connections
        self.audioConnection( self.objectSignalInput, self.HOSObjectEncoder.audioPort("audioIn")) # Audio input to HOS object Encoder 
        self.audioConnection( self.HOSObjectEncoder.audioPort("audioOut"), self.HOSLoudspeakerDecoder.audioPort("audioIn")) # HOS format to HOS gain calculator
        self.audioConnection( self.HOSLoudspeakerDecoder.audioPort("audioOut"), self.loudspeakerOutput) # HOS loudspeaker signals to the audio output





if __name__ == "__main__":
    import numpy as np
    
    fs = 48000
    blockSize = 1024
    context = visr.SignalFlowContext(blockSize, fs)
        
    
    # HOS order
    order = 4
   
    # Source Positions
    srcPos_sph = np.stack([[0, 0]], axis=-1).T
    numSrcs = srcPos_sph.shape[0]
    
    # Loudspeaker Positions
    numSpkrs = order+1 # miniumum required
    az = np.deg2rad( np.linspace(-90,90,numSpkrs) ) # optimal spkr array pos, semicircle in front of listener
    el = np.zeros(az.shape)
    r  = np.ones(az.shape) # assume radially equidistant loudspeakers
    spkrPos_sph = np.stack([az, el, r], axis=-1)

    
    renderer = ObjectToHOSLoudspeakerRenderer(context, "HOSRenderer", None,                                               
                                            loudspeakerPos=spkrPos_sph,     
                                            numObjects = numSrcs,
                                            objectPos = srcPos_sph,    
                                            sceneReceiveUdpPort = 8001,  
                                            HOSOrder = order,
                                            HOSType = 'Sine',     
                                            headOrientation = None,
                                            headPosition = None,
                                            useHeadTracking = False,
                                            usePositionTracking = False,
                                            useYawOnly = False, 
                                            beta = None,   
                                            useDelayCompensation = False,
                                            useGainCompensation = False,
                                            )
                    
        