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
import pml
import rcl

from hosrealtime.hosrt import HOSObjectEncoder


class RealtimeObjectToHOSEncoder(visr.CompositeComponent ):
    """
    Wrapper of the ObjectToHOSEncoder to include handling of an optional headtracker module
    
    Renderer to encode a set of audio objects into HOS format 
    
    Compensation for listener rotations is optionally included 
    
    """
    def __init__( self,
                 context, name, parent,
                 numObjects = 1,
                 objectPos = None,    
                 sceneReceiveUdpPort = None,  
                 HOSOrder = 0,
                 HOSType = 'Sine',     
                 headOrientation = None,
                 useOrientationTracking = False,
                 useYawOnly = True, 
                 headTracker = None,
                 headTrackerPositionalArguments = None,
                 headTrackerKeywordArguments = None,
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
        numObjects: int
            Number of audio channels/mono objects in.
        objectPos: array-like, size (numberOfObjects, 2)
            Starting positions of the objects. 
            Second dimension containing (azimuth, elevation) of the sources in rads
        sceneReceiveUdpPort: int, optional
            A UDP port number where scene object metadata (in the S3A JSON format) is to be received.
            If not given (default), no network receiver is instantiated, and the object exposes a
            top-level parameter input port "objects" expecting a pml.ObjectVector containing the scene description
        HOSOrder: int
            Order of the HOS rendering (encoding and decoding)
        HOSType: string
            Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
            Ensures angles are correctly identified depending on which axis the reproduction is along.
        headOrientation : array-like
            Head orientation as yaw pitch roll in radians (3-element vector or list). Either a static orientation (when no tracking is used),
            or the initial view direction
        useOrientationTracking: bool
            Whether dynamic head tracking (rotation) is active.
            Opens up top level parameter port named "tracking" to recieve a pml.ListenerPosition object
       useYawOnly: bool
            If False listener head orientation is tracked w.r.t 3DOF
            If True the pitch and roll of the listener orientation is ignored
       headTracker: class
            Class of a VISR headtracker
            Some headtrackers only support orientation, others also support positional tracking
            Make sure the useOrientationTracking and usePositionTracking arguments align with the requested headTracker
        headTrackerPositionalArguments: dict
            Dictionairy of positional arguments to initate the supplied VISR headTracker class with
        headTrackerKeywordArguments: dict
            Dictionairy of keyword arguments to initate the supplied VISR headTracker class with
        """
        
        super( RealtimeObjectToHOSEncoder, self ).__init__( context, name, parent )
        
        self.objectSignalInput = visr.AudioInputFloat( "audioIn", self, numObjects )
        self.hosOutput = visr.AudioOutputFloat( "audioOut", self, HOSOrder+1 )

        
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
                                                    useOrientationTracking = useOrientationTracking,
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
        
        # Setup headtracker
        if useOrientationTracking:
            if headTracker is None:
                raise ValueError('You have requested listener tracking but not supplied the headtracker setup!')
            else:
                if headTrackerPositionalArguments == None:
                    headTrackerPositionalArguments = ()
                if headTrackerKeywordArguments == None:
                    headTrackerKeywordArguments = {}
                self.trackingDevice = headTracker(context, "HeadTrackingReceiver", self,
                                                    *headTrackerPositionalArguments,
                                                    **headTrackerKeywordArguments )
                
                self.parameterConnection( self.trackingDevice.parameterPort("orientation"), self.HOSObjectEncoder.parameterPort("tracking"))
                
        # Audio Connections
        self.audioConnection( self.objectSignalInput, self.HOSObjectEncoder.audioPort("audioIn")) # Object-based audio input to HOS renderer
        self.audioConnection( self.HOSObjectEncoder.audioPort("audioOut"), self.hosOutput) # HOS loudspeaker signals to the audio output



                    
        