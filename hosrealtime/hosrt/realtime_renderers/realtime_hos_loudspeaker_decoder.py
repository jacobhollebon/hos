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


import visr

from hosrealtime.hosrt import HOSLoudspeakerDecoder


class RealtimeHOSLoudspeakerDecoder(visr.CompositeComponent ):
    """
    Wrapper of the HOSLoudspeakerDecoder to include handling of an optional headtracker module
    
    Renderer to decode HOS format signals into loudspeaker signals
    
    Compensation for listener rotations and/or translations is optionally included 
    
    """
    def __init__( self,
                 context, name, parent,
                 loudspeakerPos,       
                 HOSOrder = 0,
                 HOSType = 'Sine',     
                 headOrientation = None,
                 headPosition = None,
                 useOrientationTracking = False,
                 usePositionTracking = False,
                 useYawOnly = True, 
                 beta = None,   
                 useDelayCompensation = False,
                 useGainCompensation = False,
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
        loudspeakerPos: array-like, size (numberOfObjects,2) or numberOfObjects,3) 
            Positions of the loudspeakers in global frame of reference, with the origin at the array center
            Array ordered [az, el, radius] in [rads, rads, m]
            If only angular positions [az, el] are passed, the speakers are assumed to be radially equidistant
        HOSOrder: int
            Order of the HOS rendering (encoding and decoding)
        HOSType: string
            Type of HOS Representation, either 'Sine' (y axis reconstruction) or 'Cosine' (x axis reconstruction). 
            Ensures angles are correctly identified depending on which axis the reproduction is along.
        headOrientation : array-like
            Head orientation as yaw pitch roll in radians (3-element vector or list). Either a static orientation (when no tracking is used),
            or the initial view direction
        headPosition : array-like
            Head position in x y z cartesian coordinates (3-element vector or list). Either a static position (when no tracking is used),
            or the initial position 
        useOrientationTracking: bool
            Whether dynamic head tracking (rotation) is active.
            Opens up top level parameter port named "tracking" to recieve a pml.ListenerPosition object
        usePositionTracking: bool
            Whether dynamic head tracking (position) is active.
            Opens up top level parameter port named "tracking" to recieve a pml.ListenerPosition object
        useYawOnly: bool
            If False listener head orientation is tracked w.r.t 3DOF
            If True the pitch and roll of the listener orientation is ignored
        beta: float
            Regularisation parameter for inversion of plant matrix. If left None, then no regularisation will be used.
        useDelayCompensation: bool
            If True the loudspeakers are time aligned w.r.t to the head center of the current
            listener cartesian position, such that the array is radially acousticall equidistant
            Note the delays are minimised such that the furthest loudspeaker has 0 delay
        useGainCompensation: bool
            If True the loudspeakers are compensated by a gain to account for different spherical spreading
            attenuation, such that all loudspeakers are volume normalised at the head center of the current 
            listener cartesian position.
            Note the gains are normalised such the furthest loudspeaker has a gain of 1
            and all other gains are attenuations (no amplification)
        headTracker: class
            Class of a VISR headtracker
            Some headtrackers only support orientation, others also support positional tracking
            Make sure the useOrientationTracking and usePositionTracking arguments align with the requested headTracker
        headTrackerPositionalArguments: dict
            Dictionairy of positional arguments to initate the supplied VISR headTracker class with
        headTrackerKeywordArguments: dict
            Dictionairy of keyword arguments to initate the supplied VISR headTracker class with
        """
        
        super( RealtimeHOSLoudspeakerDecoder, self ).__init__( context, name, parent )
        
        numLoudspeakers = loudspeakerPos.shape[0]
        self.hosInput = visr.AudioInputFloat( "audioIn", self, HOSOrder+1 )
        self.loudspeakerOutput = visr.AudioOutputFloat( "audioOut", self, numLoudspeakers )

        
        # Check the HOS representation type
        if HOSType.lower() not in ['sine', 'sin', 'cosine', 'cos']:
            raise ValueError(f'Invalid type of HOS representation requested. Must be sine or cosine, supplied {HOSType}')
            
            
        # Decode the HOS format to loudspeaker signals, using Dynamic (head-tracked) HOS panning
        self.HOSLoudspeakerDecoder = HOSLoudspeakerDecoder( context, "HOSLoudspeakerDecoder", self,
                                                              loudspeakerPos,           
                                                              HOSOrder = HOSOrder,        
                                                              HOSType = HOSType,
                                                              beta = beta,
                                                              interpolationSteps = None,
                                                              headOrientation = headOrientation,
                                                              headPosition = headPosition,
                                                              useOrientationTracking = useOrientationTracking,
                                                              usePositionTracking = usePositionTracking,
                                                              useYawOnly = useYawOnly, 
                                                              useDelayCompensation = useDelayCompensation,
                                                              useGainCompensation = useGainCompensation,
                                                              )
        
        # Setup headtracker
        if useOrientationTracking or usePositionTracking:
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
        if useOrientationTracking:
            self.parameterConnection( self.trackingDevice.parameterPort("orientation"), self.HOSLoudspeakerDecoder.parameterPort("tracking"))
        elif (not useOrientationTracking) and usePositionTracking:
            self.parameterConnection( self.trackingDevice.parameterPort("orientation"), self.HOSLoudspeakerDecoder.parameterPort("tracking"))
              
                 
        # Audio Connections
        self.audioConnection( self.hosInput, self.HOSLoudspeakerDecoder.audioPort("audioIn")) # Object-based audio input to HOS renderer
        self.audioConnection( self.HOSLoudspeakerDecoder.audioPort("audioOut"), self.loudspeakerOutput) # HOS loudspeaker signals to the audio output



                    
        