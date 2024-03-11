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
from sys import platform
import serial.tools.list_ports

import visr
import rrl
import audiointerfaces as ai

# from realtime_object_to_hos_loudspeaker_renderer import RealtimeObjectToHOSLoudspeakerRenderer

from hos.realtime import ObjectToHOSLoudspeakerRenderer

# from razor_ahrs_with_udp_calibration_trigger_JH import RazorAHRSWithUdpCalibrationTrigger
# from hdm_tracker_with_udp_calibration_trigger import HdMTrackerWithUdpCalibrationTrigger
# from vive_tracker import ViveTracker

#%% General config

fs = 48000
blockSize = 1024
context = visr.SignalFlowContext(blockSize, fs)

HOSOrder = 1 # order of both encoding and decoding
HOSType = 'Sine' # sine or cosine

sceneReceiveUdpPort = None # None, or UDP port to recieve S3A Scene Metadata

whichTracker = 0 # 0: No Tracking, 1: Razor, 2: HdM, 3: Vive
useYawOnly = True # Use the yaw component of the listener orientation only


beta = 0.0001 # Tikhonov regularisation for the decoder   
useDelayCompensation = True # Adaptive loudspeaker array delay compensation 
useGainCompensation  = False # Adaptive loudspeaker array gain compensation 

#%% Initial Positions

# Sources
srcPos_sph = np.deg2rad(np.stack([[90, 0]], axis=-1)).T
numSrcs = srcPos_sph.shape[0]
    
# Loudspeakers
# az = np.deg2rad( np.linspace(-90,90,5) ) # optimal spkr array pos, semicircle in front of listener
az = np.deg2rad( np.array([-30, 30]) ) # stereo pair
el = np.zeros(az.shape)
r  = np.ones(az.shape) # assume radially equidistant loudspeakers
spkrPos_sph = np.stack([az, el, r], axis=-1)
numSpkrs = spkrPos_sph.shape[0]

# Listener
listenerOrientation = np.array([0,0,0]) # ypr in rads
listenerPosition = np.array([0,0,0]) # xyz in metres

#%% Headtracker

# Find the correct port for the headtracker
# Print all available usb devices using the below
ports = serial.tools.list_ports.comports()
availablePorts = []
print('Available ports:')
for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))
    availablePorts.append(port)


# Setup the specified headtracker
if whichTracker: # as 0 (False) means no tracking
    
    headTrackerCalibrationPort = 8889 # any message sent to this UDP port will zero the orientation
    
    # Razor Tracker
    if whichTracker == 1:
        headTracker = RazorAHRSWithUdpCalibrationTrigger
        
        trackerPort = "COM4" # Windows, USB
        trackerPort = "/dev/cu.usbserial-AJ03GR8O" # Mac, Razor tracker USB
        
        headTrackingKeywordArguments = {'port': port, 'calibrationPort': headTrackerCalibrationPort, 
                                        'displayPos': False,
                                        'yawRightHand': True,'pitchRightHand': False,'rollRightHand': False,}
        useOrientationTracking = True
        usePositionTracking = False
    
    # HdM Tracker
    elif whichTracker == 2:
        headTracker = HdMTrackerWithUdpCalibrationTrigger
        
        trackerPort = "/dev/cu.usbmodem1432201" # Mac, HdM tracker USB
        
        headTrackingKeywordArguments = {'port': port, 'calibrationPort': headTrackerCalibrationPort, 
                                        'displayPos': False,
                                        'yawRightHand': True,'pitchRightHand': False,'rollRightHand': False,}
        useOrientationTracking = True
        usePositionTracking = False
        
    # Vive Tracker
    elif whichTracker == 3:
        headTracker = ViveTracker
        
        trackerPort = 7402 # Windows/Mac, UDP port for Vive Tracker
        
        useOrientationTracking = True
        usePositionTracking = True
        
        headTrackingKeywordArguments = {'port': trackerPort, 'positionTracking': usePositionTracking, 
                                        'displayPos': True }
            
else:
    # No tracking
    headTracker = None
    headTrackerCalibrationPort = None
    headTrackerPositionalArguments = None # Use only keyword arguments
    headTrackerKeywordArguments = {'port': None, 'calibrationPort': headTrackerCalibrationPort }
    useOrientationTracking = False
    usePositionTracking = False
    print('As you did not specify a headTracker, setting head and position tracking to False...')


#%%
if numSpkrs < HOSOrder + 1:
    print('Warning! You can not achieve exact reproduction as you have less loudspeakers than HOSOrder + 1')
    print(f'Number of Loudspeakers: {numSpkrs}')
    print(f'HOSOrder: {HOSOrder}')
    
# renderer = RealtimeObjectToHOSLoudspeakerRenderer(context, "HOSRenderer", None,                                               
#                                                 loudspeakerPos = spkrPos_sph,     
#                                                 numObjects = numSrcs,
#                                                 objectPos = srcPos_sph,    
#                                                 sceneReceiveUdpPort = sceneReceiveUdpPort,  
#                                                 HOSOrder = HOSOrder,
#                                                 HOSType = HOSType,     
#                                                 headOrientation = listenerOrientation,
#                                                 headPosition = listenerPosition,
#                                                 useOrientationTracking = useOrientationTracking,
#                                                 usePositionTracking = usePositionTracking,
#                                                 useYawOnly = useYawOnly, 
#                                                 beta = beta,   
#                                                 useDelayCompensation = useDelayCompensation,
#                                                 useGainCompensation = useGainCompensation,
#                                                 headTracker = headTracker,
#                                                 headTrackerPositionalArguments = headTrackerPositionalArguments,
#                                                 headTrackerKeywordArguments = headTrackerKeywordArguments,
#                                                 )
                

renderer = ObjectToHOSLoudspeakerRenderer( context, "HOSRenderer", None,
                                                          loudspeakerPos = spkrPos_sph,     
                                                          numObjects = numSrcs,
                                                          objectPos = srcPos_sph,    
                                                          sceneReceiveUdpPort = sceneReceiveUdpPort,  
                                                          HOSOrder = HOSOrder,
                                                          HOSType = HOSType,     
                                                          headOrientation = listenerOrientation,
                                                          headPosition = listenerPosition,
                                                          useOrientationTracking = useOrientationTracking,
                                                          usePositionTracking = usePositionTracking,
                                                          useYawOnly = useYawOnly, 
                                                          beta = beta,   
                                                          useDelayCompensation = useDelayCompensation,
                                                          useGainCompensation = useGainCompensation,
                                                          )
    
#%% Configure the audio interface
if platform in ['linux', 'linux2', 'darwin' ]:
    # Either use PortAudio
    audioIfcName = "PortAudio"
    audioIfcCfg = """{ "hostapi": "CoreAudio" }""" # Mac OS X
#    audioIfcCfg = """{ "hostapi": "ALSA" }"""      # Linux

    # Or use Jack on Linux or Mac OS X
#    audioIfcCfg = """{ "clientname": "BstRenderer",
#      "autoconnect" : "false",
#      "portconfig":
#      {
#        "capture":  [{ "basename":"in", "externalport" : {} }],
#        "playback": [{ "basename":"out", "externalport" : {} }]
#      }
#    }"""
#    audioIfcName = "Jack"
    
elif platform in ['windows', 'win32' ]:
    audioIfcCfg = """{ "hostapi": "WASAPI" }"""
#    audioIfcCfg = """{ "hostapi": "ASIO" }"""   # If you have a professional audio interface with an ASIO driver
    audioIfcName = "PortAudio"

#%% Run the renderer
    
result,messages = rrl.checkConnectionIntegrity(renderer)
if not result:
   print(messages)


flow = rrl.AudioSignalFlow( renderer )

aiConfig = ai.AudioInterface.Configuration( flow.numberOfCaptureChannels,
                                           flow.numberOfPlaybackChannels,
                                           fs,
                                           blockSize )

aIfc = ai.AudioInterfaceFactory.create(audioIfcName, aiConfig, audioIfcCfg)

aIfc.registerCallback( flow )

aIfc.start()

print( "Rendering started. Press <q><Return> to quit." )
while( True ):
    i = input( "Press <q><Return> to quit." )
    if i in ['q','Q']:
        break

#aIfc.stop()
#aIfc.unregisterCallback()
#del aIfc
del flow
del renderer

