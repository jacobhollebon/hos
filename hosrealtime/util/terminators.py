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
import pml
import numpy as np

class AudioWell( visr.AtomicComponent ):
    """ Simple component to create an audio output of 0s """
    def __init__( self, context, name, parent, numChs ):
        super( AudioWell, self ).__init__( context, name, parent )
        self.output = visr.AudioOutputFloat( "audioOut", self, width=numChs )

    def process( self ):
        np.asarray(self.output)[...] = 0


class AudioSink(visr.AtomicComponent):
    """ Simple component to terminate audio inputs """
    def __init__(self, context, name, parent, numChs):
        super(AudioSink, self).__init__(context, name, parent)
        self.input = visr.AudioInputFloat("audioIn", self, width=numChs)

    def process(self):
        pass  # Nothing to do


class ParameterSink(visr.AtomicComponent):
    """ Simple component to terminate a parameter input"""
    def __init__(self, context, name, parent, parameterType):
        super(ParameterSink, self).__init__(context, name, parent)
        self.input = visr.ParameterInput("parameterIn", self,
                                          parameterType,
                                          pml.DoubleBufferingProtocol.staticType,
                                          pml.EmptyParameterConfig())
    def process(self):
        pass  # Nothing to do




if __name__ == "__main__":
    fs = 48000
    blockSize = 1024
    context = visr.SignalFlowContext(blockSize, fs)
        

    audioWell = AudioWell(context, "AudioWell", None, 2)
    audioSink = AudioSink(context, "AudioSink", None, 2)
    parameterSink = ParameterSink(context, "ParameterSink", None, pml.ListenerPosition.staticType)