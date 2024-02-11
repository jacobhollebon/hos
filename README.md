Higher-Order Stereophony
==================

Python toolbox and realtime code for reproducing spatial audio using Higher-Order Stereophony (HOS).

HOS is a generalised extension to classic stereophony, which (in a similar manner to Higher-Order Ambisonics) extends stereo to higher order soundfield reproduction using generalised arrays of loudspeakers. The approach may be applied to soundfield capture and synthesis, soundfield manipulation and reproduction over loudspeaker arrays or binaural. 

HOS reproduces the soundfield accurately over a line defined by the intramural axis (along which lies the listeners ears). Dynamic HOS extends HOS to adapt this line of reproduction based on real-time head tracking of their listener's position.

For full details of HOS, please see the work listed in citations.


Citation
-----

We kindly ask to acknowledge the use of this software in publications or software.

Jacob Hollebon and Filippo Maria Fazi, “Higher-order stereophony,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2872–2885, 2023



## Folder Structure

### Toolboxes

Core python functions required for HOS calculations


### VISR

Code to perform realtime processing using the python implementation of the Versatile Interactive Realtime Renderer (VISR) framework.

VISR is no longer actively supported by the developers, but the source code can be found [here](https://github.com/s3a-spatialaudio/VISR) along with [documentation](https://cvssp.org/data/s3a/public/VISR/visr_installers/0.12.1/macosx/build_py36/doc/userdoc/html/index.html) and a [tutorial](https://github.com/s3a-spatialaudio/visr-tutorial-code). Installers for windows, linux and intel Macs (silicon chips not supported) are available [here](https://cvssp.org/data/s3a/public/VISR/visr_installers/0.12.0/).




