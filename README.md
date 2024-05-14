Higher-Order Stereophony
==================

Python toolbox and realtime code for reproducing spatial audio using Higher-Order Stereophony (HOS).

HOS is a generalised extension to classic stereophony, which (in a similar manner to Higher-Order Ambisonics) extends stereo to higher order soundfield reproduction using generalised arrays of loudspeakers. The approach may be applied to soundfield capture and synthesis, soundfield manipulation and reproduction over loudspeaker arrays or binaural. 

HOS reproduces the soundfield accurately over a line defined by the intramural axis (along which lies the listeners ears). Dynamic HOS extends HOS to adapt this line of reproduction based on real-time head tracking of their listener's position.

Currently, this repository focuses on loudspeaker rendering of objects over HOS/Dynamic HOS. As further work is published this will be extended to include binaural rendering and as well as rendering of Higher-Order Ambisonics using HOS. 

For full details of HOS, please see the work listed in citations.


Citation
-----

We kindly ask to acknowledge the use of this software in publications or software.

Jacob Hollebon and Filippo Maria Fazi, “Higher-order stereophony,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2872–2885, 2023



## Folder Structure

The repository is split into two separate python packages. The realtime package has the pre-requisite of VISR, see below for more information.


### HOS

Core python functions required for HOS calculations, loudspeaker gain definitions, geometry handling and simulations.

### HOSrealtime

Code to perform realtime processing of HOS using the python implementation of the Versatile Interactive Realtime Renderer (VISR) framework.

## VISR

VISR is no longer actively supported by the developers, but the source code can be found [here](https://github.com/s3a-spatialaudio/VISR) along with [documentation](https://cvssp.org/data/s3a/public/VISR/visr_installers/0.12.1/macosx/build_py36/doc/userdoc/html/index.html) and a [tutorial](https://github.com/s3a-spatialaudio/visr-tutorial-code). Installers for windows, linux and intel Macs (silicon chips not supported) are available [here](https://cvssp.org/data/s3a/public/VISR/visr_installers/0.12.0/).



## Installation

You may choose to not instal the HOSrealtime package if you do not have a working VISR install.

* ``git clone https://github.com/jacobhollebon/hos/``
* ``cd hos``
* ``pip install .``
* ``cd ..``
* ``cd hosrealtime``
* ``pip install .``


