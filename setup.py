from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Higher-Order Stereophony toolbox'
LONG_DESCRIPTION = 'Higher-Order Stereophony toolbox for simulation and realtime rendering of spatial audio'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="hos", 
        version=VERSION,
        author="Jacob Hollebon",
        author_email="jacob.hollebon@soto.ac.uk",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
             "numpy",
            "matplotlib",
            "scipy>=1.7.3",
            "sofar==0.3.1",
        ], 
        keywords=['python', 'hos', 'higher-order stereophony', 'high order stereophony', 'higher-order stereo', 'high order stereo', 'spatial audio', '3D audio']
)