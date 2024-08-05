from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Higher-Order Stereophony toolbox'
LONG_DESCRIPTION = 'Higher-Order Stereophony toolbox for simulation of spatial audio'

# Setting up
setup(
        name="hos", 
        version=VERSION,
        author="Jacob Hollebon",
        author_email="j.hollebon@soton.ac.uk",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
             "numpy",
            "matplotlib",
            "os",
            "warnings",
            "scipy>=1.7.3",
            "sofar>=0.3.1",
            "spharpy>=0.4.1",
        ], 
        keywords=['python', 'hos', 'higher-order stereophony', 'high order stereophony', 'higher-order stereo', 'high order stereo', 'spatial audio', '3D audio']
)