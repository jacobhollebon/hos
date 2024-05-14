from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Higher-Order Stereophony realtime processing toolbox'
LONG_DESCRIPTION = 'Higher-Order Stereophony toolbox for realtime rendering of spatial audio'

# Setting up
setup(
        name="hosrealtime", 
        version=VERSION,
        author="Jacob Hollebon",
        author_email="j.hollebon@soton.ac.uk",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "matplotlib",
            "scipy>=1.7.3",
            "sofar==0.3.1",
	    "hos",
        ], 
        keywords=['python', 'hos', 'higher-order stereophony', 'high order stereophony', 'higher-order stereo', 'high order stereo', 'spatial audio', '3D audio']
)