# -*- coding: utf-8 -*-
# flake8: noqa ignore=F401
"""Continuous wavelet transform and support functions
based on Torrence and Compo 1998 (T&C)

(http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf)
"""

from .easy import cwt, icwt, cwtfreq
from .wavelet import MorletWave, PaulWave
from .transform import WaveletBasis

__author__ = """Paul T. Baker"""
__email__ = 'paultbaker@gmail.com'
__version__ = '0.1.0'
