# -*- coding: utf-8 -*-
"""easy to use methods for default CWT and ICWT
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .basis import WaveletBasis


def cwt(tdat, dt=1):
    """Compute continuous wavelet transform, using default ``WaveletBasis``.

    :param tdat:
        shape ``(N,)`` array of real, time domain data
    :param dt:
        sample cadence of data, needed for normalization of transforms

    :return wdat: 
        shape ``(M,N)`` array of complex, wavelet domain data. ``M`` is
        the number of scales used in the transform, and ``N`` is the
        length of the input time domain data.
    
    If you plan on doing several CWTs in the same basis you should
    consider initializing a ``WaveletBasis`` object and using:
    ``WaveletBasis.cwt()``.
    """
    WB = WaveletBasis(N=len(tdat), dt=dt)
    return WB.cwt(tdat)


def icwt(wdat, dt=1):
    """Compute inverse continuous wavelet transform, using default
    ``WaveletBasis``.

    :param wdat:
        shape ``(M,N)`` array of complex, wavelet domain data. ``M`` is
        the number of frequency scales, and ``N`` is the number of time
        samples.
    
    :return tdat:
        shape ``(N,)`` array of real, time domain data
    
    If the forward transform was performed in a different basis, then this
    function will give incorrect output!
    
    If you plan on doing several ICWTs in the same basis you should seriously
    consider initializing a ``WaveletBasis`` object and using:
    ``WaveletBasis.cwt()`` and ``WaveletBasis.icwt()``.
    """
    WB = WaveletBasis(N=wdat.shape[1], dt=dt)
    return WB.icwt(wdat)


def cwtfreq(N, dt=1):
    """Output the Fourier frequencies of the scales used in the default
    ``WaveletBasis``.

    :param N:
        number of time samples in the time domain data.
    :param dt:
        sample cadence of data

    :return freqs:
        shape ``(M,)`` array of frequencies
    """
    WB = WaveletBasis(N=N, dt=dt)
    return WB.freqs
