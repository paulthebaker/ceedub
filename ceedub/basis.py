# -*- coding: utf-8 -*-
"""WaveletBasis object containing transform methods
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from scipy.signal import fftconvolve as _fftconv
from .wavelet import MorletWave


class WaveletBasis(object):
    """An object containing a basis for wavelet transforms

    The basis is used for forward and inverse transforms of data using
    the same sample rate and frequency scales.  At initialization given
    ``N``, ``dt``, and ``dj``, the scales will be computed from the
    ``_get_scales()`` method based on the Nyquist period of the wavelet
    and the length of the data.

    See T&C section 3.f for more information about how scales are choosen.
    """
    def __init__(self, wavelet=None, N=None, dt=1, dj=1/16):
        """initialize ``WaveletBasis`` object

        :param wavelet:
            Wavelet basis function which takes two arguments: ``t`` and
            ``s``.  ``t`` is the time to evaluate the wavelet function.
            ``s`` is the scale or width parameter.  The wavelet function
            should be normalized to unit weight at scale=1, and have
            zero mean.
        :param N:
            length of time domain data that will be transformed
        :param dt:
            sample cadence of data, needed for normalization of transforms
        :param dj:
            scale step size, used to determine the frequency scales to use
            for the transform

        Note: the wavelet function used to generate a basis  has different
        requirements than ``scipy.signal.cwt``.  The Ricker and Morlet
        wavelet functions provided in ``scipy.signal`` are incompatible
        with this function.  Instances of the ``MorletWave`` and ``PaulWave``
        callable objects provided in ``ceedub.wavelet`` can be used.
        """
        if wavelet is None:
            wavelet = MorletWave()  # default to Morlet, w0=6
        if not isinstance(N, int):
            raise TypeError("N must be an integer")

        self._wavelet = wavelet
        self._dt = dt
        self._dj = dj
        self._N = N

        self._inv_root_scales = 1./np.sqrt(self.scales)

    # don't provide setters for properties!
    # all are determined at creation and frozen!
    @property
    def wavelet(self):
        """wavelet basis function

        :param t:
            time or array of times to calculate wavelet
        :param s:
            scale parameter for wavelet
        """
        return self._wavelet

    @property
    def dt(self):
        return self._dt

    @property
    def dj(self):
        return self._dj

    @property
    def N(self):
        return self._N

    @property
    def s0(self):
        if not hasattr(self, '_s0'):
            try:
                self._s0 = self.wavelet.nyquist_scale(self.dt)
            except AttributeError:
                self._s0 = 2*self.dt
        return self._s0

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            self._scales = self._get_scales()
        return self._scales

    @property
    def M(self):
        if not hasattr(self, '_M'):
            self._M = len(self.scales)
        return self._M

    @property
    def times(self):
        """sample times of data"""
        if not hasattr(self, '_times'):
            self._times = np.arange(self.N) * self.dt
        return self._times

    @property
    def freqs(self):
        if not hasattr(self, '_freqs'):
            try:
                self._freqs = 1./self.wavelet.fourier_period(self.scales)
            except AttributeError:
                self._freqs = 1./self.scales
        return self._freqs

    def cwt(self, tdat):
        """compute continuous wavelet transfomrm

        :param tdat:
            shape ``(N,)`` array of real, time domain data

        :return wdat:
            shape ``(M,N)`` array of complex, wavelet domain data.
            ``M`` is the number of scales used in the transform, and ``N``
            is the length of the input time domain data.

        Uses the wavelet function and scales of the WaveletBasis. The
        transform is calculated via FFT convolution as described in T&C.
        The FFT convolution is computed once at each wavelet scale,
        determining the frequecny resolution of the output.
        """
        if len(tdat) != self.N:
            raise ValueError("tdat is not length N={:d}".format(self.N))
        dT = self.dt
        rdT = np.sqrt(dT)
        irs = self._inv_root_scales

        wdat = np.zeros((self.M, self.N), dtype=np.complex)

        for ii, s in enumerate(self.scales):
            #try:
            #    L = 10 * self.wavelet.e_fold(s)/dT
            #except AttributeError:
            #    L = 10 * s/dT
            L = 10 * s/dT
            if L > 2*self.N:
                L = 2*self.N
            ts = np.arange(-L/2, L/2) * dT  # generate wavelet data
            norm = rdT * irs[ii]
            wave = norm * self.wavelet(ts, s)
            wdat[ii, :] = _fftconv(tdat, wave, mode='same')

        return wdat

    def icwt(self, wdat):
        """compute the inverse continuous wavelet transform

        :param wdat:
            shape ``(M,N)`` array of complex, wavelet domain data.
            ``M`` is the number of frequency scales, and ``N`` is
            the number of time samples.

        :returns tdat:
            shape ``(N,)`` array of real, time domain data

        The inverse continuous wavelet transform is computed following
        T&C section 3.i.  Uses the wavelet function and scales of the
        parent WaveletBasis.
        """
        if not hasattr(self, '_recon_norm'):
            self._recon_norm = self._get_recon_norm()
        M = self.M
        N = self.N
        if wdat.shape != (M, N):
            raise ValueError("wdat is not shape ({0:d},{1:d})".format(M, N))
        irs = self._inv_root_scales
        tdat = np.einsum('ij,i->j', np.real(wdat), irs)
        tdat *= self._recon_norm
        return tdat

    def _get_scales(self):
        r"""determine the frequency scales used in the transform

        :return scales:
            array of scale parameters, s, for use in ``cwt``

        The frequency scales have log2 frequency spacing. They are
        chosen such that :math:`s_0` is the 'smallest' scale, :math:`dj`
        is the scale step size (in log-space), and :math:`\log_2(N)` is
        the number of octaves.  Traditionally,

        .. math:
            s_j &= s_0 \cdot 2^{j\cdot dj}, \text{ for } j \in [0,J] \\
            J &= \log_2(N) / dj

        If the wavelet used contains a ``nyquist_scale()`` method, :math:`s_0`
        will correspond to the Nyquist frequency.  The largest scale is
        has a frequency given by the observation time: :math:`1/(2 T_{obs}).

        In practice the inverse transforms are more accurate when scales
        outside of the usual Fourier frequencies are used.  We use an extra
        eight octaves, the scales for :math:`j\in [-4, J+4]`.
        """
        N = self.N
        dj = self.dj
        s0 = self.s0
        Noct = np.log2(N)+1  # number of scale octaves
        J = int(Noct / dj)  # total number of scales
        s = [s0 * 2**(j * dj) for j in range(-4, J)]
        return np.array(s)

    def _get_recon_norm(self):
        r"""compute the  normalization factor for ICWT

        This is not :math:`C_\delta` from T&C, this is a normalization
        constant :math:`A`, such that in the ICWT calculation

        .. math: A\, \sum_k \psi(w_k) = tdat.

        This constant eliminates some factors which explicitly cancel in
        later calculations, for example
        :math:`\frac{dj\cdot dt^{1/2}}{\psi_0}`.
        """
        N = self.N
        dt = self.dt
        scales = self.scales
        Psi_f = self.wavelet.freq  # f-domain wavelet as f(w_k, s)
        w_k = 2*np.pi * np.fft.rfftfreq(N, dt)  # Fourier freqs

        W_d = np.zeros_like(scales)
        for ii, sc in enumerate(scales):
            norm = np.sqrt(2*np.pi / dt)
            W_d[ii] = np.sum(Psi_f(w_k, s=sc).conj()) * norm
        W_d /= N
        return 1/np.sum(np.real(W_d))
