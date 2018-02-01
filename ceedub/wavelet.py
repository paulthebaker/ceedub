# -*- coding: utf-8 -*-
"""Wavelet base class and Morlet and Paul wavelet subclasses
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from scipy.special import gamma as _gam

_SQRT2 = np.sqrt(2.)


    """
        """
        """


class MorletWave(object):
    """Morlet-Gabor wavelet: a Gaussian windowed sinusoid
    w0 is the nondimensional frequency constant.  This defines
    the base frequency and width of the mother wavelet. For
    small w0 the wavelets have non-zero mean. T&C set this to
    6 by default.
    If w0 is set to less than 5, the modified Morlet wavelet with
    better low w0 behavior is used instead.
    """

    def __init__(self, w0=6):
        """initialize Morlet-Gabor wavelet with frequency constant w0
        """
        self.w0 = w0
        self._MOD = False
        if(w0 < 5.):
            self._MOD = True

    def __call__(self, *args, **kwargs):
        """default to time domain"""
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Time domain complex Morlet wavelet, centered at zero.

        :param t: time
        :param s: scale factor
        :return psi: value of complex morlet wavelet at time, t

        The wavelets are defined by dimensionless time: x = t/s

        For w0 >= 5, computes the standard Morlet wavelet:
            psi(x) = pi**-0.25 * exp(1j*w0*x) * exp(-0.5*(x**2))

        For w0 < 5, computes the modified Morlet wavelet:
            psi(x) = pi**-0.25 *
                        (exp(1j*w0*x) - exp(-0.5*(x**2))) * exp(-0.5*(x**2))
        """
        t = np.asarray(t)
        w0 = self.w0
        x = t/s

        psi = np.exp(1j * w0 * x)  # base Morlet

        if self._MOD:
            psi -= np.exp(-0.5 * w0**2)  # modified Morlet

        psi *= np.exp(-0.5 * x**2) * np.pi**(-0.25)
        return psi

    def fourier_period(self, s):
        """The Fourier period of the Morlet wavelet with scale, s, given by:
            P = 4*pi*s / (w0 + sqrt(2 + w0**2))
        """
        w0 = self.w0
        return 4*np.pi*s / (w0 + np.sqrt(2 + w0**2))

    def nyquist_scale(self, dt=1):
        """s0 corresponding to the Nyquist period of wavelet
            s0 = 2*dt * (w0 + sqrt(2 + w0**2)) / (4*pi)
        for large w0 this is approximately dt*w0/pi
        """
        w0 = self.w0
        return dt * (w0 + np.sqrt(2 + w0**2)) / (2*np.pi)

    def freq(self, w, s=1.0):
        """
        Frequency domain representation of Morlet wavelet
        Note that the complex Morlet wavelet is real in the frequency domain.
        :param w: frequency
        :param s: scale factor
        :return psi: value of morlet wavelet at frequency, w

        Note there is no support for modified Morlet wavelets. The
        wavelets are defined by dimensionless frequency: y = w*s

        The standard Morlet wavelet is computed as:
            psi(y) = pi**-.25 * H(y) * exp((-(y-w0)**2) / 2)

        where H(y) is the Heaviside step function:
            H(y) = (y > 0) ? 1:0
        """
        w = np.asarray(w)
        H = np.zeros_like(w)  # Heaviside array for vector inputs
        H[w>0] = 1.

        w0 = self.w0
        y = w * s
        return H * np.pi**-.25 * np.exp(0.5 * (-(y - w0)**2))

    def e_fold(self, s):
        """The e-folding time for the Morlet wavelet.
        """
        return _SQRT2 * s


class PaulWave(object):
    """Paul wavelet of order m.
    By definition m is an integer, however in this implementation
    gamma functions are used in place of factorials, so non-integer
    values of m won't cause errors.
    """

    def __init__(self, m=4):
        """ initialize Paul wavelet of order m.
        """
        self.m = m

    def __call__(self, *args, **kwargs):
        """default to time domain"""
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Time domain complex Paul wavelet, centered at zero.
        :param t: time
        :param s: scale factor
        :returns psi: value of complex Paul wavelet at time, t

        The wavelets are defined by dimensionless time: x = t/s

            psi(x) = (2*1j)**m * m! / (pi*(2m)!) * (1 - 1j*x)**-(m+1)
        """
        t = np.asarray(t)
        m = self.m
        x = t / s

        psi = (2**m * 1j**m * _gam(1+m)) / np.sqrt(np.pi * _gam(1 + 2*m))
        psi *= (1 - 1j*x) ** -(m+1)
        return psi

    def fourier_period(self, s):
        """The Fourier period of the Paul wavelet given by:
            P = 4*pi*s / (2*m + 1)
        """
        m = self.m
        return 4*np.pi*s / (2*m + 1)

    def nyquist_scale(self, dt=1):
        """s0 corresponding to the Nyquist period of wavelet
            s0 = 2*dt (2*m + 1)/(4*pi)
        """
        m = self.m
        return dt * (2*m+1)/(2*np.pi)

    def freq(self, w, s=1.0):
        """
        Frequency domain representation of Paul wavelet
        Note that the complex Paul wavelet is real in the frequency domain.
        :param w: frequency
        :param s: scale factor
        :returns psi: value of morlet wavelet at frequency, w

        wavelets are defined by dimensionless frequency: y = w*s

        The Paul wavelet is computed as:
            psi(y) = 2**m / np.sqrt(m * (2*m-1)!) * H(y) * (y)**m * exp(-y)

        where H(y) is the Heaviside step function:
            H(y) = (y > 0) ? 1:0
        """
        w = np.asarray(w)
        H = np.zeros_like(w)  # Heaviside array for vector inputs
        H[w>0] = 1.

        m = self.m
        y = w * s

        psi = H * 2**m / np.sqrt(m * _gam(2*m))
        psi *= y**m * np.exp(-y)
        return psi

    def e_fold(self, s):
        """The e-folding time for the Morlet wavelet.
        """
        return s / _SQRT2
