# -*- coding: utf-8 -*-
"""Wavelet base class and Morlet and Paul wavelet subclasses
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from scipy.special import gamma as _gam

_SQRT2 = np.sqrt(2.)


class Wavelet(object):
    """base class for wavelet objects
    """
    def __call__(self, *args, **kwargs):
        """
        """
        return self.time(*args, **kwargs)


class MorletWave(object):
    """Morlet-Gabor wavelet: a Gaussian windowed sinusoid

    ``w0`` is the nondimensional frequency constant.  This defines
    the base frequency and width of the mother wavelet. For
    small ``w0`` the wavelets have non-zero mean. T&C set ``w0`` to
    6 by default.
    If ``w0`` is set to less than 5, the modified Morlet wavelet with
    better low ``w0`` behavior is used instead.
    """

    def __init__(self, w0=6):
        """initialize Morlet-Gabor wavelet

        :param w0:
            Frequency constant defining width and base frequency of
            mother wavelet.
        """
        self.w0 = w0
        self._MOD = False
        if(w0 < 5.):
            self._MOD = True

    def __call__(self, *args, **kwargs):
        """default to time domain"""
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        r"""
        Time domain complex Morlet wavelet, centered at zero.

        :param t:
            time or list of times to calculate wavelet
        :param s:
            scale factor of wavelet (defines which frequency scale)

        :return psi:
            value of complex morlet wavelet at given times

        The wavelets are defined by dimensionless time: :math:`x = t/s`.
        For :math:`w_0 \ge 5`, the standard Morlet-Gabor wavelet is used,
        and for :math`w_0<5`, the modified Morlet-Gabor is used for better
        stability.

        .. math:: \psi(x) =
                \Biggl \lbrace
                {
                \pi^{-1/4}\, \exp(i\, w_0\, x)\, \exp(-x^2/2),
                    \text{ for }{w_0 \ge 5}
                \atop
                \pi^{-1/4}\,
                    \left[\exp(i\, w_0\, x) - \exp(-x^2/2)\right] \exp(-x^2/2),
                    \text{ for }{w_0 \lt 5}
                }
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
        r"""compute Fourier period of the Morlet wavelet

        :param s:
            scale factor of wavelet

        :return period:
            period of wavelet

        For a wavelet with scale, :math:`s`, the period is:

        .. math:: P = \frac{4\pi\, s}{w_0 + \sqrt{2 + {w_0}^2}}.
        """
        w0 = self.w0
        return 4*np.pi*s / (w0 + np.sqrt(2 + w0**2))

    def nyquist_scale(self, dt=1):
        r"""compute the scale factor corresponding to the Nyquist period

        :param dt:
            sampling time of wavelet basis (time between samples)

        For wavelet basis defined by the mother wavelet with width,
        :math:`w_0`, and a sampling time, :math:`dt`, the Nyquist period
        is the smallest period (largest frequency) which can be resolved.
        For a Morlet wavelet this is

        .. math:: s_0 = 2\,dt\,\frac{w_0 + \sqrt{2 + {w_0}^2}}{4\pi}

        for large :math:`w_0` this is approximately
        :math:`dt\cdot w_0/\pi`.
        """
        w0 = self.w0
        return dt * (w0 + np.sqrt(2 + w0**2)) / (2*np.pi)

    def freq(self, w, s=1.0):
        r"""Frequency domain representation of Morlet wavelet

        :param w:
            frequency or list of frequencies to calculate wavelet
        :param s:
            scale factor

        :return psi:
            value of morlet wavelet at frequency

        The wavelets are defined by dimensionless frequency:
        :math:`y = w\cdot s`.  The complex Morlet wavelet is real
        in the frequency domain.  The standard Morlet wavelet is
        defined in the frequency domain as:

        .. math::
            \psi(y) = \pi^{-1/4}\, \mathcal{H}(y)\,
                \exp\left(-\frac{(y-w_0)^2}{2}\right)

        where :math:`\mathcal{H}(y)` is the Heaviside step function:

        .. math::
            \mathcal{H}(y) =
                \Biggl \lbrace
                {
                1, \text{ for } y \gt 0
                \atop
                0, \text{ for } y \le 0
                }

        There is currently no support for modified Morlet wavelets!
        """
        w = np.asarray(w)
        H = np.zeros_like(w)  # Heaviside array for vector inputs
        H[w>0] = 1.

        w0 = self.w0
        y = w * s
        return H * np.pi**-.25 * np.exp(0.5 * (-(y - w0)**2))

    def e_fold(self, s):
        r"""The e-folding time for the Morlet wavelet.  In dimensionless
        units of time this is :math:`\sqrt{2}\cdot s`.
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
