#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ceedub` package."""

from __future__ import division, print_function

import pytest

import numpy as np
import ceedub as cw


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_Morlet():
    """test Morlet agains T&C table 2
    Psi_t(0) = pi^(-1/4)
    Psi_f(0) = 0
    """
    morl = cw.MorletWave()
    assert(np.isclose(morl(0), np.pi**(-1/4), atol=1.e-12))
    assert(np.isclose(morl.freq(0), 0, atol=1.e-12))
