#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'scipy'
]

setup_requirements = [
    'pytest-runner',
    # TODO(paulthebaker): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='wavelet',
    version='0.1.0',
    description="a minimal continuous wavelet transform module",
    long_description=readme + '\n\n' + history,
    author="Paul T. Baker",
    author_email='paultbaker@gmail.com',
    url='https://github.com/paulthebaker/wavelet',
    packages=find_packages(include=['wavelet']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='wavelet',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
