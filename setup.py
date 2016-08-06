#!/usr/bin/env python

"Setuptools params"

from setuptools import setup, find_packages

VERSION = '0.1'

modname = distname = 'fibte'

setup(
        name=distname,
        version=VERSION,
        description='Loadbalancer tools, and traffic generator',
        author='Ferran Llamas',
        author_email='lferran@student.ethz.ch',
        packages=find_packages(),
        include_package_data = True,
        classifiers=[
                    "License :: OSI Approved :: BSD License",
                    "Programming Language :: Python",
                    "Development Status :: 2 - Pre-Alpha",
                    "Intended Audience :: Developers",
                    "Topic :: System :: Networking",
                    ],
        keywords='networking traffic loadbalancing',
        license='GPLv2',
        install_requires=[
                    'setuptools',
                    'mako',
                    'networkx',
                    'py2-ipaddress'
                ],
        extras_require={
            'draw': ['matplotlib'],
                },
    tests_require=['pytest'],
    setup_requires=['pytest-runner']
    )
