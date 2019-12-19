#!/usr/bin/env python

from setuptools import setup, find_packages

from hyperion import _LIBRARY_VERSION


            
        

setup(name = 'HyperionAPI',
      version = _LIBRARY_VERSION,
      description = 'Public API for Hyperion Instruments from Micron Optics, Inc.',
      author = 'Dustin W. Carr',
      author_email = 'dcarr@micronoptics.com',
      packages = find_packages(exclude=("test","test_*","tests",))
      )
      
