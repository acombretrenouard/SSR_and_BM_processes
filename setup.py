#!/usr/bin/env python

from setuptools import setup

setup(name='markovian_simulator',
      version='1.0',
      description='Simulates markovian dynamics',
      author='Antoine Combret--Renouard',
      author_email='antoine2031@gmail.com',
      packages=['sample'],
      install_requires=['numpy','matplotlib','threading'],
     )