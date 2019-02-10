#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:34:57 2018

@author: Mirkazemi
"""
from setuptools import setup

# set the version number
with open('photometryML/_version.py') as f:
    exec(f.read())
    
    
setup_args = dict(
                  name = 'photometryML',
                  version = __version__,
                  author = 'Mohammad Mirkazemi',
                  author_email='mohammad.mirkazemi@gmail.com',
                  url='https://github.com/Mirkazemi/photometryML',
                  description='A toolkit for handling photometric data and pipelines of dataflow in machine learning project.',
                  license = 'GPLv2',
                  install_requires=['numpy', 'scipy'],
                  classifiers=['Intended Audience :: Developers',
                               'Intended Audience :: Science/Research',
                               'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
                               'Operating System :: Unix',
                               'Programming Language :: Python',
                               'Programming Language :: Python :: 3',
                               'Topic :: Scientific/Engineering',
                               'Topic :: Scientific/Engineering :: Astrophysics',],
        platforms=['Unix'],
    )


if __name__ == '__main__':
    setup(**setup_args)
