
__author__ = "(c) Antoine Grimaldi & Laurent Perrinet INT - CNRS (2020-)"

import os
from setuptools import setup, find_packages

NAME ='hots'
import hots
VERSION = "1.0"

setup(
    name=NAME,
    version=VERSION,
    # package source directory
    package_dir={'hots': NAME},
    packages=find_packages(),#exclude=['contrib', 'docs', 'probe']),
    author='Antoine Grimaldi, Institut de Neurosciences de la Timone (CNRS/Aix-Marseille Université)',
    description=' This is a collection of python scripts to do Pattern recognition with of event-based stream',
    long_description=open('README.md').read(),
    license='LICENSE.txt',
    keywords=('Event based Pattern Recognition', 'Hierarchical model'),
    #url = 'https://github.com/VictorBoutin/' + NAME, # use the URL to the github repo
    #download_url = 'https://github.com/VictorBoutin/' + NAME + '/tarball/' + VERSION,
    classifiers = ['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'License :: OSI Approved :: GNU General Public License (GPL)',
               'Operating System :: POSIX',
               'Topic :: Scientific/Engineering',
               'Topic :: Utilities',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.7',
              ],
    extras_require={
                'html' : [
                         'notebook',
                         'matplotlib'
                         'jupyter']
    },
)