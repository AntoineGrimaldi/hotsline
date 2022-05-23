
__author__ = "(c) Antoine Grimaldi & Laurent Perrinet INT - CNRS (2020-)"

from setuptools import setup, find_packages

NAME ='hots'
import hots
VERSION = "1.0"

setup(
    name=NAME,
    version=VERSION,
    package_dir={'hots': NAME},
    packages=find_packages(),
    author='Antoine Grimaldi, Institut de Neurosciences de la Timone (CNRS/Aix-Marseille Universite)',
    author_email='antoine.grimaldi@univ-amu.fr',
    url = 'https://github.com/AntoineGrimaldi/hotsline',
    description=' This is a collection of python scripts to do Pattern recognition with an event-based stream',
    long_description=open('README.md').read(),
    license='GNU General Public License v3.0',
    #keywords=('Event based Pattern Recognition', 'Hierarchical model'),
)