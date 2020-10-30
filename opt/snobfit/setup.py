#!/usr/bin/env python

import codecs, os, re, glob
from setuptools import setup, find_packages
from codecs import open


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# https://packaging.python.org/guides/single-sourcing-package-version/
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='SQSnobFit',
    version=find_version('python', 'SQSnobFit', '_version.py'),
    description='SnobFit - Stable Noisy Optimization by Branch and FIT',
    long_description=long_description,

    url='http://scikit-quant.org/',

    maintainer='Wim Lavrijsen',
    maintainer_email='WLavrijsen@lbl.gov',

    license='other',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',

        'License :: Other/Proprietary License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        'Natural Language :: English'
    ],

    install_requires=['numpy', 'SQCommon'],

    keywords='quantum computing optimization',

    package_dir={'': 'python'},
    packages=find_packages('python', include=['SQSnobFit']),
)
