import codecs, glob, os, re, subprocess, sys
from setuptools import setup, find_packages
from distutils import log

setup_requirements = ['wheel']
add_pkg = []
requirements = []

# optimizers sub-package requirements
requirements.append('SQCommon==0.3.2')
requirements.append('SQImFil==0.3.7')
requirements.append('SQSnobFit==0.4.5')

# NOMAD is slow to build, so only add if requested
extras_require = {
    'NOMAD': 'SQNomad==0.2.2',
}

# external optimizers
requirements.append('Py-BOBYQA>=1.2')


here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
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
    name='scikit-quant',
    version=find_version('skquant', '_version.py'),
    description='Integrator for Python-based quantum computing software',
    long_description=long_description,

    url='http://scikit-quant.org',

    maintainer='Wim Lavrijsen',
    maintainer_email='WLavrijsen@lbl.gov',

    license='LBNL BSD',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: C++',

        'Natural Language :: English'
    ],

    setup_requires=setup_requirements,
    install_requires=requirements,

    extras_require=extras_require,

    keywords='quantum computing optimizers',

    package_dir={'': '.'},
    packages=find_packages('.', include=['skquant', 'skquant.opt', 'skquant.interop']),
)
