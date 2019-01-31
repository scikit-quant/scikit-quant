import codecs, glob, os, sys, re
from setuptools import setup, find_packages
from distutils import log

setup_requirements = []
add_pkg = []
requirements = []

# optimizers sub-package requirements
requirements.append('SQImFil>=0.2')

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
    description='Cling-based Python-C++ bindings',
    long_description=long_description,

    url='http://scikit-quant.org',

    maintainer='Wim Lavrijsen',
    maintainer_email='WLavrijsen@lbl.gov',

    license='LBNL BSD',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

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

    keywords='quantum computing optimizers',

    package_dir={'': '.'},
    packages=find_packages('.', include=['skquant', 'skquant.opt']),
)
