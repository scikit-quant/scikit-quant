#!/usr/bin/env python

import codecs, os, sys, glob, re, tempfile
from setuptools import setup, find_packages, Extension
try:
    from numpy.distutils.command.build import build as _build
    from numpy.distutils.command.build_ext import build_ext as _build_ext
    from numpy.distutils.command.config_compiler import config_cc
except ImportError:     # no numpy / no PEP517
    from distutils.command.build import build as _build
    from distutils.command.build_ext import build_ext as _build_ext
from distutils.util import get_platform
from setuptools import Command
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


#
# customized commands
#
class my_build_extension(_build_ext):
    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.warn_error = False

    def build_extension(self, ext):
        ext.extra_compile_args += ['-DUSE_SGTELIB=1']

        if 'linux' in sys.platform:
            if not 'NOOMP' in os.environ:
                ext.extra_compile_args += ['-fopenmp']
                ext.extra_link_args += ['-fopenmp']
            ext.extra_compile_args += ['-Wno-unused-value']
            ext.extra_link_args += ['-Wl,-Bsymbolic-functions', '-Wl,--as-needed']
        elif 'darwin' in sys.platform:
            ext.extra_compile_args += ['-Wno-unused-value', '-Wno-unused-private-field',
                                       '-Wno-overloaded-virtual']

        # adding numpy late to allow setup to install it in the build env
        import numpy
        ext.extra_compile_args += ['-I'+os.path.join(numpy.__path__[0], 'core/include/numpy')]

        # force C++14
        if 'linux' in sys.platform or 'darwin' in sys.platform:
            ext.extra_compile_args += ['-std=c++14']
        elif 'win32' in sys.platform:
            if not 'NOOMP' in os.environ:
                ext.extra_compile_args += ['/openmp']
          # define symbol exports for sgtelib and NOMAD::Clock
            ext.extra_compile_args += ['/std:c++14', '/DDLL_EXPORTS', '/DDLL_UTIL_EXPORTS']

        return _build_ext.build_extension(self, ext)

class my_build_src(Command):           # just needs to exist (used by numpy to build SWIG etc.)
    def initialize_options(self):
        self.build_src = None

    def finalize_options(self):
        if self.build_src is None:
            plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
            self.build_src = os.path.join(tempfile.gettempdir(), 'src'+plat_specifier)
        pass

    def run(self):
        pass

class  my_config_fc(Command):          # dummify as fortran goes unused
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        pass


cmdclass = {
    'build':     _build,
    'build_src': my_build_src,
    'build_ext': my_build_extension }

try:
    cmdclass.update({
        'config_cc' : config_cc,
        'config_fc' : my_config_fc })
except NameError:
    pass      # numpy not available


setup(
    name='SQNomad',
    version=find_version('python', 'SQNomad', '_version.py'),
    description='NOMAD - A blackbox optimization software',
    long_description=long_description,

    url='http://scikit-quant.org/',

    maintainer='Wim Lavrijsen',
    maintainer_email='WLavrijsen@lbl.gov',

    license='LPGL',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',

        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',

        'Natural Language :: English'
    ],

    setup_requires=['wheel', 'numpy'],
    install_requires=['numpy', 'SQCommon'],

    keywords='quantum computing optimization',

    ext_modules=[Extension('libsqnomad',
        sources=glob.glob(os.path.join('src', '*', '*.cpp'))+\
                glob.glob(os.path.join('src', 'Algos', '*', '*.cpp')),
        include_dirs=['src'])],

    package_dir={'': 'python'},
    packages=find_packages('python', include=['SQNomad']),

    cmdclass=cmdclass,

    zip_safe=False,
)
