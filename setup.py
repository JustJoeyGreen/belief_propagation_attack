from distutils.core import setup
from distutils.extension import Extension
import sys
import numpy

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False
ext = 'pyx' if USE_CYTHON else 'c'
extensions = [Extension("belief_propagation_attack.*",
                        ["belief_propagation_attack/*."+ext],
                        language='c',
                        include_dirs=[numpy.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    ext_modules = extensions
)
