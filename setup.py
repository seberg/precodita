import os
import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



ext_modules=[
    Extension("precodita._dispatcher",
            ["precodita/_dispatcher.pyx"],
            ),
    ]


if __name__ == "__main__":
    setup(
      name="precodita",
      cmdclass={"build_ext": build_ext},
      packages=setuptools.find_packages(),
      ext_modules=ext_modules)

