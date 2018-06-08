from setuptools import find_packages
from setuptools import setup

setup(name='keras_gcnn',
      version='1.0',
      description='Group-equivariant layers for Keras',
      author='Bas Veeling',
      author_email='basveeling@gmail.com',
      url='https://github.com/basveeling/keras-gcnn',
      install_requires=['GrouPy'],
      dependency_links=[
          'git+https://github.com/tscohen/GrouPy.git#egg=GrouPy'
      ],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
