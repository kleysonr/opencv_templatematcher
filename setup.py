# https://python-packaging.readthedocs.io/en/latest/minimal.html
# python3 setup.py register sdist upload

from setuptools import setup, find_packages

setup(name='imagematcher',
      version='0.0.0',
      description='OpenCV Image Matcher',
      url='https://github.com/kleysonr/opencvimagematcher',
      author='Kleyson Rios',
      author_email='kleysonr@gmail.com',
      packages=find_packages(),
      zip_safe=False)