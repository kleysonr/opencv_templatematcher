# https://python-packaging.readthedocs.io/en/latest/minimal.html
# python3 setup.py register sdist upload

from setuptools import setup, find_packages

setup(name='templatematcher',
      version='0.0.0',
      description='OpenCV Template Matcher',
      url='https://github.com/kleysonr/opencv_templatematcher',
      author='Kleyson Rios',
      author_email='kleysonr@gmail.com',
      packages=find_packages(),
      zip_safe=False)