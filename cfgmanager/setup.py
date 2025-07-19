"""Setup for pip package."""
import unittest
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'omegaconf',
    'pandas',
]

print(find_packages())

setup(
    name='cfgmanager',
    version='1.0',
    description='Hydra config file manager',
    url='https://code.itp.ac.cn/lzh/cfgmanager',
    author='lzh',
    author_email='lzh@iphy.ac.cn',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['pytest']},
    platforms=['any'],
)
