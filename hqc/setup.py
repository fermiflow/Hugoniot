"""Setup for pip package."""
import unittest
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'jax',
    'jaxlib',
    'numpy',
    'pyscf'
]

def hqc_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('./test', pattern='test_*.py')
    return test_suite

print(find_packages())

setup(
    name='hqc',
    version='0.1.11',
    description='Quantum chemistry calculations in Hydrogen system.',
    url='https://code.itp.ac.cn/lzh/hydrogen-qc',
    author='lzh',
    author_email='lzh@iphy.ac.cn',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['pytest']},
    platforms=['any'],
    test_suite='setup.hqc_test_suite'
)
