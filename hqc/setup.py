"""Setup for pip package."""
import unittest
from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

REQUIRED_PACKAGES = [
    'jax>=0.4.0',
    'jaxlib>=0.4.0',
    'numpy>=1.20.0',
    'pyscf>=2.0.0'
]

def hqc_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('./test', pattern='test_*.py')
    return test_suite

setup(
    name='hqc',
    version='0.1.11',
    description='Quantum chemistry calculations for Hydrogen systems using JAX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://code.itp.ac.cn/lzh/hydrogen-qc',
    author='Zihang Li',
    author_email='lzh@iphy.ac.cn',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='quantum-chemistry hartree-fock dft jax hydrogen periodic-boundary-conditions',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        'testing': ['pytest>=7.0.0'],
    },
    python_requires='>=3.9',
    platforms=['any'],
    test_suite='setup.hqc_test_suite',
    include_package_data=True,
    project_urls={
        'Bug Reports': 'https://code.itp.ac.cn/lzh/hydrogen-qc/issues',
        'Source': 'https://code.itp.ac.cn/lzh/hydrogen-qc',
    },
)
