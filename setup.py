from setuptools import setup, find_packages


setup(
    name='prep_prescription',
    description='PrEP Prescription',
    version='0.1.0',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'ray',
        'scipy',
        'xarray'
    ],
)