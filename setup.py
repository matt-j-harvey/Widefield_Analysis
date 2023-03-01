from setuptools import setup
from setuptools import find_packages

setup(
    name='widefield',
    version='1.0.0',
    description='Tools for the analysis of widefield calcium imaging data',
    author='Matthew Harvey',
    packages=find_packages(),
    package_data={'': ['*.npy']},
    include_package_data=True,

)

