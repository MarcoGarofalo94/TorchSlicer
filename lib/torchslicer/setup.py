from setuptools import setup, find_packages

requirements = [
    'numpy',
]

setup(
    name='torchslicer',
    version='0.2.0',
    install_requires=requirements,
    packages=find_packages(),
    author='Marco Garofalo',
    author_email='garofalomarco58@gmail.com',
    description='Split learning framework for PyTorch.',
    url='https://github.com/MarcoGarofalo94/TorchSlicer',
)
