from setuptools import setup
from setuptools import setup, find_packages

requirements = [
    # 'torch==2.3.0',
    'numpy',
]
dependency_links = [
    # 'https://download.pytorch.org/whl/cpu'
]

setup(
    name='torchslicer',
    version='0.1',
    install_requires=requirements,
    dependency_links=dependency_links,
    packages=['torchslicer'],
    author='Marco Garofalo',
    author_email='garofalomarco58@gmail.com',
    description='This is a python package that allows you to slice a neural network and train it in a distributed way. It is based on PyTorch.',
    url='https://github.com/MarcoGarofalo94/TorchSlicer',
        classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

