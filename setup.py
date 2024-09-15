from setuptools import setup

setup(
    name='spit',
    version='0.1',
    description='A Spitting Image: Modular Superpixel Tokenization in Vision Transformers',
    author='Marius Aasan <mariuaas(at)ifi.uio.no>',
    licence='GNUv3',
    packages=['spit'],
    install_requires = [
        'torch >= 2.1.0',
        'torchvision >= 0.16.0',
        'scipy >= 1.10.1',
        'cupy >= 13.0.0',
        'numba >= 0.59.0',
    ]
)