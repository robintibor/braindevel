import os
from setuptools import find_packages
from setuptools import setup

version = '0.0.1'

here = os.path.abspath(os.path.dirname(__file__))


setup(
    name="Braindecode",
    version=version,
    description="A library to decode brain signals, for now from EEG.",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Brain State Decoding",
        ],
    keywords="",
    author="Robin Tibor Schirrmeister",
    author_email="robintibor@googlegroups.com",
    url="https://github.com/robintibor/braindecode",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    )
