#!/usr/bin/env python3

import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="fastnode2vec",
    version="0.0.7",
    author="Louis Abraham",
    license="MIT",
    author_email="louis.abraham@yahoo.fr",
    description="Fast implementation of node2vec",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/louisabraham/fastnode2vec",
    packages=["fastnode2vec"],
    install_requires=["numpy", "numba", "gensim", "click", "tqdm"],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["fastnode2vec = fastnode2vec.cli:node2vec"]},
    classifiers=["Topic :: Scientific/Engineering :: Artificial Intelligence"],
)
