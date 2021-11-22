import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="naivenlp",
    version="0.0.9",
    description="Transformers based toolkit for modeling NLP tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naivenlp/naivenlp",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    # include_package_data=True,
    package_data={

    },
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch-lightning",
    ],
    dependency_links=[

    ],
    extras_require={

    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    )
)
