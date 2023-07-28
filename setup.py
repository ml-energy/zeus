# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

extras_require={
    "lint": ["ruff", "black==22.6.0"],
    "test": ["pytest==7.3.2", "pytest-mock==3.10.0", "pytest-xdist==3.3.1"],
    "torch": ["torch==2.0.1"],
}
extras_require["dev"] = extras_require["lint"] + extras_require["test"] + extras_require["torch"]

setup(
    name="zeus-ml",
    version="0.6.0",
    description="A framework for deep learning energy measurement and optimization.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SymbioticLab/Zeus",
    author="Jae-Won Chung",
    author_email="jwnchung@umich.edu",
    license="Apache-2.0",
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords=["deep-learning", "power", "energy", "sustainability", "mlsys"],
    project_urls={
        "Documentation": "https://ml.energy/zeus",
    },
    packages=find_packages("."),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nvidia-ml-py",
        "pydantic",
    ],
    python_requires=">=3.8",
    extras_require=extras_require,
)
