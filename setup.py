# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
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

setup(
    name="zeus-ml",
    version="0.2.1",
    description="An Energy Optimization Framework for DNN Training",
    long_description="# Zeus: An Energy Optimization Framework for DNN Training\n",
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
    keywords=["deep-learning", "power", "energy", "mlsys"],
    project_urls={
        "Documentation": "https://ml.energy/zeus",
    },
    packages=find_packages("."),
    install_requires=[
        "torch",
        "numpy",
        "pandas==1.4.2",
        "scikit-learn",
        "pynvml",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": ["pylint==2.14.5", "black==22.6.0", "pydocstyle==6.1.1"],
    }
)
