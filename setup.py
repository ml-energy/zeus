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
    author="Jae-Won Chung",
    author_email="jwnchung@umich.edu",
    url="https://github.com/SymbioticLab/Zeus",
    version="0.1.0",
    packages=find_packages("."),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "pandas==1.4.2",
        "scikit-learn",
        "pynvml",
    ],
    extras_require={
        "dev": ["pylint==2.14.5", "black==22.6.0", "pydocstyle==6.1.1"],
    }
)
