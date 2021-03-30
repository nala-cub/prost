#!/bin/bash
# Copyright 2021 The PROST Authors
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
# ==============================================================================
# Test repo as a Python Package

set -e
set -x

readonly VENV_DIR=/tmp/prost-env

# Install deps in a virtual env.
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install labtools
python -m pip install -r tools/no_bzl_requirements.txt

# Install all python dependencies
python -m pip install -r tools/requirements.txt 
  

# run a single experiment (no results)
cd src 
python run.py --model_name=openai-gpt