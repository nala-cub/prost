# Copyright 2021 The PROST Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

workspace(name = "prost")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

######################
# Tools & Boilerplate.
######################
git_repository(
    name = "labtools",
    # branch = "main",
    commit = "cb39a487b4c15e59cafcc3e1ab1c35299a1efce9",
    remote = "https://github.com/corypaik/labtools",
    shallow_since = "1622415674 -0600",
)

# local_repository(
#     name = "labtools",
#     path = "/home/corypaik/projects/labtools",
# )

load("@labtools//repositories:repositories.bzl", labtools_repos = "repositories")

labtools_repos()

load("@labtools//repositories:deps.bzl", labtools_deps = "deps")

labtools_deps()

##############
# Dependencies
##############

load("@com_github_ali5h_rules_pip//:defs.bzl", "pip_import")

pip_import(
    name = "pip",
    python_interpreter = "python3",
    requirements = "//tools:requirements.txt",
)

load("@pip//:requirements.bzl", "pip_install")

pip_install(["--no-deps"])
