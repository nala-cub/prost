# Description: PROST Dataset and Baselines

load(
    "@labtools//pytype:defs.bzl",
    "pytype_binary",
    "pytype_library",
    "pytype_test",
)

package(default_visibility = ["//visibility:private"])

pytype_library(
    name = "prost",
    srcs = glob(
        ["prost/*.py"],
        exclude = ["prost/*_test.py"],
    ),
    imports = ["."],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pip//:datasets",
        "@pip//:dm-tree",
        "@pip//:icecream",
        "@pip//:pandas",
        "@pip//:pyyaml",
        "@pip//:toolz",
    ],
)

pytype_library(
    name = "baselines",
    srcs = glob(
        ["baselines/*.py"],
        exclude = ["baselines/*_test.py"],
    ),
    imports = ["."],
    visibility = ["//:__subpackages__"],
    deps = [
        ":prost",
        "@labtools//labtools",
        "@pip//:absl-py",
        "@pip//:datasets",
        "@pip//:einops",
        "@pip//:icecream",
        "@pip//:numpy",
        "@pip//:pandas",
        "@pip//:pyyaml",
        "@pip//:regex",
        "@pip//:sentencepiece",
        "@pip//:tabulate",
        "@pip//:tokenizers",
        "@pip//:toolz",
        "@pip//:torch",
        "@pip//:transformers",
        "@pip//:dm-tree",
        "@pip//:ml_collections",
        # results
        "@pip//:kaleido",
        "@pip//:plotly",
        "@pip//:tables",
        "@pip//:altair",
        "@pip//:altair-saver",
        "@pip//:google-cloud-storage",
    ],
)

pytype_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":baselines"],
)

pytype_test(
    name = "input_pipeline_test",
    srcs = ["baselines/input_pipeline_test.py"],
    env = {
        "HF_DATASETS_CACHE": "/tmp/input_pipeline_test/hf_cache",
    },
    tags = ["manual"],
    deps = [":baselines"],
)

pytype_test(
    name = "prost_test",
    srcs = ["prost/prost_test.py"],
    data = ["//data:prost"],
    deps = [
        ":prost",
        "@pip//:absl-py",
        "@pip//:icecream",
    ],
)
