load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")
load("@labtools//jupyter:defs.bzl", "jupyterlab_server")

buildifier(name = "buildifier")

jupyterlab_server(
    name = "jupyterlab",
    deps = [
        "//src:baselines",
        "//src:prost",
        "@labtools//labtools",
        "@pip//:altair",
        "@pip//:altair-saver",
        "@pip//:numpy",
        "@pip//:pandas",
        "@pip//:tables",
    ],
)

exports_files([
    "pytype.cfg",
    "setup.cfg",
])
