workspace(name = "msc")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# http_archive(
#     name = "bazel_skylib",
#     sha256 = "2e6fa9a61db799266072df115a719a14a9af0e8a630b1f770ef0bd757e68cd71",
#     strip_prefix = "bazel-skylib-de3035d605b4c89a62d6da060188e4ab0c5034b9",
#     urls = ["https://github.com/bazelbuild/bazel-skylib/archive/de3035d605b4c89a62d6da060188e4ab0c5034b9.tar.gz"],
# )

# load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

# bazel_skylib_workspace()

# http_archive(
#     name = "rules_cuda",
#     sha256 = "30e1c474346ac834d44f8ff12a74b41577ae12be4e6094b74e0e9399a3d56758",
#     strip_prefix = "rules_cuda-e714dae5d6781292ffd7f5bd6235ff5cd476dbdb",
#     urls = ["https://github.com/cloudhan/rules_cuda/archive/e714dae5d6781292ffd7f5bd6235ff5cd476dbdb.tar.gz"],
# )

# load("@rules_cuda//cuda:deps.bzl", "register_detected_cuda_toolchains", "rules_cuda_deps")

# rules_cuda_deps()

# register_detected_cuda_toolchains()

# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cuda",
    sha256 = "f80438bee9906e9ecb1a8a4ae2365374ac1e8a283897281a2db2fb7fcf746333",
    strip_prefix = "runtime-b1c7cce21ba4661c17ac72421c6a0e2015e7bef3/third_party/rules_cuda",
    urls = ["https://github.com/tensorflow/runtime/archive/b1c7cce21ba4661c17ac72421c6a0e2015e7bef3.tar.gz"],
)

load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

rules_cuda_dependencies()

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

rules_cc_toolchains()
