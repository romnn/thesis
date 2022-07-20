# load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "kernel",
    srcs = ["mm.cu"],
    hdrs = ["mm.h"],
)

cc_binary(
    name = "main",
    srcs = ["mm.cc"],
    deps = [":kernel"],
)
