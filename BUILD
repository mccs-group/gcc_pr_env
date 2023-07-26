load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
	name = "gcc_pr",
	srcs = [
		"__init__.py",
	],
	visibility = ["//visibility:public"],
	deps = [
		"//compiler_gym/service",
		"//compiler_gym/service/proto",
		"//compiler_gym/service/runtime",
        "//compiler_gym/envs/gcc_pr/shuffler:actions_py",
        "//compiler_gym/envs/gcc_pr/datasets",
	],
    data = [
        "//compiler_gym/envs/gcc_pr/service:gcc-pr-service-bin",
    ],
)
