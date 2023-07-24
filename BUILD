load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
	name = "gcc_pr",
	srcs = [
		"__init__.py",
	],
	data = [ "//compiler_gym/envs/gcc_pr/service:gcc-pr-service" ],
	visibility = ["//visibility:public"],
	deps = [
		"//compiler_gym/envs/gcc_pr/datasets",
	],
)
