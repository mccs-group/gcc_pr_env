load("@rules_python//python:defs.bzl", "py_library")

py_library(
	name = "datasets",
	srcs = [
		"__init__.py",
		"cbench.py",
	],
	visibility = ["//visibility:public"],
	deps = [
		"//compiler_gym/datasets",
		"//compiler_gym/service/proto",
		"//compiler_gym/util",
	],
)
