#load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_binary(
	name = "gcc-pr-service",
	srcs = [
		"gcc_pr_service.py",
	],
	main="gcc_pr_service.py",
	visibility = ["//visibility:public"],
	deps = [
		"//compiler_gym/service",
		"//compiler_gym/service/proto",
		"//compiler_gym/service/runtime",
	],
)
