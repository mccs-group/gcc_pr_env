filegroup(
	name = "gcc-pr-service-files",
	srcs = [
		"gcc_pr_service.py",
	],
    visibility = ["//visibility:public"],
)

genrule(
    name = "gcc-pr-service-bin",
    srcs = [
        ":gcc-pr-service-files",
    ],
    outs = [
        "gcc-pr-service",
    ],
    cmd = "cp $(location :gcc-pr-service-files) $@ && " +
        "chmod 666 $@",
    visibility = ["//visibility:public"],
)
