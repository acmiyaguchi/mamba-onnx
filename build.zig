const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const backend = b.option(Backend, "backend", "Which backend to build (default: zig)") orelse .zig;

    // Build the selected backend
    const lib = switch (backend) {
        .cpp => buildCpp(b, target, optimize),
        .zig => buildZig(b, target, optimize),
    };
    b.installArtifact(lib);

    // Convenience step to build both backends
    const both_step = b.step("both", "Build both cpp and zig backends");
    const cpp_lib = buildCpp(b, target, optimize);
    const zig_lib = buildZig(b, target, optimize);
    const cpp_install = b.addInstallArtifact(cpp_lib, .{
        .dest_sub_path = "libmamba_ops_cpp.so",
    });
    const zig_install = b.addInstallArtifact(zig_lib, .{
        .dest_sub_path = "libmamba_ops_zig.so",
    });
    both_step.dependOn(&cpp_install.step);
    both_step.dependOn(&zig_install.step);

    // Copy step to put libs in src/mamba_onnx/
    const copy_step = b.step("install-py", "Install to Python package directory");
    const copy_lib = b.addInstallFileWithDir(
        lib.getEmittedBin(),
        .{ .custom = "../src/mamba_onnx" },
        "libmamba_ops.so",
    );
    copy_step.dependOn(&copy_lib.step);
}

const Backend = enum { cpp, zig };

fn buildCpp(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        .pic = true,
    });

    mod.addIncludePath(b.path("vendor/onnxruntime/include/onnxruntime/core/session"));

    mod.addCSourceFile(.{
        .file = b.path("src/native/cpp/selective_scan.cc"),
        .flags = &.{
            "-std=c++17",
            "-mavx2",
            "-mfma",
            "-fno-signed-zeros",
            "-fno-trapping-math",
            // OpenMP disabled - Zig's clang uses libomp (not installed by default)
            // Use Zig backend for parallel execution instead
        },
    });

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "mamba_ops",
        .root_module = mod,
    });

    return lib;
}

fn buildZig(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/native/zig/selective_scan.zig"),
        .target = target,
        .optimize = optimize,
        .pic = true,
        .link_libc = true, // Required for @cImport
    });

    mod.addIncludePath(b.path("vendor/onnxruntime/include/onnxruntime/core/session"));

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "mamba_ops",
        .root_module = mod,
    });

    return lib;
}
