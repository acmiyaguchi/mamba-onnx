const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ext = switch (target.result.os.tag) {
        .macos => ".dylib",
        .windows => ".dll",
        else => ".so",
    };
    const prefix = if (target.result.os.tag == .windows) "" else "lib";

    // Build both backends by default
    const cpp_lib = buildCpp(b, target, optimize);
    const zig_lib = buildZig(b, target, optimize);

    const cpp_install = b.addInstallArtifact(cpp_lib, .{
        .dest_sub_path = b.fmt("{s}mamba_ops_cpp{s}", .{ prefix, ext }),
    });
    const zig_install = b.addInstallArtifact(zig_lib, .{
        .dest_sub_path = b.fmt("{s}mamba_ops_zig{s}", .{ prefix, ext }),
    });

    b.getInstallStep().dependOn(&cpp_install.step);
    b.getInstallStep().dependOn(&zig_install.step);

    // Copy step to put libs in build/lib/
    const copy_step = b.step("install-py", "Install to build/lib/ for Python");
    const copy_cpp = b.addInstallFileWithDir(
        cpp_lib.getEmittedBin(),
        .{ .custom = "../build/lib" },
        b.fmt("{s}mamba_ops_cpp{s}", .{ prefix, ext }),
    );
    const copy_zig = b.addInstallFileWithDir(
        zig_lib.getEmittedBin(),
        .{ .custom = "../build/lib" },
        b.fmt("{s}mamba_ops_zig{s}", .{ prefix, ext }),
    );
    copy_step.dependOn(&copy_cpp.step);
    copy_step.dependOn(&copy_zig.step);
}

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
        },
    });

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "mamba_ops_cpp",
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
        .link_libc = true,
    });

    mod.addIncludePath(b.path("vendor/onnxruntime/include/onnxruntime/core/session"));

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "mamba_ops_zig",
        .root_module = mod,
    });

    return lib;
}
