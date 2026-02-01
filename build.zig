const std = @import("std");

/// Try to locate the LLVM/Clang OpenMP include and lib directories.
/// Returns (include_path, lib_path) or null if not found.
/// Requires libomp-dev: `apt install libomp-dev`
fn findOpenMP() ?struct { include: []const u8, lib: []const u8 } {
    inline for (.{ 20, 19, 18, 17, 16, 15, 14 }) |ver| {
        const s = std.fmt.comptimePrint("{d}", .{ver});
        const include = "/usr/lib/llvm-" ++ s ++ "/lib/clang/" ++ s ++ "/include";
        const lib = "/usr/lib/llvm-" ++ s ++ "/lib";
        if (std.fs.accessAbsolute(include ++ "/omp.h", .{})) |_| {
            return .{ .include = include, .lib = lib };
        } else |_| {}
    }
    return null;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_forkjoin = b.option(bool, "forkjoin", "Use ForkJoin instead of std.Thread.Pool (default: false)") orelse false;

    const ext = switch (target.result.os.tag) {
        .macos => ".dylib",
        .windows => ".dll",
        else => ".so",
    };
    const prefix = if (target.result.os.tag == .windows) "" else "lib";

    // Build both backends by default
    const cpp_lib = buildCpp(b, target, optimize);
    const zig_lib = buildZig(b, target, optimize, use_forkjoin);

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

    // OpenMP: auto-detect LLVM/Clang libomp (apt install libomp-dev)
    const omp = findOpenMP();
    if (omp) |paths| {
        mod.addSystemIncludePath(.{ .cwd_relative = paths.include });
        mod.addLibraryPath(.{ .cwd_relative = paths.lib });
        mod.linkSystemLibrary("omp", .{});
    }

    mod.addCSourceFile(.{
        .file = b.path("src/native/cpp/selective_scan.cc"),
        .flags = if (omp != null) &.{
            "-std=c++17",
            "-mavx2",
            "-mfma",
            "-fno-signed-zeros",
            "-fno-trapping-math",
            "-fopenmp",
        } else &.{
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
    use_forkjoin: bool,
) *std.Build.Step.Compile {
    const options = b.addOptions();
    options.addOption(bool, "use_forkjoin", use_forkjoin);

    const mod = b.createModule(.{
        .root_source_file = b.path("src/native/zig/selective_scan.zig"),
        .target = target,
        .optimize = optimize,
        .pic = true,
        .link_libc = true,
    });
    mod.addOptions("build_options", options);

    mod.addIncludePath(b.path("vendor/onnxruntime/include/onnxruntime/core/session"));

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "mamba_ops_zig",
        .root_module = mod,
    });

    return lib;
}
