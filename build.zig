const std = @import("std");
const zwin32 = @import("libs/zwin32/build.zig");
const zmath = @import("libs/zmath/build.zig");
const zstbi = @import("libs/zstbi/build.zig");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("zw", "src/main.zig");

    exe.addPackage(zwin32.pkg);
    exe.addPackage(zmath.pkg);
    exe.addPackage(zstbi.pkg);

    const dxc_step = buildShaders(b);
    exe.step.dependOn(dxc_step);

    exe.addIncludePath("libs");
    const imguiDir = "libs/imgui";
    exe.addIncludePath(imguiDir);
    exe.addCSourceFile(imguiDir ++ "/imgui.cpp", &.{""});
    exe.addCSourceFile(imguiDir ++ "/imgui_widgets.cpp", &.{""});
    exe.addCSourceFile(imguiDir ++ "/imgui_tables.cpp", &.{""});
    exe.addCSourceFile(imguiDir ++ "/imgui_draw.cpp", &.{""});
    exe.addCSourceFile(imguiDir ++ "/imgui_demo.cpp", &.{""});
    exe.addCSourceFile(imguiDir ++ "/cimgui.cpp", &.{""});

    exe.linkSystemLibraryName("c");
    exe.linkSystemLibraryName("c++");

    zstbi.link(exe);

    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest("src/main.zig");
    exe_tests.setTarget(target);
    exe_tests.setBuildMode(mode);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&exe_tests.step);
}

fn buildShaders(b: *std.build.Builder) *std.build.Step {
    const dxc_step = b.step("dxc", "Build shaders");

    var dxc_command = makeDxcCmd(
        "color.hlsl",
        "vsMain",
        "color.vs.cso",
        "vs",
        ""
    );
    dxc_step.dependOn(&b.addSystemCommand(&dxc_command).step);

    dxc_command = makeDxcCmd(
        "color.hlsl",
        "psMain",
        "color.ps.cso",
        "ps",
        ""
    );
    dxc_step.dependOn(&b.addSystemCommand(&dxc_command).step);

    dxc_command = makeDxcCmd(
        "texture.hlsl",
        "vsMain",
        "texture.vs.cso",
        "vs",
        ""
    );
    dxc_step.dependOn(&b.addSystemCommand(&dxc_command).step);

    dxc_command = makeDxcCmd(
        "texture.hlsl",
        "psMain",
        "texture.ps.cso",
        "ps",
        ""
    );
    dxc_step.dependOn(&b.addSystemCommand(&dxc_command).step);

    return dxc_step;
}

fn makeDxcCmd(
    comptime input_path: []const u8,
    comptime entry_point: []const u8,
    comptime output_filename: []const u8,
    comptime profile: []const u8,
    comptime define: []const u8,
) [7][]const u8 {
    const shader_ver = "6_0";
    const shader_dir = thisDir() ++ "/src/shaders/";
    return [7][]const u8 {
        "dxc.exe",
        shader_dir ++ input_path,
        "/E " ++ entry_point,
        "/Fo " ++ shader_dir ++ output_filename,
        "/Fc " ++ shader_dir ++ output_filename ++ ".txt",
        "/T " ++ profile ++ "_" ++ shader_ver,
        if (define.len == 0) "" else "/D " ++ define,
    };
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
