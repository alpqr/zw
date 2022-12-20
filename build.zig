const std = @import("std");
const zwin32 = @import("libs/zwin32/build.zig");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("zw", "src/main.zig");

    exe.addPackage(zwin32.pkg);

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
