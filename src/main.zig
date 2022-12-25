const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const zr = @import("zr.zig");

const color_vs = @embedFile("shaders/color.vs.cso");
const color_ps = @embedFile("shaders/color.ps.cso");

fn create_pipeline(fw: *zr.Fw) !zr.ObjectHandle {
    const input_element_descs = [_]d3d12.INPUT_ELEMENT_DESC {
        d3d12.INPUT_ELEMENT_DESC {
            .SemanticName = "POSITION",
            .SemanticIndex = 0,
            .Format = .R32G32B32_FLOAT,
            .InputSlot = 0,
            .AlignedByteOffset = 0,
            .InputSlotClass = .PER_VERTEX_DATA,
            .InstanceDataStepRate = 0
        },
        d3d12.INPUT_ELEMENT_DESC {
            .SemanticName = "TEXCOORD",
            .SemanticIndex = 0,
            .Format = .R32G32B32_FLOAT,
            .InputSlot = 0,
            .AlignedByteOffset = 3 * @sizeOf(f32),
            .InputSlotClass = .PER_VERTEX_DATA,
            .InstanceDataStepRate = 0
        }
    };
    var pso_desc = std.mem.zeroes(d3d12.GRAPHICS_PIPELINE_STATE_DESC);
    pso_desc.InputLayout = .{
        .pInputElementDescs = &input_element_descs,
        .NumElements = input_element_descs.len
    };
    pso_desc.VS = .{
        .pShaderBytecode = color_vs,
        .BytecodeLength = color_vs.len
    };
    pso_desc.PS = .{
        .pShaderBytecode = color_ps,
        .BytecodeLength = color_ps.len
    };
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    pso_desc.RasterizerState.FillMode = .SOLID;
    pso_desc.RasterizerState.CullMode = .BACK;
    pso_desc.PrimitiveTopologyType = .TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
    pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

    const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
        .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_0,
        .u = .{
            .Desc_1_0 = .{
                .NumParameters = 1,
                .pParameters = &[_]d3d12.ROOT_PARAMETER {
                    .{
                        .ParameterType = .CBV,
                        .u = .{
                            .Descriptor = .{
                                .ShaderRegister = 0,
                                .RegisterSpace = 0
                            }
                        },
                        .ShaderVisibility = .ALL
                    }
                },
                .NumStaticSamplers = 0,
                .pStaticSamplers = null,
                .Flags = d3d12.ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            }
        }
    };

    return try fw.lookupOrCreateGraphicsPipeline(&pso_desc, &rs_desc);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var fw = try zr.Fw.init(allocator, zr.Fw.Options {
        .enable_debug_layer = true
    });
    defer fw.deinit(allocator);

    const pipeline_handle = try create_pipeline(&fw);
    var vbuf = try fw.createBuffer(.DEFAULT, 64);

    // var s = try zr.StagingArea.init(fw.getDevice(), 1024, .READBACK);
    // s.reset();
    // _ = s.getBuffer();
    // std.debug.print("{any}\n", .{s.allocate(32)});
    // std.debug.print("{any}\n", .{s.allocate(32)});
    while (zr.Fw.handleWindowEvents()) {
        try fw.beginFrame();
        const vbuf_res = fw.getResourcePool().lookupRef(vbuf).?;
        _ = fw.getCurrentSmallUploadStagingArea().allocate(@intCast(u32, vbuf_res.desc.Width)).?;

        const c = fw.getCommandList();
        const rt_cpu_handle = fw.getBackBufferCpuDescriptorHandle();
        c.OMSetRenderTargets(1, &[_]d3d12.CPU_DESCRIPTOR_HANDLE { rt_cpu_handle }, w32.TRUE, null);
        c.ClearRenderTargetView(rt_cpu_handle, &[4]f32 { 0.0, 1.0, 0.0, 1.0 }, 0, null);

        fw.setPipeline(pipeline_handle);
        c.IASetPrimitiveTopology(.TRIANGLELIST);

        try fw.endFrame();
    }
//    s.deinit();
    std.debug.print("Exiting\n", .{});
}
