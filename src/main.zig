const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const zr = @import("zr.zig");

const color_vs = @embedFile("shaders/color.vs.cso");
const color_ps = @embedFile("shaders/color.ps.cso");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var fw = try zr.Fw.init(allocator, zr.Fw.Options {
        .enable_debug_layer = true
    });
    defer fw.deinit(allocator);

    var device = fw.getDevice();
    var pipeline_pool = try zr.ObjectPool(zr.Pipeline).init(allocator);
    errdefer pipeline_pool.deinit();
    var pipeline_cache = zr.Pipeline.Cache.init(allocator);
    errdefer pipeline_cache.deinit();

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

    var sha: [zr.Pipeline.sha_length]u8 = undefined;
    zr.Pipeline.getGraphicsPipelineSha(&pso_desc, &sha);
    var pipeline_handle = zr.ObjectHandle.invalid();
    if (pipeline_cache.get(&sha)) |h| {
        pipeline_handle = h;
    } else {
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
        var signature: *d3d.IBlob = undefined;
        try zwin32.hrErrorOnFail(d3d12.D3D12SerializeVersionedRootSignature(&rs_desc,
                                                                            @ptrCast(*?*d3d.IBlob, &signature),
                                                                            null));
        defer _ = signature.Release();

        var rs: *d3d12.IRootSignature = undefined;
        try zwin32.hrErrorOnFail(device.CreateRootSignature(0,
                                                            signature.GetBufferPointer(),
                                                            signature.GetBufferSize(),
                                                            &d3d12.IID_IRootSignature,
                                                            @ptrCast(*?*anyopaque, &rs)));
        pso_desc.pRootSignature = rs;

        var pso: *d3d12.IPipelineState = undefined;
        try zwin32.hrErrorOnFail(device.CreateGraphicsPipelineState(&pso_desc,
                                                                    &d3d12.IID_IPipelineState,
                                                                    @ptrCast(*?*anyopaque, &pso)));

        pipeline_handle = try zr.Pipeline.addToPool(&pipeline_pool, pso, rs, zr.Pipeline.Type.Graphics);
        try pipeline_cache.add(&sha, pipeline_handle);
    }

    while (zr.Fw.handleWindowEvents()) {
        try fw.beginFrame();

        const c = fw.getCommandList();
        const rt_cpu_handle = fw.getBackBufferCpuDescriptorHandle();
        c.OMSetRenderTargets(1, &[_]d3d12.CPU_DESCRIPTOR_HANDLE { rt_cpu_handle }, w32.TRUE, null);
        c.ClearRenderTargetView(rt_cpu_handle, &[4]f32 { 0.0, 1.0, 0.0, 1.0 }, 0, null);

        // var psodesc = std.mem.zeroes(d3d12.GRAPHICS_PIPELINE_STATE_DESC);
        // {
        //     var sha: [Pipeline.sha_length]u8 = undefined;
        //     Pipeline.getGraphicsPipelineSha(&psodesc, &sha);
        //     var h = std.mem.zeroes(ObjectHandle);
        //     try p.add(&sha, h);
        //     var csdesc = std.mem.zeroes(d3d12.COMPUTE_PIPELINE_STATE_DESC);
        //     Pipeline.getComputePipelineSha(&csdesc, &sha);
        // }
        // {
        //     var sha: [Pipeline.sha_length]u8 = undefined;
        //     Pipeline.getGraphicsPipelineSha(&psodesc, &sha);
        //     p.remove(&sha);
        // }
        try fw.endFrame();
    }

    std.debug.print("Exiting\n", .{});

    pipeline_cache.deinit();
    pipeline_pool.deinit();
}
