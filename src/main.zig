const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const zm = @import("zmath");
const zr = @import("zr.zig");

const color_vs = @embedFile("shaders/color.vs.cso");
const color_ps = @embedFile("shaders/color.ps.cso");

const Vertex = struct {
    position: [3]f32,
    color: [3]f32
};
comptime { std.debug.assert(@sizeOf([2]Vertex) == 48); }

const vertices = [_]Vertex {
    .{
        .position = [3]f32 { -1.0, -1.0, 0.0 },
        .color = [3]f32 { 1.0, 0.0, 0.0 }
    },
    .{
        .position = [3]f32 { 1.0, -1.0, 0.0 },
        .color = [3]f32 { 0.0, 1.0, 0.0 }
    },
    .{
        .position = [3]f32 { 0.0, 1.0, 0.0 },
        .color = [3]f32 { 0.0, 0.0, 1.0 }
    }
};

const CbData = struct {
    mvp: [16]f32
};
comptime { std.debug.assert(@sizeOf(CbData) == 64); }

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
    pso_desc.SampleMask = 0xFFFFFFFF;
    pso_desc.RasterizerState.FillMode = .SOLID;
    pso_desc.RasterizerState.CullMode = .NONE; // .BACK
    pso_desc.RasterizerState.FrontCounterClockwise = w32.TRUE;
    pso_desc.PrimitiveTopologyType = .TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
    pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

    const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
        .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_1,
        .u = .{
            .Desc_1_1 = .{
                .NumParameters = 1,
                .pParameters = &[_]d3d12.ROOT_PARAMETER1 {
                    .{
                        // .ParameterType = .CBV,
                        // .u = .{
                        //     .Descriptor = .{
                        //         .ShaderRegister = 0,
                        //         .RegisterSpace = 0,
                        //         .Flags = d3d12.ROOT_DESCRIPTOR_FLAG_NONE
                        //     }
                        // },
                        .ParameterType = .DESCRIPTOR_TABLE,
                        .u = .{
                            .DescriptorTable = .{
                                .NumDescriptorRanges = 1,
                                .pDescriptorRanges = &[_]d3d12.DESCRIPTOR_RANGE1 {
                                    .{
                                        .RangeType = .CBV,
                                        .NumDescriptors = 1,
                                        .BaseShaderRegister = 0,
                                        .RegisterSpace = 0,
                                        .Flags = d3d12.DESCRIPTOR_RANGE_FLAG_NONE,
                                        .OffsetInDescriptorsFromTableStart = 0
                                    }
                                }
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

    const device = fw.getDevice();
    const resource_pool = fw.getResourcePool();

    var cbv_srv_uav_cpu_descriptor_pool = try zr.CpuDescriptorPool.init(allocator,
                                                                        device,
                                                                        .CBV_SRV_UAV);

    const pipeline_handle = try create_pipeline(&fw);
    var vbuf = try fw.createBuffer(.DEFAULT, 3 * @sizeOf(Vertex));
    var ibuf = try fw.createBuffer(.DEFAULT, 3 * @sizeOf(u16));
    var needs_upload = true;
    var rotation: f32 = 0.0;
    var last_size_used_for_projection = zr.Size.empty();
    var projection = zm.identity();

    const cbuf_size = zr.alignedSize(@sizeOf(CbData), 256);
    var cbuf = try fw.createBuffer(.UPLOAD, cbuf_size);
    var cbuf_p = try fw.mapBuffer(u8, cbuf);
    var cbv_cpu_descriptor = try cbv_srv_uav_cpu_descriptor_pool.allocate(1);
    device.CreateConstantBufferView(
        &.{
            .BufferLocation = resource_pool.lookupRef(cbuf).?.resource.GetGPUVirtualAddress(),
            .SizeInBytes = cbuf_size
        },
        cbv_cpu_descriptor.cpu_handle);

    while (zr.Fw.handleWindowEvents()) {
        if (try fw.beginFrame() != zr.Fw.BeginFrameResult.success) {
            continue;
        }
        const cmd_list = fw.getCommandList();
        const staging = fw.getCurrentSmallStagingArea();
        const shader_visible_cbv_srv_uav = fw.getCurrentShaderVisibleDescriptorHeapRange();
        const output_pixel_size = fw.getBackBufferPixelSize();

        if (!std.meta.eql(output_pixel_size, last_size_used_for_projection)) {
            last_size_used_for_projection = output_pixel_size;
            projection = zm.perspectiveFovLh(45.0,
                                             @intToFloat(f32, output_pixel_size.width) / @intToFloat(f32, output_pixel_size.height),
                                             0.01,
                                             1000.0);
        }

        if (needs_upload) {
            needs_upload = false;
            fw.addTransitionBarrier(vbuf, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(ibuf, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.recordTransitionBarriers();
            {
                const byte_size = 3 * @sizeOf(Vertex);
                const alloc = staging.allocate(byte_size).?;
                std.mem.copy(Vertex, alloc.castCpuSlice(Vertex), &vertices);
                cmd_list.CopyBufferRegion(resource_pool.lookupRef(vbuf).?.resource, 0, alloc.buffer, alloc.buffer_offset, byte_size);
            }
            {
                const byte_size = 3 * @sizeOf(u16);
                const alloc = staging.allocate(byte_size).?;
                var idata = alloc.castCpuSlice(u16);
                idata[0] = 0;
                idata[1] = 1;
                idata[2] = 2;
                cmd_list.CopyBufferRegion(resource_pool.lookupRef(ibuf).?.resource, 0, alloc.buffer, alloc.buffer_offset, byte_size);
            }
            fw.addTransitionBarrier(vbuf, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(ibuf, d3d12.RESOURCE_STATE_INDEX_BUFFER);
            fw.recordTransitionBarriers();
        }

        //var cbuf_area = staging.allocate(@sizeOf(CbData)).?;
        //zm.storeMat(cbuf_area.castCpuSlice(f32), zm.transpose(mvp));

        var cb_data: CbData = undefined;
        const model = zm.rotationY(rotation);
        const view = zm.translation(0.0, 0.0, 5.0);
        const modelview = zm.mul(model, view);
        const mvp = zm.mul(modelview, projection);

        // Mat is stored as row major, transpose to get column major
        zm.storeMat(&cb_data.mvp, zm.transpose(mvp));
        std.mem.copy(u8, cbuf_p, std.mem.asBytes(&cb_data));

        rotation += 0.05;

        const rt_cpu_handle = fw.getBackBufferCpuDescriptorHandle();
        cmd_list.OMSetRenderTargets(1, &[_]d3d12.CPU_DESCRIPTOR_HANDLE { rt_cpu_handle }, w32.TRUE, null);
        cmd_list.ClearRenderTargetView(rt_cpu_handle, &[4]f32 { 0.4, 0.7, 0.0, 1.0 }, 0, null);

        cmd_list.RSSetViewports(1, &[_]d3d12.VIEWPORT {
            .{
                .TopLeftX = 0.0,
                .TopLeftY = 0.0,
                .Width = @intToFloat(f32, output_pixel_size.width),
                .Height = @intToFloat(f32, output_pixel_size.height),
                .MinDepth = 0.0,
                .MaxDepth = 1.0
            }
        });
        cmd_list.RSSetScissorRects(1, &[_]d3d12.RECT {
            .{
                .left = 0,
                .top = 0,
                .right = @intCast(i32, output_pixel_size.width),
                .bottom = @intCast(i32, output_pixel_size.height)
            }
        });

        fw.setPipeline(pipeline_handle);

        cmd_list.IASetPrimitiveTopology(.TRIANGLELIST);
        cmd_list.IASetVertexBuffers(0, 1, &[_]d3d12.VERTEX_BUFFER_VIEW {
            .{
                .BufferLocation = resource_pool.lookupRef(vbuf).?.resource.GetGPUVirtualAddress(),
                .SizeInBytes = 3 * @sizeOf(Vertex),
                .StrideInBytes = @sizeOf(Vertex),
            }
        });
        cmd_list.IASetIndexBuffer(&.{
            .BufferLocation = resource_pool.lookupRef(ibuf).?.resource.GetGPUVirtualAddress(),
            .SizeInBytes = 3 * @sizeOf(u16),
            .Format = .R16_UINT,
        });

        //cmd_list.SetGraphicsRootConstantBufferView(0, cbuf_area.gpu_addr);
        cmd_list.SetDescriptorHeaps(1, &[_]*d3d12.IDescriptorHeap {
            fw.getShaderVisibleDescriptorHeap()
        });
        device.CopyDescriptorsSimple(1, shader_visible_cbv_srv_uav.get(1).cpu_handle, cbv_cpu_descriptor.cpu_handle, .CBV_SRV_UAV);
        cmd_list.SetGraphicsRootDescriptorTable(0, shader_visible_cbv_srv_uav.at(0).gpu_handle);

        cmd_list.DrawIndexedInstanced(3, 1, 0, 0, 0);

        try fw.endFrame();
    }

    std.debug.print("Exiting\n", .{});

    cbv_srv_uav_cpu_descriptor_pool.deinit();
}
