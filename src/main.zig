const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const zr = @import("zr.zig");
const zm = zr.zm;
const zstbi = zr.zstbi;
const zmesh = zr.zmesh;
const imgui = zr.imgui;

const simple_vs = @embedFile("shaders/simple.vs.cso");
const simple_ps = @embedFile("shaders/simple.ps.cso");
const color_vs = @embedFile("shaders/color.vs.cso");
const color_ps = @embedFile("shaders/color.ps.cso");
const texture_vs = @embedFile("shaders/texture.vs.cso");
const texture_ps = @embedFile("shaders/texture.ps.cso");
const gltf_vs = @embedFile("shaders/gltf.vs.cso");
const gltf_ps = @embedFile("shaders/gltf.ps.cso");

const VertexWithColor = struct {
    position: [3]f32,
    color: [3]f32
};
comptime { std.debug.assert(@sizeOf([2]VertexWithColor) == 48); }

const vertices_with_color = [_]VertexWithColor {
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

const VertexWithUv = struct {
    position: [3]f32,
    uv: [2]f32
};
comptime { std.debug.assert(@sizeOf([2]VertexWithUv) == 40); }

const vertices_with_uv = [_]VertexWithUv {
    .{
        .position = [3]f32 { -1.0, -1.0, 0.0 },
        .uv = [2]f32 { 0.0, 1.0 }
    },
    .{
        .position = [3]f32 { 1.0, -1.0, 0.0 },
        .uv = [2]f32 { 1.0, 1.0 }
    },
    .{
        .position = [3]f32 { 0.0, 1.0, 0.0 },
        .uv = [2]f32 { 0.5, 0 }
    }
};

const SimpleCbData = struct {
    mvp: [16]f32,
    color: [4]f32
};
comptime { std.debug.assert(@sizeOf(SimpleCbData) == 80); }

fn createSimplePipeline(fw: *zr.Fw) !zr.ObjectHandle {
    const input_element_descs = [_]d3d12.INPUT_ELEMENT_DESC {
        d3d12.INPUT_ELEMENT_DESC {
            .SemanticName = "POSITION",
            .SemanticIndex = 0,
            .Format = .R32G32B32_FLOAT,
            .InputSlot = 0,
            .AlignedByteOffset = 0,
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
        .pShaderBytecode = simple_vs,
        .BytecodeLength = simple_vs.len
    };
    pso_desc.PS = .{
        .pShaderBytecode = simple_ps,
        .BytecodeLength = simple_ps.len
    };
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    pso_desc.SampleMask = 0xFFFFFFFF;
    pso_desc.RasterizerState.FillMode = .WIREFRAME;
    pso_desc.RasterizerState.CullMode = .NONE;
    pso_desc.DepthStencilState.DepthEnable = w32.TRUE;
    pso_desc.DepthStencilState.DepthWriteMask = .ALL;
    pso_desc.DepthStencilState.DepthFunc = .LESS;
    pso_desc.PrimitiveTopologyType = .TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
    pso_desc.DSVFormat = zr.Fw.dsv_format;
    pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

    const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
        .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_1,
        .u = .{
            .Desc_1_1 = .{
                .NumParameters = 1,
                .pParameters = &[_]d3d12.ROOT_PARAMETER1 {
                    .{
                        .ParameterType = .CBV,
                        .u = .{
                            .Descriptor = .{
                                .ShaderRegister = 0,
                                .RegisterSpace = 0,
                                .Flags = d3d12.ROOT_DESCRIPTOR_FLAG_NONE
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

    return try fw.lookupOrCreatePipeline(&pso_desc, null, &rs_desc);
}

const CbData = struct {
    mvp: [16]f32
};
comptime { std.debug.assert(@sizeOf(CbData) == 64); }

fn createColorPipeline(fw: *zr.Fw) !zr.ObjectHandle {
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
    pso_desc.RasterizerState.CullMode = .NONE;
    pso_desc.RasterizerState.FrontCounterClockwise = w32.TRUE;
    pso_desc.DepthStencilState.DepthEnable = w32.TRUE;
    pso_desc.DepthStencilState.DepthWriteMask = .ALL;
    pso_desc.DepthStencilState.DepthFunc = .LESS;
    pso_desc.PrimitiveTopologyType = .TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
    pso_desc.DSVFormat = zr.Fw.dsv_format;
    pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

    const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
        .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_1,
        .u = .{
            .Desc_1_1 = .{
                .NumParameters = 1,
                .pParameters = &[_]d3d12.ROOT_PARAMETER1 {
                    .{
                        .ParameterType = .CBV,
                        .u = .{
                            .Descriptor = .{
                                .ShaderRegister = 0,
                                .RegisterSpace = 0,
                                .Flags = d3d12.ROOT_DESCRIPTOR_FLAG_NONE
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

    return try fw.lookupOrCreatePipeline(&pso_desc, null, &rs_desc);
}

fn createTexturePipeline(fw: *zr.Fw) !zr.ObjectHandle {
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
            .Format = .R32G32_FLOAT,
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
        .pShaderBytecode = texture_vs,
        .BytecodeLength = texture_vs.len
    };
    pso_desc.PS = .{
        .pShaderBytecode = texture_ps,
        .BytecodeLength = texture_ps.len
    };
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    pso_desc.SampleMask = 0xFFFFFFFF;
    pso_desc.RasterizerState.FillMode = .SOLID;
    pso_desc.RasterizerState.CullMode = .NONE;
    pso_desc.RasterizerState.FrontCounterClockwise = w32.TRUE;
    pso_desc.DepthStencilState.DepthEnable = w32.TRUE;
    pso_desc.DepthStencilState.DepthWriteMask = .ALL;
    pso_desc.DepthStencilState.DepthFunc = .LESS;
    pso_desc.PrimitiveTopologyType = .TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
    pso_desc.DSVFormat = zr.Fw.dsv_format;
    pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

    const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
        .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_1,
        .u = .{
            .Desc_1_1 = .{
                .NumParameters = 2,
                .pParameters = &[_]d3d12.ROOT_PARAMETER1 {
                    .{
                        .ParameterType = .DESCRIPTOR_TABLE,
                        .u = .{
                            .DescriptorTable = .{
                                .NumDescriptorRanges = 2,
                                .pDescriptorRanges = &[_]d3d12.DESCRIPTOR_RANGE1 {
                                    .{
                                        .RangeType = .CBV,
                                        .NumDescriptors = 1,
                                        .BaseShaderRegister = 0, // b0
                                        .RegisterSpace = 0,
                                        .Flags = d3d12.DESCRIPTOR_RANGE_FLAG_NONE,
                                        .OffsetInDescriptorsFromTableStart = 0
                                    },
                                    .{
                                        .RangeType = .SRV,
                                        .NumDescriptors = 1,
                                        .BaseShaderRegister = 0, // t0
                                        .RegisterSpace = 0,
                                        .Flags = d3d12.DESCRIPTOR_RANGE_FLAG_NONE,
                                        .OffsetInDescriptorsFromTableStart = 1
                                    }
                                }
                            }
                        },
                        .ShaderVisibility = .ALL
                    },
                    .{
                        .ParameterType = .DESCRIPTOR_TABLE,
                        .u = .{
                            .DescriptorTable = .{
                                .NumDescriptorRanges = 1,
                                .pDescriptorRanges = &[_]d3d12.DESCRIPTOR_RANGE1 {
                                    .{
                                        .RangeType = .SAMPLER,
                                        .NumDescriptors = 1,
                                        .BaseShaderRegister = 0, // s0
                                        .RegisterSpace = 0,
                                        .Flags = d3d12.DESCRIPTOR_RANGE_FLAG_NONE,
                                        .OffsetInDescriptorsFromTableStart = 0
                                    }
                                }
                            }
                        },
                        .ShaderVisibility = .PIXEL
                    }
                },
                .NumStaticSamplers = 0,
                .pStaticSamplers = null,
                .Flags = d3d12.ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            }
        }
    };

    return try fw.lookupOrCreatePipeline(&pso_desc, null, &rs_desc);
}

const GltfCbData = struct {
    mvp: [16]f32,
    base_color: [4]f32
};
comptime { std.debug.assert(@sizeOf(GltfCbData) == 80); }

fn createGltfPipeline(fw: *zr.Fw) !zr.ObjectHandle {
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
            .SemanticName = "NORMAL",
            .SemanticIndex = 0,
            .Format = .R32G32B32_FLOAT,
            .InputSlot = 0,
            .AlignedByteOffset = 3 * @sizeOf(f32),
            .InputSlotClass = .PER_VERTEX_DATA,
            .InstanceDataStepRate = 0
        },
        d3d12.INPUT_ELEMENT_DESC {
            .SemanticName = "TEXCOORD",
            .SemanticIndex = 0,
            .Format = .R32G32_FLOAT,
            .InputSlot = 0,
            .AlignedByteOffset = 6 * @sizeOf(f32),
            .InputSlotClass = .PER_VERTEX_DATA,
            .InstanceDataStepRate = 0
        }
        // tangents are there in the buffer but skipped here
    };
    var pso_desc = std.mem.zeroes(d3d12.GRAPHICS_PIPELINE_STATE_DESC);
    pso_desc.InputLayout = .{
        .pInputElementDescs = &input_element_descs,
        .NumElements = input_element_descs.len
    };
    pso_desc.VS = .{
        .pShaderBytecode = gltf_vs,
        .BytecodeLength = gltf_vs.len
    };
    pso_desc.PS = .{
        .pShaderBytecode = gltf_ps,
        .BytecodeLength = gltf_ps.len
    };
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    pso_desc.SampleMask = 0xFFFFFFFF;
    pso_desc.RasterizerState.FillMode = .SOLID;
    pso_desc.RasterizerState.CullMode = .BACK;
    pso_desc.RasterizerState.DepthClipEnable = w32.TRUE;
    pso_desc.DepthStencilState.DepthEnable = w32.TRUE;
    pso_desc.DepthStencilState.DepthWriteMask = .ALL;
    pso_desc.DepthStencilState.DepthFunc = .LESS;
    pso_desc.PrimitiveTopologyType = .TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
    pso_desc.DSVFormat = zr.Fw.dsv_format;
    pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

    const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
        .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_1,
        .u = .{
            .Desc_1_1 = .{
                .NumParameters = 3,
                .pParameters = &[_]d3d12.ROOT_PARAMETER1 {
                    .{
                        .ParameterType = .CBV,
                        .u = .{
                            .Descriptor = .{
                                .ShaderRegister = 0, // b0
                                .RegisterSpace = 0,
                                .Flags = d3d12.ROOT_DESCRIPTOR_FLAG_NONE
                            }
                        },
                        .ShaderVisibility = .ALL
                    },
                    .{
                        .ParameterType = .DESCRIPTOR_TABLE,
                        .u = .{
                            .DescriptorTable = .{
                                .NumDescriptorRanges = 1,
                                .pDescriptorRanges = &[_]d3d12.DESCRIPTOR_RANGE1 {
                                    .{
                                        .RangeType = .SRV,
                                        .NumDescriptors = 1,
                                        .BaseShaderRegister = 0, // t0
                                        .RegisterSpace = 0,
                                        .Flags = d3d12.DESCRIPTOR_RANGE_FLAG_NONE,
                                        .OffsetInDescriptorsFromTableStart = 0
                                    }
                                }
                            }
                        },
                        .ShaderVisibility = .PIXEL
                    },
                    .{
                        .ParameterType = .DESCRIPTOR_TABLE,
                        .u = .{
                            .DescriptorTable = .{
                                .NumDescriptorRanges = 1,
                                .pDescriptorRanges = &[_]d3d12.DESCRIPTOR_RANGE1 {
                                    .{
                                        .RangeType = .SAMPLER,
                                        .NumDescriptors = 1,
                                        .BaseShaderRegister = 0, // s0
                                        .RegisterSpace = 0,
                                        .Flags = d3d12.DESCRIPTOR_RANGE_FLAG_NONE,
                                        .OffsetInDescriptorsFromTableStart = 0
                                    }
                                }
                            }
                        },
                        .ShaderVisibility = .PIXEL
                    }
                },
                .NumStaticSamplers = 0,
                .pStaticSamplers = null,
                .Flags = d3d12.ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            }
        }
    };

    return try fw.lookupOrCreatePipeline(&pso_desc, null, &rs_desc);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var fw = try zr.Fw.init(allocator, zr.Fw.Options {
        .enable_debug_layer = true
    });
    defer fw.deinit();

    const device = fw.getDevice();
    const resource_pool = fw.getResourcePool();

    const simple_pipeline = try createSimplePipeline(&fw);
    const color_pipeline = try createColorPipeline(&fw);
    const texture_pipeline = try createTexturePipeline(&fw);
    const gltf_pipeline = try createGltfPipeline(&fw);

    var vbuf_color = try fw.createBuffer(.DEFAULT, 3 * @sizeOf(VertexWithColor));
    var vbuf_uv = try fw.createBuffer(.DEFAULT, 3 * @sizeOf(VertexWithUv));
    var ibuf = try fw.createBuffer(.DEFAULT, 3 * @sizeOf(u16));

    var needs_upload = true;
    var rotation: f32 = 0.0;
    var last_size_used_for_projection = zr.Size.empty();
    var projection = zm.identity();

    var cbv_srv_uav_pool = try zr.CpuDescriptorPool.init(allocator,
                                                         device,
                                                         .CBV_SRV_UAV);
    defer cbv_srv_uav_pool.deinit();

    const one_cbuf_size = zr.alignedSize(@sizeOf(CbData), 256);
    const cbuf_size = one_cbuf_size * 2;
    var cbuf = try fw.createBuffer(.UPLOAD, cbuf_size);
    var cbuf_p = try fw.mapBuffer(u8, cbuf);
    var cbv2 = try cbv_srv_uav_pool.allocate(1);
    // Not a great strategy, but want to play with descriptor tables
    // here so create a cbv pointing to the second set of cb data.
    device.CreateConstantBufferView(
        &.{
            .BufferLocation = resource_pool.lookupRef(cbuf).?.resource.GetGPUVirtualAddress() + one_cbuf_size,
            .SizeInBytes = one_cbuf_size
        },
        cbv2.cpu_handle);

    // No static sampler here, but exercise the helper.
    var sampler_desc = std.mem.zeroes(d3d12.SAMPLER_DESC);
    sampler_desc.Filter = .MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = .CLAMP;
    sampler_desc.AddressV = .CLAMP;
    sampler_desc.AddressW = .CLAMP;
    sampler_desc.MaxLOD = std.math.floatMax(f32); // mipmapping
    const sampler = try fw.lookupOrCreateSampler(.{ .desc = sampler_desc, .stype = .ShaderVisible });

    var image = try zstbi.Image.init("maps/test.png", 4); // goes on the stbiArena
    const image_size = zr.Size { .width = image.width, .height = image.height };
    const texture = try fw.createTexture2DSimple(.R8G8B8A8_UNORM, image_size, zr.mipLevelsForSize(image_size));
    var srv = try cbv_srv_uav_pool.allocate(1);
    fw.createSrv2DAllMips(&srv, texture);

    // If we didn't want to build the tables in the shader visible heap on every
    // frame it could be done once in the permanent area instead.
    //
    // const cbv_srv_uav_start = fw.getPermanentShaderVisibleCbvSrvUavHeapRange().get(2);
    // var cpu_handle = cbv_srv_uav_start.cpu_handle;
    // device.CopyDescriptorsSimple(1, cpu_handle, cbv2.cpu_handle, .CBV_SRV_UAV);
    // cpu_handle.ptr += fw.getPermanentShaderVisibleCbvSrvUavHeapRange().descriptor_byte_size;
    // device.CopyDescriptorsSimple(1, cpu_handle, srv.cpu_handle, .CBV_SRV_UAV);

    var torus = zmesh.Shape.initTorus(10, 10, 0.2); // mesh arena
    const torus_vertex_count = @intCast(u32, torus.positions.len);
    const torus_index_count = @intCast(u32, torus.indices.len);
    var vbuf_torus = try fw.createBuffer(.DEFAULT, @intCast(u32, torus_vertex_count * 3 * @sizeOf(f32)));
    var ibuf_torus = try fw.createBuffer(.DEFAULT, @intCast(u32, torus_index_count * @sizeOf(u32)));

    //var gltf_mesh = try fw.loadGltf("models/duck/Duck.gltf");
    var gltf_mesh = try fw.loadGltf("models/sponza/Sponza.gltf");
    defer gltf_mesh.deinit();
    const gltf_vertex_count = @intCast(u32, gltf_mesh.vertices.items.len);
    const gltf_index_count = @intCast(u32, gltf_mesh.indices.items.len);
    var vbuf_gltf = try fw.createBuffer(.DEFAULT, gltf_vertex_count * gltf_mesh.vertex_stride);
    var ibuf_gltf = try fw.createBuffer(.DEFAULT, gltf_index_count * gltf_mesh.index_stride);
    var gltf_texture_objects = std.ArrayList(zr.ObjectHandle).init(allocator);
    defer gltf_texture_objects.deinit();
    try gltf_texture_objects.ensureTotalCapacity(gltf_mesh.textures.items.len);
    var gltf_texture_srvs = std.ArrayList(zr.Descriptor).init(allocator);
    defer gltf_texture_srvs.deinit();
    try gltf_texture_srvs.ensureTotalCapacity(gltf_mesh.textures.items.len);
    for (gltf_mesh.textures.items) |gtex| {
        if (gtex.image == null) {
            gltf_texture_objects.appendAssumeCapacity(zr.ObjectHandle.invalid());
            continue;
        }
        const resource_handle = try fw.createTexture2DSimple(.R8G8B8A8_UNORM, zr.Size { .width = gtex.image.?.width, .height = gtex.image.?.height }, 1);
        gltf_texture_objects.appendAssumeCapacity(resource_handle);
        var gltf_srv = try cbv_srv_uav_pool.allocate(1);
        fw.createSrv2DNoMips(&gltf_srv, resource_handle);
        gltf_texture_srvs.appendAssumeCapacity(gltf_srv);
    }
    // for glTF models with lots of big textures, the 32 MB per-frame staging is not enough
    var tex_staging_area: ?zr.StagingArea = try zr.StagingArea.init(device, 512 * 1024 * 1024, .UPLOAD);
    defer if (tex_staging_area != null) tex_staging_area.?.deinit();
    var gltf_sampler_desc = std.mem.zeroes(d3d12.SAMPLER_DESC);
    gltf_sampler_desc.Filter = .MIN_MAG_MIP_LINEAR;
    gltf_sampler_desc.AddressU = .WRAP;
    gltf_sampler_desc.AddressV = .WRAP;
    gltf_sampler_desc.AddressW = .WRAP;
    const gltf_sampler = try fw.lookupOrCreateSampler(.{ .desc = gltf_sampler_desc, .stype = .ShaderVisible });

    var camera = zr.Camera { };
    const GuiState = struct {
        demo_window_open: bool = true,
        window_open: bool = true,
        rotate: bool = true
    };
    var gui_state = GuiState { };

    while (zr.Fw.handleWindowEvents()) {
        if (try fw.beginFrame() != zr.Fw.BeginFrameResult.success) {
            continue;
        }

        fw.updateCamera(&camera);
        const view_matrix = camera.getViewMatrix();

        const output_pixel_size = fw.getBackBufferPixelSize();
        const cmd_list = fw.getCommandList();
        // 'current' = per-frame, their start ptr is reset to zero in beginFrame()
        const staging = fw.getCurrentStagingArea();
        const shader_visible_cbv_srv_uav_heap = fw.getCurrentShaderVisibleCbvSrvUavHeapRange();
        //const shader_visible_sampler_heap = fw.getCurrentShaderVisibleSamplerHeapRange();

        if (!std.meta.eql(output_pixel_size, last_size_used_for_projection)) {
            last_size_used_for_projection = output_pixel_size;
            projection = zm.perspectiveFovLh(45.0,
                                             @intToFloat(f32, output_pixel_size.width) / @intToFloat(f32, output_pixel_size.height),
                                             0.01,
                                             1000.0);
        }

        if (needs_upload) {
            needs_upload = false;
            fw.addTransitionBarrier(vbuf_color, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(vbuf_uv, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(ibuf, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(vbuf_torus, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(ibuf_torus, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(vbuf_gltf, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(ibuf_gltf, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.recordTransitionBarriers();

            try fw.uploadBuffer(VertexWithColor, vbuf_color, &vertices_with_color, staging);
            try fw.uploadBuffer(VertexWithUv, vbuf_uv, &vertices_with_uv, staging);
            try fw.uploadBuffer(u16, ibuf, &[_]u16 { 0, 1, 2}, staging);

            try fw.uploadBuffer([3]f32, vbuf_torus, torus.positions, staging);
            try fw.uploadBuffer(u32, ibuf_torus, torus.indices, staging);

            try fw.uploadBuffer(zr.MeshVertex, vbuf_gltf, gltf_mesh.vertices.items[0..], staging);
            try fw.uploadBuffer(u32, ibuf_gltf, gltf_mesh.indices.items[0..], staging);

            try fw.uploadTexture2DSimple(texture, image.data, image.bytes_per_component * image.num_components, image.bytes_per_row, staging);

            for (gltf_texture_objects.items) |handle, index| {
                if (resource_pool.isValid(handle)) {
                    fw.addTransitionBarrier(handle, d3d12.RESOURCE_STATE_COPY_DEST);
                    fw.recordTransitionBarriers();
                    const im = gltf_mesh.textures.items[index].image.?;
                    try fw.uploadTexture2DSimple(handle, im.data, im.bytes_per_component * im.num_components, im.bytes_per_row, &tex_staging_area.?);
                    fw.addTransitionBarrier(handle, d3d12.RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                    fw.recordTransitionBarriers();
                }
            }

            fw.addTransitionBarrier(vbuf_color, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(vbuf_uv, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(ibuf, d3d12.RESOURCE_STATE_INDEX_BUFFER);
            fw.addTransitionBarrier(vbuf_torus, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(ibuf_torus, d3d12.RESOURCE_STATE_INDEX_BUFFER);
            fw.addTransitionBarrier(vbuf_gltf, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(ibuf_gltf, d3d12.RESOURCE_STATE_INDEX_BUFFER);
            fw.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_ALL_SHADER_RESOURCE);
            fw.recordTransitionBarriers();

            try fw.generateTexture2DMipmaps(texture);

            fw.resetMeshArena();
            fw.resetImageArena();

            // can only drop this when we know for sure the command list has finished on the GPU
            fw.deferredReleaseCallback(zr.Fw.releaseStagingArea, &tex_staging_area);
        }

        const rtv = fw.getBackBufferCpuDescriptorHandle();
        const dsv = fw.getDepthStencilBufferCpuDescriptorHandle();
        cmd_list.OMSetRenderTargets(1, &[_]d3d12.CPU_DESCRIPTOR_HANDLE { rtv }, w32.TRUE, &dsv);
        cmd_list.ClearRenderTargetView(rtv, &[4]f32 { 0.4, 0.7, 0.0, 1.0 }, 0, null);
        cmd_list.ClearDepthStencilView(dsv, d3d12.CLEAR_FLAG_DEPTH | d3d12.CLEAR_FLAG_STENCIL, 1.0, 0, 0, null);

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

        fw.setPipeline(color_pipeline);
        cmd_list.IASetPrimitiveTopology(.TRIANGLELIST);
        cmd_list.IASetVertexBuffers(0, 1, &[_]d3d12.VERTEX_BUFFER_VIEW {
            .{
                .BufferLocation = resource_pool.lookupRef(vbuf_color).?.resource.GetGPUVirtualAddress(),
                .SizeInBytes = 3 * @sizeOf(VertexWithColor),
                .StrideInBytes = @sizeOf(VertexWithColor)
            }
        });
        cmd_list.IASetIndexBuffer(&.{
            .BufferLocation = resource_pool.lookupRef(ibuf).?.resource.GetGPUVirtualAddress(),
            .SizeInBytes = 3 * @sizeOf(u16),
            .Format = .R16_UINT
        });
        var cb_data: CbData = undefined;
        {
            const model = zm.mul(zm.rotationY(rotation), zm.translation(-3.0, 0.0, 0.0));
            const modelview = zm.mul(model, view_matrix);
            const mvp = zm.mul(modelview, projection);
            // Mat is stored as row major, transpose to get column major
            zm.storeMat(&cb_data.mvp, zm.transpose(mvp));
            std.mem.copy(u8, cbuf_p, std.mem.asBytes(&cb_data));
        }
        cmd_list.SetGraphicsRootConstantBufferView(0, resource_pool.lookupRef(cbuf).?.resource.GetGPUVirtualAddress());
        cmd_list.DrawIndexedInstanced(3, 1, 0, 0, 0);

        fw.setPipeline(texture_pipeline);
        {
            const model = zm.mul(zm.rotationY(rotation), zm.translation(3.0, 0.0, 0.0));
            const modelview = zm.mul(model, view_matrix);
            const mvp = zm.mul(modelview, projection);
            zm.storeMat(&cb_data.mvp, zm.transpose(mvp));
            std.mem.copy(u8, cbuf_p[one_cbuf_size..], std.mem.asBytes(&cb_data));
        }
        // param 0: cbv, srv
        // param 1: sampler
        const cbv_srv_uav_start = try shader_visible_cbv_srv_uav_heap.get(2);
        var cpu_handle = cbv_srv_uav_start.cpu_handle;
        device.CopyDescriptorsSimple(1, cpu_handle, cbv2.cpu_handle, .CBV_SRV_UAV);
        cpu_handle.ptr += shader_visible_cbv_srv_uav_heap.descriptor_byte_size;
        device.CopyDescriptorsSimple(1, cpu_handle, srv.cpu_handle, .CBV_SRV_UAV);
        // don't need this if we have just a single sampler with .ShaderVisible
        // const sampler_table_start = try shader_visible_sampler_heap.get(1);
        // device.CopyDescriptorsSimple(1, sampler_table_start.cpu_handle, sampler.cpu_handle, .SAMPLER);
        cmd_list.SetGraphicsRootDescriptorTable(0, cbv_srv_uav_start.gpu_handle);
        //cmd_list.SetGraphicsRootDescriptorTable(1, sampler_table_start.gpu_handle);
        cmd_list.SetGraphicsRootDescriptorTable(1, sampler.gpu_handle);
        cmd_list.IASetVertexBuffers(0, 1, &[_]d3d12.VERTEX_BUFFER_VIEW {
            .{
                .BufferLocation = resource_pool.lookupRef(vbuf_uv).?.resource.GetGPUVirtualAddress(),
                .SizeInBytes = 3 * @sizeOf(VertexWithUv),
                .StrideInBytes = @sizeOf(VertexWithUv)
            }
        });
        cmd_list.DrawIndexedInstanced(3, 1, 0, 0, 0);

        fw.setPipeline(simple_pipeline);
        cmd_list.IASetVertexBuffers(0, 1, &[_]d3d12.VERTEX_BUFFER_VIEW {
            .{
                .BufferLocation = resource_pool.lookupRef(vbuf_torus).?.resource.GetGPUVirtualAddress(),
                .SizeInBytes = torus_vertex_count * 3 * @sizeOf(f32),
                .StrideInBytes = 3 * @sizeOf(f32)
            }
        });
        cmd_list.IASetIndexBuffer(&.{
            .BufferLocation = resource_pool.lookupRef(ibuf_torus).?.resource.GetGPUVirtualAddress(),
            .SizeInBytes = torus_index_count * @sizeOf(u32),
            .Format = .R32_UINT
        });
        var torus_cb_data: SimpleCbData = undefined;
        const torus_cb_alloc = try staging.get(@sizeOf(SimpleCbData));
        {
            const model = zm.mul(zm.rotationY(-rotation), zm.translation(0.0, 0.0, 0.0));
            const modelview = zm.mul(model, view_matrix);
            const mvp = zm.mul(modelview, projection);
            zm.storeMat(&torus_cb_data.mvp, zm.transpose(mvp));
            torus_cb_data.color = [4]f32 { 0.0, 1.0, 1.0, 1.0 };
            std.mem.copy(SimpleCbData, torus_cb_alloc.castCpuSlice(SimpleCbData), &[_]SimpleCbData { torus_cb_data });
        }
        cmd_list.SetGraphicsRootConstantBufferView(0, torus_cb_alloc.gpu_addr);
        cmd_list.DrawIndexedInstanced(torus_index_count, 1, 0, 0, 0);

        fw.setPipeline(gltf_pipeline);
        const submesh_count = @intCast(u32, gltf_mesh.submeshes.items.len);
        const gltf_one_cb_size = zr.alignedSize(@sizeOf(GltfCbData), 256);
        const gltf_cb_alloc = try staging.get(gltf_one_cb_size * submesh_count);
        const gltf_model = zm.mul(zm.scaling(0.01, 0.01, 0.01), zm.translation(0.0, 3.0, 0.0));
        const gltf_modelview = zm.mul(gltf_model, view_matrix);
        const gltf_mvp = zm.transpose(zm.mul(gltf_modelview, projection));
        var submesh_index: u32 = 0;
        while (submesh_index < submesh_count) : (submesh_index += 1) {
            const material = gltf_mesh.materials.items[gltf_mesh.submeshes.items[submesh_index].material_index];
            var gltf_cb_data: GltfCbData = undefined;
            zm.storeMat(&gltf_cb_data.mvp, gltf_mvp);
            gltf_cb_data.base_color = material.base_color;
            std.mem.copy(u8, gltf_cb_alloc.cpu_slice[gltf_one_cb_size * submesh_index..], std.mem.asBytes(&gltf_cb_data));
        }
        const gltf_vbuf_resource = resource_pool.lookupRef(vbuf_gltf).?.resource;
        const gltf_ibuf_resource = resource_pool.lookupRef(ibuf_gltf).?.resource;
        for (gltf_mesh.submeshes.items) |submesh, index| {
            const material = gltf_mesh.materials.items[submesh.material_index];
            cmd_list.IASetVertexBuffers(0, 1, &[_]d3d12.VERTEX_BUFFER_VIEW {
                .{
                    .BufferLocation = gltf_vbuf_resource.GetGPUVirtualAddress() + submesh.vertices_start_index * gltf_mesh.vertex_stride,
                    .SizeInBytes = submesh.vertex_count * gltf_mesh.vertex_stride,
                    .StrideInBytes = gltf_mesh.vertex_stride
                }
            });
            cmd_list.IASetIndexBuffer(&.{
                .BufferLocation = gltf_ibuf_resource.GetGPUVirtualAddress() + submesh.indices_start_index * gltf_mesh.index_stride,
                .SizeInBytes = submesh.index_count * gltf_mesh.index_stride,
                .Format = gltf_mesh.index_format
            });
            cmd_list.SetGraphicsRootConstantBufferView(0, gltf_cb_alloc.gpu_addr + gltf_one_cb_size * index);
            const base_color_tex_srv_cpu = if (material.base_color_tex_index) |base_color_tex_index|
                gltf_texture_srvs.items[base_color_tex_index]
                else try fw.getDummyTexture(&cbv_srv_uav_pool);
            const base_color_tex_srv_gpu = try shader_visible_cbv_srv_uav_heap.get(1);
            device.CopyDescriptorsSimple(1, base_color_tex_srv_gpu.cpu_handle, base_color_tex_srv_cpu.cpu_handle, .CBV_SRV_UAV);
            cmd_list.SetGraphicsRootDescriptorTable(1, base_color_tex_srv_gpu.gpu_handle);
            cmd_list.SetGraphicsRootDescriptorTable(2, gltf_sampler.gpu_handle);
            cmd_list.DrawIndexedInstanced(submesh.index_count, 1, 0, 0, 0);
        }

        try fw.beginGui(&cbv_srv_uav_pool);
        if (gui_state.demo_window_open) {
            imgui.igShowDemoWindow(&gui_state.demo_window_open);
        }
        if (gui_state.window_open) {
            imgui.igSetNextWindowPos(imgui.ImVec2 { .x = 0, .y = 0 }, imgui.ImGuiCond_FirstUseEver, imgui.ImVec2 { .x = 0, .y = 0 });
            imgui.igSetNextWindowSize(imgui.ImVec2 { .x = 650, .y = 120 }, imgui.ImGuiCond_FirstUseEver);
            if (imgui.igBegin("Test", &gui_state.window_open, imgui.ImGuiWindowFlags_None)) {
                imgui.igText("Mouse + WASDRF to move the camera (when no ImGui window is focused)");
                _ = imgui.igCheckbox("Rotate", &gui_state.rotate);
                _ = imgui.igText(fw.formatTempZ("Test formatting {} {s}", .{123, "abcd"}).ptr);
            }
            imgui.igEnd();
        }
        try fw.endGui();

        try fw.endFrame();

        if (gui_state.rotate) {
            rotation += 0.01;
        }
    }

    std.debug.print("Exiting\n", .{});
}
