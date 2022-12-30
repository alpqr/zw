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

fn create_simple_pipeline(fw: *zr.Fw) !zr.ObjectHandle {
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

fn create_color_pipeline(fw: *zr.Fw) !zr.ObjectHandle {
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

fn create_texture_pipeline(fw: *zr.Fw) !zr.ObjectHandle {
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

fn loadGltf(gltf_path: [:0]const u8) zmesh.gltf.Error!*zmesh.gltf.Data {
    const options = zmesh.gltf.Options {
        .memory = .{
            // .alloc_func = zmesh.mem.zmeshAllocUser,
            // .free_func = zmesh.mem.zmeshFreeUser,
        },
    };
    const data = try zmesh.gltf.parseFile(options, gltf_path);
    errdefer zmesh.gltf.free(data);
    try zmesh.gltf.loadBuffers(options, data, gltf_path);
    return data;
}

fn copySubMeshData(
    data: *zmesh.gltf.Data,
    mesh_index: u32,
    submesh_index: u32,
    indices: *std.ArrayList(u32),
    positions: *std.ArrayList([3]f32),
    normals: *std.ArrayList([3]f32),
    texcoords0: *std.ArrayList([2]f32),
) !void {
    std.debug.assert(mesh_index < data.meshes_count);
    std.debug.assert(submesh_index < data.meshes.?[mesh_index].primitives_count);
    const mesh = &data.meshes.?[mesh_index];
    const submesh = &mesh.primitives[submesh_index];

    {
        const num_indices: u32 = @intCast(u32, submesh.indices.?.count);
        try indices.ensureTotalCapacity(indices.items.len + num_indices);
        const accessor = submesh.indices.?;
        const buffer_view = accessor.buffer_view.?;
        std.debug.assert(buffer_view.buffer.data != null);
        const data_addr = @alignCast(4, @ptrCast([*]const u8, buffer_view.buffer.data) + accessor.offset + buffer_view.offset);
        if (accessor.stride == 1) {
            std.debug.assert(accessor.component_type == .r_8u);
            const src = @ptrCast([*]const u8, data_addr);
            var i: u32 = 0;
            while (i < num_indices) : (i += 1) {
                indices.appendAssumeCapacity(src[i]);
            }
        } else if (accessor.stride == 2) {
            std.debug.assert(accessor.component_type == .r_16u);
            const src = @ptrCast([*]const u16, data_addr);
            var i: u32 = 0;
            while (i < num_indices) : (i += 1) {
                indices.appendAssumeCapacity(src[i]);
            }
        } else if (accessor.stride == 4) {
            std.debug.assert(accessor.component_type == .r_32u);
            const src = @ptrCast([*]const u32, data_addr);
            var i: u32 = 0;
            while (i < num_indices) : (i += 1) {
                indices.appendAssumeCapacity(src[i]);
            }
        }
    }

    for (submesh.attributes[0..submesh.attributes_count]) |attrib| {
        const accessor = attrib.data;
        std.debug.assert(accessor.component_type == .r_32f);
        const buffer_view = accessor.buffer_view.?;
        std.debug.assert(buffer_view.buffer.data != null);
        try positions.ensureTotalCapacity(positions.items.len + accessor.count);
        try normals.ensureTotalCapacity(normals.items.len + accessor.count);
        try texcoords0.ensureTotalCapacity(texcoords0.items.len + accessor.count);
        var offset = accessor.offset + buffer_view.offset;
        var i: u32 = 0;
        while (i < accessor.count) : (i += 1) {
            const data_addr = @ptrCast([*]const u8, buffer_view.buffer.data) + offset;
            if (attrib.type == .position) {
                std.debug.assert(accessor.type == .vec3);
                const d = @ptrCast([*]const [3]f32, @alignCast(4, data_addr));
                positions.appendAssumeCapacity(d[0]);
            } else if (attrib.type == .normal) {
                std.debug.assert(accessor.type == .vec3);
                const d = @ptrCast([*]const [3]f32, @alignCast(4, data_addr));
                normals.appendAssumeCapacity(d[0]);
            } else if (attrib.type == .texcoord) {
                std.debug.assert(accessor.type == .vec2);
                const d = @ptrCast([*]const [2]f32, @alignCast(4, data_addr));
                texcoords0.appendAssumeCapacity(d[0]);
            }
            offset += if (buffer_view.stride != 0) buffer_view.stride else accessor.stride;
        }
    }
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

    const simple_pipeline = try create_simple_pipeline(&fw);
    const color_pipeline = try create_color_pipeline(&fw);
    const texture_pipeline = try create_texture_pipeline(&fw);

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

    var image = try zstbi.Image.init("maps/test.png", 4);
    defer image.deinit();
    const image_size = zr.Size { .width = image.width, .height = image.height };
    const texture = try fw.createTexture2DSimple(.R8G8B8A8_UNORM, image_size, zr.mipLevelsForSize(image_size));
    var srv = try cbv_srv_uav_pool.allocate(1);
    device.CreateShaderResourceView(
        resource_pool.lookupRef(texture).?.resource,
        &.{
            .Format = .R8G8B8A8_UNORM,
            .ViewDimension = .TEXTURE2D,
            .Shader4ComponentMapping = d3d12.DEFAULT_SHADER_4_COMPONENT_MAPPING,
            .u = .{
                .Texture2D = .{
                    .MostDetailedMip = 0,
                    .MipLevels = resource_pool.lookupRef(texture).?.desc.MipLevels,
                    .PlaneSlice = 0,
                    .ResourceMinLODClamp = 0.0
                }
            }
        },
        srv.cpu_handle);

    // If we didn't want to build the tables in the shader visible heap on every
    // frame it could be done once in the permanent area instead.
    //
    // const cbv_srv_uav_start = fw.getPermanentShaderVisibleCbvSrvUavHeapRange().get(2);
    // var cpu_handle = cbv_srv_uav_start.cpu_handle;
    // device.CopyDescriptorsSimple(1, cpu_handle, cbv2.cpu_handle, .CBV_SRV_UAV);
    // cpu_handle.ptr += fw.getPermanentShaderVisibleCbvSrvUavHeapRange().descriptor_byte_size;
    // device.CopyDescriptorsSimple(1, cpu_handle, srv.cpu_handle, .CBV_SRV_UAV);

    var gltf = try loadGltf("models/duck/Duck.gltf");
    defer zmesh.gltf.free(gltf);

    const SubMesh = struct {
        indices_start_index: u32,
        vertex_start_index: u32,
        num_indices: u32,
        num_vertices: u32,
        material_index: u32
    };

    // const MeshVertex = struct {
    //     position: [3]f32,
    //     normal: [3]f32,
    //     texcoords0: [2]f32,
    // };

    var submeshes = std.ArrayList(SubMesh).init(allocator);
    defer submeshes.deinit();
    var mesh_vertices = std.ArrayList([3]f32).init(allocator);
    defer mesh_vertices.deinit();
    var mesh_indices = std.ArrayList(u32).init(allocator);
    defer mesh_indices.deinit();

    {
        var arena_allocator = std.heap.ArenaAllocator.init(allocator);
        defer arena_allocator.deinit();
        const arena = arena_allocator.allocator();

        var indices = std.ArrayList(u32).init(arena);
        var positions = std.ArrayList([3]f32).init(arena);
        var normals = std.ArrayList([3]f32).init(arena);
        var texcoords0 = std.ArrayList([2]f32).init(arena);

        const num_meshes = @intCast(u32, gltf.meshes_count);
        const num_materials = @intCast(u32, gltf.materials_count);

        var mesh_index: u32 = 0;
        while (mesh_index < num_meshes) : (mesh_index += 1) {
            const num_submeshes = @intCast(u32, gltf.meshes.?[mesh_index].primitives_count);
            var submesh_index: u32 = 0;
            while (submesh_index < num_submeshes) : (submesh_index += 1) {
                const indices_start_index = indices.items.len;
                const positions_start_index = positions.items.len;

                try copySubMeshData(
                    gltf,
                    submesh_index,
                    submesh_index,
                    &indices,
                    &positions,
                    &normals,
                    &texcoords0
                );

                var material_index: u32 = 0;
                var assigned_material_index: u32 = std.math.maxInt(u32);
                while (material_index < num_materials) : (material_index += 1) {
                    const submesh = &gltf.meshes.?[mesh_index].primitives[submesh_index];
                    if (submesh.material == &gltf.materials.?[material_index]) {
                        assigned_material_index = material_index;
                        break;
                    }
                }
                if (assigned_material_index == std.math.maxInt(u32)) {
                    continue;
                }
                try submeshes.append(.{
                    .indices_start_index = @intCast(u32, indices_start_index),
                    .vertex_start_index = @intCast(u32, positions_start_index),
                    .num_indices = @intCast(u32, indices.items.len - indices_start_index),
                    .num_vertices = @intCast(u32, positions.items.len - positions_start_index),
                    .material_index = assigned_material_index,
                });
            }
        }

        try mesh_indices.ensureTotalCapacity(indices.items.len);
        for (indices.items) |index| {
            mesh_indices.appendAssumeCapacity(index);
        }

        try mesh_vertices.ensureTotalCapacity(positions.items.len);
        for (positions.items) |_, index| {
            mesh_vertices.appendAssumeCapacity(positions.items[index]);
            // .{
            //     .position = positions.items[index],
            //     .normal = normals.items[index],
            //     .texcoords0 = texcoords0.items[index],
            // });
        }
    }
    std.debug.print("{any}\n", .{submeshes});

    const torus_vertex_count = submeshes.items[0].num_vertices;
    const torus_index_count = submeshes.items[0].num_indices;
    var vbuf_torus = try fw.createBuffer(.DEFAULT, @intCast(u32, torus_vertex_count * 3 * @sizeOf(f32)));
    var ibuf_torus = try fw.createBuffer(.DEFAULT, @intCast(u32, torus_index_count * @sizeOf(u32)));

    // var torus = zmesh.Shape.initTorus(10, 10, 0.2);
    // defer torus.deinit();
    // const torus_vertex_count = @intCast(u32, torus.positions.len);
    // const torus_index_count = @intCast(u32, torus.indices.len);
    // var vbuf_torus = try fw.createBuffer(.DEFAULT, @intCast(u32, torus_vertex_count * 3 * @sizeOf(f32)));
    // var ibuf_torus = try fw.createBuffer(.DEFAULT, @intCast(u32, torus_index_count * @sizeOf(u32)));

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
            fw.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_COPY_DEST);
            fw.recordTransitionBarriers();

            try fw.uploadBuffer(VertexWithColor, vbuf_color, &vertices_with_color, staging);
            try fw.uploadBuffer(VertexWithUv, vbuf_uv, &vertices_with_uv, staging);
            try fw.uploadBuffer(u16, ibuf, &[_]u16 { 0, 1, 2}, staging);

            try fw.uploadBuffer([3]f32, vbuf_torus, mesh_vertices.items[submeshes.items[0].vertex_start_index..submeshes.items[0].vertex_start_index + submeshes.items[0].num_vertices], staging);
            try fw.uploadBuffer(u32, ibuf_torus, mesh_indices.items[submeshes.items[0].indices_start_index..submeshes.items[0].indices_start_index + submeshes.items[0].num_indices], staging);

            try fw.uploadTexture2DSimple(texture, image.data, image.bytes_per_component * image.num_components, image.bytes_per_row, staging);

            fw.addTransitionBarrier(vbuf_color, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(vbuf_uv, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(ibuf, d3d12.RESOURCE_STATE_INDEX_BUFFER);
            fw.addTransitionBarrier(vbuf_torus, d3d12.RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            fw.addTransitionBarrier(ibuf_torus, d3d12.RESOURCE_STATE_INDEX_BUFFER);
            fw.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_ALL_SHADER_RESOURCE);
            fw.recordTransitionBarriers();

            try fw.generateTexture2DMipmaps(texture);
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
        const torus_cb_alloc = try staging.allocate(@sizeOf(SimpleCbData));
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
