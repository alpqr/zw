const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const dxgi = zwin32.dxgi;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const d3d12d = zwin32.d3d12d;
pub const zm = @import("zmath");
pub const zstbi = @import("zstbi");
pub const zmesh = @import("zmesh");
pub const imgui = @cImport({
    @cDefine("CIMGUI_DEFINE_ENUMS_AND_STRUCTS", "");
    @cDefine("CIMGUI_NO_EXPORT", "");
    @cInclude("cimgui.h");
});

const imgui_font = @embedFile("fonts/RobotoMono-Medium.ttf");
const imgui_vs = @embedFile("shaders/imgui.vs.cso");
const imgui_ps = @embedFile("shaders/imgui.ps.cso");
const mipmapgen_cs = @embedFile("shaders/mipmap.cs.cso");

pub const Descriptor = struct {
    cpu_handle: d3d12.CPU_DESCRIPTOR_HANDLE,
    gpu_handle: d3d12.GPU_DESCRIPTOR_HANDLE,

    pub fn invalid() Descriptor {
        return .{
            .cpu_handle = std.mem.zeroes(d3d12.CPU_DESCRIPTOR_HANDLE),
            .gpu_handle = std.mem.zeroes(d3d12.GPU_DESCRIPTOR_HANDLE)
        };
    }
};

pub const DescriptorHeap = struct {
    heap: ?*d3d12.IDescriptorHeap,
    base: Descriptor,
    size: u32,
    capacity: u32,
    descriptor_byte_size: u32,
    heap_type: d3d12.DESCRIPTOR_HEAP_TYPE,
    heap_flags: d3d12.DESCRIPTOR_HEAP_FLAGS,

    pub fn init(device: *d3d12.IDevice9,
                capacity: u32,
                heap_type: d3d12.DESCRIPTOR_HEAP_TYPE,
                heap_flags: d3d12.DESCRIPTOR_HEAP_FLAGS) !DescriptorHeap {
        var heap: *d3d12.IDescriptorHeap = undefined;
        try zwin32.hrErrorOnFail(device.CreateDescriptorHeap(
            &.{
                .Type = heap_type,
                .NumDescriptors = capacity,
                .Flags = heap_flags,
                .NodeMask = 0,
            },
            &d3d12.IID_IDescriptorHeap, @ptrCast(*?*anyopaque, &heap)));
        errdefer _ = heap.Release();

        const descriptor_byte_size = device.GetDescriptorHandleIncrementSize(heap_type);
        const cpu_handle = heap.GetCPUDescriptorHandleForHeapStart();
        var gpu_handle = d3d12.GPU_DESCRIPTOR_HANDLE { .ptr = 0 };
        if ((heap_flags & d3d12.DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) != 0) {
            gpu_handle = heap.GetGPUDescriptorHandleForHeapStart();
        }

        return DescriptorHeap {
            .heap = heap,
            .base = .{
                .cpu_handle = cpu_handle,
                .gpu_handle = gpu_handle
            },
            .size = 0,
            .capacity = capacity,
            .descriptor_byte_size = descriptor_byte_size,
            .heap_type = heap_type,
            .heap_flags = heap_flags
        };
    }

    pub fn initWithExisting(other: DescriptorHeap, offset_in_descriptors: u32, capacity: u32) DescriptorHeap {
        var base = other.base;
        base.cpu_handle.ptr += offset_in_descriptors * other.descriptor_byte_size;
        if (base.gpu_handle.ptr != 0) {
            base.gpu_handle.ptr += offset_in_descriptors * other.descriptor_byte_size;
        }
        return DescriptorHeap {
            .heap = null,
            .base = base,
            .size = 0,
            .capacity = capacity,
            .descriptor_byte_size = other.descriptor_byte_size,
            .heap_type = other.heap_type,
            .heap_flags = other.heap_flags
        };
    }

    pub fn deinit(self: *DescriptorHeap) void {
        if (self.heap) |h| {
            _ = h.Release();
        }
    }

    pub fn get(self: *DescriptorHeap, count: u32) !Descriptor {
        std.debug.assert(count > 0);
        if (self.size + count > self.capacity) {
            std.debug.print("Cannot get {} descriptors as that would exceed capacity {}\n",
                            .{ count, self.capacity });
            return error.DescriptorHeapOverflow;
        }
        self.size += count;
        return self.at(self.size - count);
    }

    pub fn at(self: *const DescriptorHeap, index: u32) Descriptor {
        const start_offset = index * self.descriptor_byte_size;

        const cpu_handle = d3d12.CPU_DESCRIPTOR_HANDLE {
            .ptr = self.base.cpu_handle.ptr + start_offset
        };
        var gpu_handle = d3d12.GPU_DESCRIPTOR_HANDLE {
            .ptr = if (self.base.gpu_handle.ptr != 0) self.base.gpu_handle.ptr + start_offset else 0
        };

        return Descriptor {
            .cpu_handle = cpu_handle,
            .gpu_handle = gpu_handle
        };
    }

    pub fn reset(self: *DescriptorHeap) void {
        self.size = 0;
    }

    pub fn remainingCapacity(self: *const DescriptorHeap) u32 {
        return self.capacity - self.size;
    }
};

pub const CpuDescriptorPool = struct {
    const descriptors_per_heap = 256;

    const HeapWithMap = struct {
        heap: DescriptorHeap,
        map: std.StaticBitSet(descriptors_per_heap)
    };
    heaps: std.ArrayList(HeapWithMap),
    descriptor_byte_size: u32,
    device: *d3d12.IDevice9,

    pub fn init(allocator: std.mem.Allocator,
                device: *d3d12.IDevice9,
                heap_type: d3d12.DESCRIPTOR_HEAP_TYPE) !CpuDescriptorPool {
        var firstHeap = try DescriptorHeap.init(device, descriptors_per_heap, heap_type, d3d12.DESCRIPTOR_HEAP_FLAG_NONE);
        errdefer firstHeap.deinit();
        const h = HeapWithMap {
            .heap = firstHeap,
            .map = std.StaticBitSet(descriptors_per_heap).initEmpty()
        };
        var heaps = std.ArrayList(HeapWithMap).init(allocator);
        try heaps.append(h);
        return CpuDescriptorPool {
            .heaps = heaps,
            .descriptor_byte_size = firstHeap.descriptor_byte_size,
            .device = device,
        };
    }

    pub fn deinit(self: *CpuDescriptorPool) void {
        for (self.heaps.items) |*h| {
            h.heap.deinit();
        }
        self.heaps.deinit();
    }

    pub fn allocate(self: *CpuDescriptorPool, count: u32) !Descriptor {
        std.debug.assert(count > 0 and count <= descriptors_per_heap);
        var last = &self.heaps.items[self.heaps.items.len - 1];
        if (last.heap.size + count <= last.heap.capacity) {
            const first_index = last.heap.size;
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                last.map.set(first_index + i);
            }
            return last.heap.get(count);
        }
        for (self.heaps.items) |*h| {
            var i: u32 = 0;
            var free_count: u32 = 0;
            while (i < descriptors_per_heap) : (i += 1) {
                if (h.map.isSet(i)) {
                    free_count = 0;
                } else {
                    free_count += 1;
                    if (free_count == count) {
                        const first_index = i - (free_count - 1);
                        var j: u32 = 0;
                        while (j < count) : (j + 1) {
                            h.map.set(first_index + j);
                            return h.heap.at(first_index);
                        }
                    }
                }
            }
        }
        var new_heap = try DescriptorHeap.init(self.device, descriptors_per_heap,
                                               last.heap.heap_type, last.heap.heap_flags);
        errdefer new_heap.deinit();
        const h = HeapWithMap {
            .heap = new_heap,
            .map = std.StaticBitSet(descriptors_per_heap).initEmpty()
        };
        try self.heaps.append(h);
        last = &self.heaps.items[self.heaps.items.len - 1];
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            last.map.set(i);
        }
        return last.heap.get(count);
    }

    pub fn release(self: *CpuDescriptorPool, descriptor: Descriptor, count: u32) void {
        std.debug.assert(count > 0 and count <= descriptors_per_heap);
        const addr = descriptor.cpu_handle.ptr;
        for (self.heaps.items) |*h| {
            const begin = h.heap.base.cpu_handle.ptr;
            const end = begin + h.heap.descriptor_byte_size * h.heap.capacity;
            if (addr >= begin and addr < end) {
                const first_index = (addr - begin) / h.heap.descriptor_byte_size;
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    h.map.unset(first_index + i);
                }
                return;
            }
        }
    }
};

pub const ObjectHandle = struct {
    index: u32,
    generation: u32,

    pub fn invalid() ObjectHandle {
        return ObjectHandle {
            .index = 0,
            .generation = 0
        };
    }
};

pub fn ObjectPool(comptime T: type) type {
    return struct {
        const Self = @This();

        const Data = struct {
            object: ?T,
            generation: u32
        };

        data: std.ArrayList(Data),

        pub fn init(allocator: std.mem.Allocator) !Self {
            var data = std.ArrayList(Data).init(allocator);
            const dummy = Data {
                .object = null,
                .generation = 0
            };
            try data.append(dummy);
            return Self {
                .data = data
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.data.items) |*d| {
                if (d.object) |*object| {
                    object.releaseResources();
                }
            }
            self.data.deinit();
        }

        pub fn isValid(self: *const Self, handle: ObjectHandle) bool {
            return handle.index > 0
                and handle.index < self.data.items.len
                and handle.generation > 0
                and handle.generation == self.data.items[handle.index].generation
                and self.data.items[handle.index].object != null;
        }

        pub fn lookup(self: *const Self, handle: ObjectHandle) ?T {
            if (self.isValid(handle)) {
                return self.data.items[handle.index].object;
            }
            return null;
        }

        /// add() may invalidate the result
        pub fn lookupRef(self: *Self, handle: ObjectHandle) ?*T {
            if (self.isValid(handle)) {
                return &self.data.items[handle.index].object.?;
            }
            return null;
        }

        pub fn add(self: *Self, object: T) !ObjectHandle {
            const count = self.data.items.len;
            var index: u32 = 1;
            while (index < count) : (index += 1) {
                if (self.data.items[index].object == null)
                    break;
            }
            if (index < count) {
                self.data.items[index].object = object;
                var generation = &self.data.items[index].generation;
                generation.* +%= 1;
                if (generation.* == 0)
                    generation.* = 1;
                return ObjectHandle {
                    .index = index,
                    .generation = generation.*
                };
            } else {
                try self.data.append(Data {
                    .object = object,
                    .generation = 1
                });
                return ObjectHandle {
                    .index = @intCast(u32, self.data.items.len) - 1,
                    .generation = 1
                };
            }
        }

        pub fn remove(self: *Self, handle: ObjectHandle) void {
            if (self.lookupRef(handle)) |object| {
                object.releaseResources();
                self.data.items[handle.index].object = null;
            }
        }
    };
}

pub const Resource = struct {
    resource: *d3d12.IResource,
    state: d3d12.RESOURCE_STATES,
    desc: d3d12.RESOURCE_DESC,

    pub fn addToPool(pool: *ObjectPool(Resource), resource: *d3d12.IResource, state: d3d12.RESOURCE_STATES) !ObjectHandle {
        return try pool.add(Resource {
            .resource = resource,
            .state = state,
            .desc = resource.GetDesc()
        });
    }

    pub fn releaseResources(self: *Resource) void {
        _ = self.resource.Release();
    }
};

pub const Pipeline = struct {
    pub const Type = enum {
        Graphics,
        Compute,
    };

    pso: *d3d12.IPipelineState,
    rs: *d3d12.IRootSignature,
    ptype: Type,

    pub fn addToPool(pool: *ObjectPool(Pipeline), pso: *d3d12.IPipelineState, rs: *d3d12.IRootSignature, ptype: Type) !ObjectHandle {
        return try pool.add(Pipeline {
            .pso = pso,
            .rs = rs,
            .ptype = ptype
        });
    }

    pub fn releaseResources(self: *Pipeline) void {
        _ = self.pso.Release();
        _ = self.rs.Release();
    }

    pub const sha_length = 32;

    pub fn getGraphicsPipelineSha(pso_desc: *const d3d12.GRAPHICS_PIPELINE_STATE_DESC, result: *[sha_length]u8) void {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        if (pso_desc.VS.pShaderBytecode != null) {
            hasher.update(@ptrCast([*]const u8, pso_desc.VS.pShaderBytecode.?)[0..pso_desc.VS.BytecodeLength]);
        }
        if (pso_desc.PS.pShaderBytecode != null) {
            hasher.update(@ptrCast([*]const u8, pso_desc.PS.pShaderBytecode.?)[0..pso_desc.PS.BytecodeLength]);
        }
        if (pso_desc.DS.pShaderBytecode != null) {
            hasher.update(@ptrCast([*]const u8, pso_desc.DS.pShaderBytecode.?)[0..pso_desc.DS.BytecodeLength]);
        }
        if (pso_desc.HS.pShaderBytecode != null) {
            hasher.update(@ptrCast([*]const u8, pso_desc.HS.pShaderBytecode.?)[0..pso_desc.HS.BytecodeLength]);
        }
        if (pso_desc.GS.pShaderBytecode != null) {
            hasher.update(@ptrCast([*]const u8, pso_desc.GS.pShaderBytecode.?)[0..pso_desc.GS.BytecodeLength]);
        }
        hasher.update(std.mem.asBytes(&pso_desc.BlendState));
        hasher.update(std.mem.asBytes(&pso_desc.SampleMask));
        hasher.update(std.mem.asBytes(&pso_desc.RasterizerState));
        hasher.update(std.mem.asBytes(&pso_desc.DepthStencilState));
        hasher.update(std.mem.asBytes(&pso_desc.InputLayout.NumElements));
        if (pso_desc.InputLayout.pInputElementDescs) |elements| {
            var i: u32 = 0;
            while (i < pso_desc.InputLayout.NumElements) : (i += 1) {
                hasher.update(std.mem.asBytes(&elements[i].Format));
                hasher.update(std.mem.asBytes(&elements[i].InputSlot));
                hasher.update(std.mem.asBytes(&elements[i].AlignedByteOffset));
                hasher.update(std.mem.asBytes(&elements[i].InputSlotClass));
                hasher.update(std.mem.asBytes(&elements[i].InstanceDataStepRate));
            }
        }
        hasher.update(std.mem.asBytes(&pso_desc.IBStripCutValue));
        hasher.update(std.mem.asBytes(&pso_desc.PrimitiveTopologyType));
        hasher.update(std.mem.asBytes(&pso_desc.NumRenderTargets));
        hasher.update(std.mem.asBytes(&pso_desc.RTVFormats));
        hasher.update(std.mem.asBytes(&pso_desc.DSVFormat));
        hasher.update(std.mem.asBytes(&pso_desc.SampleDesc));
        hasher.update(std.mem.asBytes(&pso_desc.Flags));
        hasher.final(result);
    }

    pub fn getComputePipelineSha(pso_desc: *const d3d12.COMPUTE_PIPELINE_STATE_DESC, result: *[sha_length]u8) void {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        if (pso_desc.CS.pShaderBytecode != null) {
            hasher.update(@ptrCast([*]const u8, pso_desc.CS.pShaderBytecode.?)[0..pso_desc.CS.BytecodeLength]);
        }
        hasher.update(std.mem.asBytes(&pso_desc.Flags));
        hasher.final(result);
    }

    pub const Cache = struct {
        allocator: std.mem.Allocator,
        data: std.StringHashMap(ObjectHandle),

        pub fn init(allocator: std.mem.Allocator) Cache {
            return Cache {
                .allocator = allocator,
                .data = std.StringHashMap(ObjectHandle).init(allocator)
            };
        }

        pub fn deinit(self: *Cache) void {
            self.clear();
            self.data.deinit();
        }

        pub fn clear(self: *Cache) void {
            var it = self.data.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            self.data.clearAndFree();
        }

        pub fn add(self: *Cache, sha: []const u8, handle: ObjectHandle) !void {
            var v = try self.data.getOrPut(sha);
            if (!v.found_existing) {
                var key = try self.allocator.alloc(u8, sha.len);
                std.mem.copy(u8, key, sha);
                v.key_ptr.* = key;
                v.value_ptr.* = handle;
            }
        }

        pub fn remove(self: *Cache, sha: []const u8) void {
            const kv = self.data.fetchRemove(sha) orelse return;
            self.allocator.free(kv.key);
        }

        pub fn count(self: *const Cache) u32 {
            return self.data.count();
        }

        pub fn get(self: *const Cache, sha: []const u8) ?ObjectHandle {
            return self.data.get(sha);
        }
    };
};

pub fn alignedSize(size: u32, alignment: u32) u32 {
    return (size + alignment - 1) & ~(alignment - 1);
}

pub fn mipLevelsForSize(pixel_size: Size) u32 {
    return @floatToInt(u32, std.math.floor(std.math.log2(
        @intToFloat(f32, std.math.max(pixel_size.width, pixel_size.height))))) + 1;
}

pub fn sizeForMipLevel(mip_level: u32, base_level_pixel_size: Size) Size {
    const w = std.math.max(1, base_level_pixel_size.width >> @intCast(u5, mip_level));
    const h = std.math.max(1, base_level_pixel_size.height >> @intCast(u5, mip_level));
    return Size { .width = w, .height = h };
}

pub const StagingArea = struct {
    pub const alignment: u32 = 512; // D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT

    pub const Allocation = struct {
        cpu_slice: []u8,
        gpu_addr: d3d12.GPU_VIRTUAL_ADDRESS,
        buffer: *d3d12.IResource,
        buffer_offset: u32,

        pub fn castCpuSlice(self: *const Allocation, comptime T: type) []T {
            return std.mem.bytesAsSlice(T, @alignCast(@alignOf(T), self.cpu_slice));
        }
    };

    mem: Allocation,
    size: u32,
    capacity: u32,

    pub fn init(device: *d3d12.IDevice9, capacity: u32, heap_type: d3d12.HEAP_TYPE) !StagingArea {
        std.debug.assert(heap_type == .UPLOAD or heap_type == .READBACK);
        var resource: *d3d12.IResource = undefined;
        var heap_properties = std.mem.zeroes(d3d12.HEAP_PROPERTIES);
        heap_properties.Type = heap_type;
        try zwin32.hrErrorOnFail(device.CreateCommittedResource(
            &heap_properties,
            d3d12.HEAP_FLAG_NONE,
            &.{
                .Dimension = .BUFFER,
                .Alignment = 0,
                .Width = capacity,
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = .UNKNOWN,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .ROW_MAJOR,
                .Flags = d3d12.RESOURCE_FLAG_NONE
            },
            d3d12.RESOURCE_STATE_COMMON,
            null,
            &d3d12.IID_IResource,
            @ptrCast(*?*anyopaque, &resource),
        ));
        errdefer _ = resource.Release();
        var p: [*]u8 = undefined;
        try zwin32.hrErrorOnFail(resource.Map(0,  &.{ .Begin = 0, .End = 0 }, @ptrCast(*?*anyopaque, &p)));
        return StagingArea {
            .mem = .{
                .cpu_slice = p[0..capacity],
                .gpu_addr = resource.GetGPUVirtualAddress(),
                .buffer = resource,
                .buffer_offset = 0
            },
            .size = 0,
            .capacity = capacity
        };
    }

    pub fn deinit(self: *StagingArea) void {
        _ = self.mem.buffer.Release();
    }

    pub fn get(self: *StagingArea, size: u32) !Allocation {
        const alloc_size = alignedSize(size, alignment);
        if (self.size + alloc_size > self.capacity) {
            std.debug.print("Failed to allocate {} bytes from staging area of size {}\n",
                            .{ size, self.capacity });
            return error.OutOfStagingMemory;
        }
        const offset = self.size;
        const cpu_slice = (self.mem.cpu_slice.ptr + offset)[0..size];
        const gpu_addr = self.mem.gpu_addr + offset;
        self.size += alloc_size;
        return .{
            .cpu_slice = cpu_slice,
            .gpu_addr = gpu_addr,
            .buffer = self.mem.buffer,
            .buffer_offset = offset
        };
    }

    pub fn reset(self: *StagingArea) void {
        self.size = 0;
    }

    pub fn remainingCapacity(self: *const StagingArea) u32 {
        return self.capacity - self.size;
    }
};

pub const HostVisibleBuffer = struct {
    resource_handle: ObjectHandle = ObjectHandle.invalid(),
    p: []u8 = undefined,

    pub fn ptrAs(self: *const HostVisibleBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, @alignCast(@alignOf(T), self.p));
    }
};

pub const SamplerDescriptorType = enum {
    Cpu,
    ShaderVisible
};

pub const SamplerCacheKey = struct {
    desc: d3d12.SAMPLER_DESC,
    stype: SamplerDescriptorType
};

pub const SamplerCacheHashContext = struct {
    pub fn hash(self: @This(), s: SamplerCacheKey) u64 {
        _ = self;
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&s));
    }
    pub fn eql(self: @This(), a: SamplerCacheKey, b: SamplerCacheKey) bool {
        _ = self;
        return std.meta.eql(a, b);
    }
};

pub const Size = struct {
    width: u32,
    height: u32,

    pub fn empty() Size {
        return .{
            .width = 0,
            .height = 0
        };
    }

    pub fn isEmpty(self: *const Size) bool {
        return self.width == 0 or self.height == 0;
    }
};

pub const Camera = struct {
    pos: [3]f32 = .{ 0.0, 0.0, -10.0 },
    forward: [3]f32 = .{ 0.0, 0.0, 1.0 },
    pitch: f32 = 0.0,
    yaw: f32 = 0.0,

    pub fn getViewMatrix(self: *const Camera) zm.Mat {
        return zm.lookToLh(
            zm.load(&self.pos, zm.Vec, 3),
            zm.load(&self.forward, zm.Vec, 3),
            zm.f32x4(0.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn rotate(self: *Camera, dx: f32, dy: f32, speed: f32) void {
        self.pitch += speed * dy;
        self.yaw += speed * dx;
        self.pitch = std.math.min(self.pitch, 0.48 * std.math.pi);
        self.pitch = std.math.max(self.pitch, -0.48 * std.math.pi);
        self.yaw = zm.modAngle(self.yaw);
        const transform = zm.mul(zm.rotationX(self.pitch), zm.rotationY(self.yaw));
        const forward = zm.normalize3(zm.mul(zm.f32x4(0.0, 0.0, 1.0, 0.0), transform));
        zm.store(&self.forward, forward, 3);
    }

    pub fn moveForward(self: *Camera, speed: f32) void {
        const forward = zm.load(&self.forward, zm.Vec, 3);
        var pos = zm.load(&self.pos, zm.Vec, 3);
        pos += zm.f32x4s(speed) * forward;
        zm.store(&self.pos, pos, 3);
    }

    pub fn moveBackward(self: *Camera, speed: f32) void {
        const forward = zm.load(&self.forward, zm.Vec, 3);
        var pos = zm.load(&self.pos, zm.Vec, 3);
        pos -= zm.f32x4s(speed) * forward;
        zm.store(&self.pos, pos, 3);
    }

    pub fn moveRight(self: *Camera, speed: f32) void {
        const forward = zm.load(&self.forward, zm.Vec, 3);
        const right = zm.normalize3(zm.cross3(zm.f32x4(0.0, 1.0, 0.0, 0.0), forward));
        var pos = zm.load(&self.pos, zm.Vec, 3);
        pos += zm.f32x4s(speed) * right;
        zm.store(&self.pos, pos, 3);
    }

    pub fn moveLeft(self: *Camera, speed: f32) void {
        const forward = zm.load(&self.forward, zm.Vec, 3);
        const right = zm.normalize3(zm.cross3(zm.f32x4(0.0, 1.0, 0.0, 0.0), forward));
        var pos = zm.load(&self.pos, zm.Vec, 3);
        pos -= zm.f32x4s(speed) * right;
        zm.store(&self.pos, pos, 3);
    }

    pub fn moveUp(self: *Camera, speed: f32) void {
        const up = zm.f32x4(0.0, 1.0, 0.0, 0.0);
        var pos = zm.load(&self.pos, zm.Vec, 3);
        pos += zm.f32x4s(speed) * up;
        zm.store(&self.pos, pos, 3);
    }

    pub fn moveDown(self: *Camera, speed: f32) void {
        const up = zm.f32x4(0.0, 1.0, 0.0, 0.0);
        var pos = zm.load(&self.pos, zm.Vec, 3);
        pos -= zm.f32x4s(speed) * up;
        zm.store(&self.pos, pos, 3);
    }
};

pub const MeshVertex = struct {
    position: [3]f32,
    normal: [3]f32,
    texcoord0: [2]f32,
    tangent: [4]f32
};
comptime { std.debug.assert(@sizeOf([2]MeshVertex) == 96); }

pub const SubMesh = struct {
    indices_start_index: u32,
    vertices_start_index: u32,
    index_count: u32,
    vertex_count: u32,
    material_index: u32
};

pub const MeshTexture = struct {
    source: [:0]u8,
    image: ?zstbi.Image // image arena
};

const MeshMaterial = struct {
    base_color: [4]f32,
    metallic: f32,
    roughness: f32,
    base_color_tex_index: ?u32,
    metallic_roughness_tex_index: ?u32,
    normal_tex_index: ?u32
};

pub const Mesh = struct {
    allocator: std.mem.Allocator,
    source: ?[:0]u8,
    submeshes: std.ArrayList(SubMesh),
    vertices: std.ArrayList(MeshVertex), // mesh arena
    indices: std.ArrayList(u32), // mesh arena
    vertex_stride: u32,
    index_stride: u32,
    index_format: dxgi.FORMAT,
    textures: std.ArrayList(MeshTexture),
    materials: std.ArrayList(MeshMaterial),

    pub fn init(allocator: std.mem.Allocator, mesh_arena: std.mem.Allocator) Mesh {
        return Mesh {
            .allocator = allocator,
            .source = null,
            .submeshes = std.ArrayList(SubMesh).init(allocator),
            .vertices = std.ArrayList(MeshVertex).init(mesh_arena),
            .indices = std.ArrayList(u32).init(mesh_arena),
            .vertex_stride = @sizeOf(MeshVertex),
            .index_stride = @sizeOf(u32),
            .index_format = .R32_UINT,
            .textures = std.ArrayList(MeshTexture).init(allocator),
            .materials = std.ArrayList(MeshMaterial).init(allocator)
        };
    }

    pub fn deinit(self: *Mesh) void {
        self.materials.deinit();
        for (self.textures.items) |texture| {
            self.allocator.free(texture.source);
        }
        self.textures.deinit();
        self.submeshes.deinit();
        if (self.source) |s| {
            self.allocator.free(s);
        }
    }
};

pub const Fw = struct {
    pub const Options = struct {
        window_size: Size = .{ .width = 1280, .height = 720 },
        window_name: [*:0]const u8 = "zigapp",
        enable_debug_layer: bool = false,
        swap_interval: u32 = 1,
        staging_area_capacity_per_frame: u32 = 32 * 1024 * 1024,
        shader_visible_permanent_cbv_srv_uav_heap_range_capacity: u32 = 256,
        shader_visible_per_frame_cbv_srv_uav_heap_range_capacity: u32 = 1024,
        shader_visible_permanent_sampler_heap_range_capacity: u32 = 16,
        shader_visible_per_frame_sampler_heap_range_capacity: u32 = 64,
    };

    pub const max_frames_in_flight = 2;
    const swapchain_buffer_count = 3;
    const transition_resource_barrier_buffer_size = 16;
    pub const dsv_format = dxgi.FORMAT.D24_UNORM_S8_UINT;

    const SwapchainBuffer = struct {
        handle: ObjectHandle,
        descriptor: Descriptor
    };

    const TransitionResourceBarrier = struct {
        resource_handle: ObjectHandle,
        state_before: d3d12.RESOURCE_STATES,
        state_after: d3d12.RESOURCE_STATES
    };

    const DeferredReleaseEntry = struct {
        const Type = enum {
            Resource,
            Pipeline,
            Callback
        };
        rtype: Type,
        handle: ObjectHandle = ObjectHandle.invalid(),
        frame_slot_to_be_released_in: ?u32 = null,
        view_pool: ?*CpuDescriptorPool = null,
        view_start: ?Descriptor = null,
        view_count: ?u32 = null,
        callback_fn: ?*const fn (*anyopaque) void = null,
        callback_user_data: ?*anyopaque = null
    };

    const ImguiWData = struct {
        mouse_window: ?w32.HWND = null,
        mouse_tracked: bool = false,
        mouse_buttons_down: u32 = 0
    };

    const CameraWData = struct {
        last_cursor_pos: w32.POINT,

        fn init() CameraWData {
            var pos: w32.POINT = undefined;
            _ = w32.GetCursorPos(&pos);
            return CameraWData {
                .last_cursor_pos = pos
            };
        }
    };

    const Data = struct {
        options: Options,
        instance: w32.HINSTANCE,
        window: w32.HWND,
        window_size: Size,
        dxgiFactory: *dxgi.IFactory6,
        device: *d3d12.IDevice9,
        cmdqueue: *d3d12.ICommandQueue,
        swapchain: *dxgi.ISwapChain3,
        swapchain_flags: u32,
        swapchain_size: Size,
        frame_fence: *d3d12.IFence,
        frame_fence_event: w32.HANDLE,
        frame_fence_counter: u64,
        cmd_allocators: [max_frames_in_flight]*d3d12.ICommandAllocator,
        cmd_list: *d3d12.IGraphicsCommandList6,
        rtv_pool: CpuDescriptorPool,
        dsv_pool: CpuDescriptorPool,
        swapchain_buffers: [swapchain_buffer_count]SwapchainBuffer,
        current_frame_slot: u32,
        current_back_buffer_index: u32,
        present_allow_tearing_supported: bool,
        resource_pool: ObjectPool(Resource),
        pipeline_pool: ObjectPool(Pipeline),
        pipeline_cache: Pipeline.Cache,
        release_queue: std.ArrayList(DeferredReleaseEntry),
        transition_resource_barriers: []TransitionResourceBarrier,
        trb_next: u32,
        staging_areas: [max_frames_in_flight]StagingArea,
        shader_visible_cbv_srv_uav_heap: DescriptorHeap,
        shader_visible_permanent_cbv_srv_uav_heap_range: DescriptorHeap,
        shader_visible_per_frame_cbv_srv_uav_heap_ranges: [max_frames_in_flight]DescriptorHeap,
        shader_visible_sampler_heap: DescriptorHeap,
        shader_visible_permanent_sampler_heap_range: DescriptorHeap,
        shader_visible_per_frame_sampler_heap_ranges: [max_frames_in_flight]DescriptorHeap,
        depth_stencil_buffer: ObjectHandle,
        dsv: Descriptor,
        current_pipeline_handle: ObjectHandle,
        imgui_font_data: ?[]u8,
        imgui_font_texture: ObjectHandle,
        imgui_font_srv: Descriptor,
        imgui_pipeline: ObjectHandle,
        imgui_vbuf: [max_frames_in_flight]HostVisibleBuffer,
        imgui_ibuf: [max_frames_in_flight]HostVisibleBuffer,
        imgui_wdata: ImguiWData,
        camera_wdata: CameraWData,
        mipmapgen_pipeline: ObjectHandle,
        dummy_texture_handle: ObjectHandle,
        dummy_texture: ?Descriptor,
        format_work_area: std.ArrayList(u8),
        sampler_cache_map: std.HashMap(SamplerCacheKey, Descriptor, SamplerCacheHashContext, 80),
        sampler_cpu_pool: CpuDescriptorPool,
        mesh_arena: std.heap.ArenaAllocator,
        image_arena: std.heap.ArenaAllocator,
        frame_timer: std.time.Timer,
        elapsed_begin_to_begin_ns: u64,
    };

    allocator: std.mem.Allocator,
    d: *Data,

    fn fromWCHAR(work: []u8, src: []const u16) []const u8 {
        const len = std.unicode.utf16leToUtf8(work, src) catch 0;
        if (len > 0) {
            for (work) | c, idx | {
                if (c == 0) {
                    return work[0..idx];
                }
            }
        }
        return &.{};
    }

    pub fn init(allocator: std.mem.Allocator, options: Options) !Fw {
        var d = allocator.create(Data) catch unreachable;
        errdefer allocator.destroy(d);
        d.options = options;

        _ = imgui.igCreateContext(null);
        errdefer imgui.igDestroyContext(null);
        var io = imgui.igGetIO().?;
        io.*.BackendPlatformUserData = d;
        io.*.BackendFlags |= imgui.ImGuiBackendFlags_RendererHasVtxOffset;
        io.*.IniFilename = null;
        io.*.FontAllowUserScaling = true;

        _ = w32.ole32.CoInitializeEx(null, @enumToInt(w32.COINIT_APARTMENTTHREADED) | @enumToInt(w32.COINIT_DISABLE_OLE1DDE));
        errdefer w32.ole32.CoUninitialize();

        _ = w32.SetProcessDPIAware(); // no scaling by the system, but we won't care either -> always in pixels

        d.instance = @ptrCast(w32.HINSTANCE, w32.kernel32.GetModuleHandleW(null));

        const winclass = w32.user32.WNDCLASSEXA {
            .style = 0,
            .lpfnWndProc = processWindowMessage,
            .cbClsExtra = 0,
            .cbWndExtra = 0,
            .hInstance = d.instance,
            .hIcon = null,
            .hCursor = w32.LoadCursorA(null, @intToPtr(w32.LPCSTR, 32512)),
            .hbrBackground = null,
            .lpszMenuName = null,
            .lpszClassName = options.window_name,
            .hIconSm = null,
        };
        _ = try w32.user32.registerClassExA(&winclass);

        // will be first updated on WM_SIZE
        d.window_size = Size.empty();

        const style = w32.user32.WS_OVERLAPPEDWINDOW; // resizable, minimize, maximize
        var rect = w32.RECT {
            .left = 0,
            .top = 0,
            .right = @intCast(i32, options.window_size.width),
            .bottom = @intCast(i32, options.window_size.height)
        };
        try w32.user32.adjustWindowRectEx(&rect, style, false, 0);

        d.window = try w32.user32.createWindowExA(
            0,
            options.window_name,
            options.window_name,
            style + w32.user32.WS_VISIBLE,
            w32.user32.CW_USEDEFAULT,
            w32.user32.CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            null,
            null,
            winclass.hInstance,
            null,
        );

        var factory: *dxgi.IFactory6 = undefined;
        try zwin32.hrErrorOnFail(dxgi.CreateDXGIFactory2(
            if (options.enable_debug_layer) dxgi.CREATE_FACTORY_DEBUG else 0,
            &dxgi.IID_IFactory6,
            @ptrCast(*?*anyopaque, &factory)));
        errdefer _ = factory.Release();
        d.dxgiFactory = factory;

        var allow_tearing_supported: w32.BOOL = w32.FALSE;
        try zwin32.hrErrorOnFail(factory.CheckFeatureSupport(dxgi.FEATURE.PRESENT_ALLOW_TEARING,
                                                             &allow_tearing_supported,
                                                             @sizeOf(w32.BOOL)));
        d.present_allow_tearing_supported = if (allow_tearing_supported != 0) true else false;

        if (options.enable_debug_layer) {
            var maybe_debug: ?*d3d12d.IDebug1 = null;
            _ = d3d12.D3D12GetDebugInterface(&d3d12d.IID_IDebug1, @ptrCast(*?*anyopaque, &maybe_debug));
            if (maybe_debug) |debug| {
                std.debug.print("Enabling debug layer\n", .{});
                debug.EnableDebugLayer();
                _ = debug.Release();
            }
        }

        std.debug.print("allow_tearing_supported = {}\n", .{d.present_allow_tearing_supported});

        var chosen_adapter: ?*dxgi.IAdapter1 = null;
        var maybe_adapter1: ?*dxgi.IAdapter1 = null;
        var adapter_index: u32 = 0;
        while (factory.EnumAdapterByGpuPreference(adapter_index,
                                                  dxgi.GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                  &dxgi.IID_IAdapter1,
                                                  &maybe_adapter1) == w32.S_OK) : (adapter_index += 1) {
            if (maybe_adapter1) |adapter1| {
                var desc: dxgi.ADAPTER_DESC1 = undefined;
                if (adapter1.GetDesc1(&desc) == w32.S_OK) {
                    var tmp = [_:0]u8 { 0 } ** 256;
                    std.debug.print("Adapter {}: {s} flags=0x{X}\n", .{adapter_index, fromWCHAR(&tmp, &desc.Description), desc.Flags});
                    if (chosen_adapter == null) {
                        chosen_adapter = adapter1;
                        std.debug.print("  using this adapter\n", .{});
                    } else {
                        _ = adapter1.Release();
                    }
                }
            }
        }
        defer if (chosen_adapter) |adapter| { _ = adapter.Release(); };

        var device: *d3d12.IDevice9 = undefined;
        try zwin32.hrErrorOnFail(d3d12.D3D12CreateDevice(if (chosen_adapter) |adapter| @ptrCast(*w32.IUnknown, adapter) else null,
                                                         .FL_12_0,
                                                         &d3d12.IID_IDevice9,
                                                         @ptrCast(*?*anyopaque, &device)));
        errdefer _ = device.Release();
        d.device = device;

        var cmdqueue: *d3d12.ICommandQueue = undefined;
        try zwin32.hrErrorOnFail(device.CreateCommandQueue(
            &.{
                .Type = .DIRECT,
                .Priority = @enumToInt(d3d12.COMMAND_QUEUE_PRIORITY.NORMAL),
                .Flags = d3d12.COMMAND_QUEUE_FLAG_NONE,
                .NodeMask = 0
            },
            &d3d12.IID_ICommandQueue, @ptrCast(*?*anyopaque, &cmdqueue)));
        errdefer _ = cmdqueue.Release();
        d.cmdqueue = cmdqueue;

        d.swapchain_size = d.window_size;

        var swapchain: *dxgi.ISwapChain1 = undefined;
        d.swapchain_flags = 0;
        if (options.swap_interval == 0 and d.present_allow_tearing_supported)
            d.swapchain_flags |= dxgi.SWAP_CHAIN_FLAG_ALLOW_TEARING;
        const swapchainDesc = &dxgi.SWAP_CHAIN_DESC1 {
            .Width = d.swapchain_size.width,
            .Height = d.swapchain_size.height,
            .Format = .R8G8B8A8_UNORM,
            .Stereo = w32.FALSE,
            .SampleDesc = .{ .Count = 1, .Quality = 0 },
            .BufferUsage = dxgi.USAGE_RENDER_TARGET_OUTPUT,
            .BufferCount = swapchain_buffer_count,
            .Scaling = .NONE,
            .SwapEffect = .FLIP_DISCARD,
            .AlphaMode = .UNSPECIFIED,
            .Flags = d.swapchain_flags
        };
        try zwin32.hrErrorOnFail(factory.CreateSwapChainForHwnd(@ptrCast(*w32.IUnknown, cmdqueue),
                                                                d.window,
                                                                swapchainDesc,
                                                                null,
                                                                null,
                                                                @ptrCast(*?*dxgi.ISwapChain1, &swapchain)));
        errdefer _ = swapchain.Release();

        var swapchain3: *dxgi.ISwapChain3 = undefined;
        try zwin32.hrErrorOnFail(swapchain.QueryInterface(&dxgi.IID_ISwapChain3, @ptrCast(*?*anyopaque, &swapchain3)));
        errdefer _ = swapchain3.Release();
        defer _ = swapchain.Release();
        d.swapchain = swapchain3;

        var frame_fence: *d3d12.IFence = undefined;
        try zwin32.hrErrorOnFail(device.CreateFence(0,
                                                    d3d12.FENCE_FLAG_NONE,
                                                    &d3d12.IID_IFence,
                                                    @ptrCast(*?*anyopaque, &frame_fence)));
        errdefer _ = frame_fence.Release();
        d.frame_fence = frame_fence;

        const frame_fence_event = try w32.CreateEventEx(null,
                                                        "frame_fence_event",
                                                        0,
                                                        w32.EVENT_ALL_ACCESS);
        errdefer w32.CloseHandle(frame_fence_event);
        d.frame_fence_event = frame_fence_event;

        d.frame_fence_counter = 0;

        var cmd_allocators: [max_frames_in_flight]*d3d12.ICommandAllocator = undefined;
        for (cmd_allocators) |_, index| {
            try zwin32.hrErrorOnFail(device.CreateCommandAllocator(.DIRECT,
                                                                   &d3d12.IID_ICommandAllocator,
                                                                   @ptrCast(*?*anyopaque, &cmd_allocators[index])));
        }
        errdefer for (cmd_allocators) |cmd_allocator| { _ = cmd_allocator.Release(); };
        d.cmd_allocators = cmd_allocators;

        var cmd_list: *d3d12.IGraphicsCommandList6 = undefined;
        try zwin32.hrErrorOnFail(device.CreateCommandList(0, .DIRECT, cmd_allocators[0], null,
                                                          &d3d12.IID_IGraphicsCommandList6,
                                                          @ptrCast(*?*anyopaque, &cmd_list)));
        errdefer _ = cmd_list.Release();
        d.cmd_list = cmd_list;
        try zwin32.hrErrorOnFail(cmd_list.Close());

        d.rtv_pool = try CpuDescriptorPool.init(allocator, device, .RTV);
        errdefer d.rtv_pool.deinit();

        d.dsv_pool = try CpuDescriptorPool.init(allocator, device, .DSV);
        errdefer d.dsv_pool.deinit();

        d.resource_pool = try ObjectPool(Resource).init(allocator);
        errdefer d.resource_pool.deinit();

        d.pipeline_pool = try ObjectPool(Pipeline).init(allocator);
        errdefer d.pipeline_pool.deinit();

        d.pipeline_cache = Pipeline.Cache.init(allocator);
        errdefer d.pipeline_cache.deinit();

        d.release_queue = std.ArrayList(DeferredReleaseEntry).init(allocator);

        d.transition_resource_barriers = try allocator.alloc(TransitionResourceBarrier, transition_resource_barrier_buffer_size);
        errdefer allocator.free(d.transition_resource_barriers);
        d.trb_next = 0;

        for (d.staging_areas) |_, index| {
            d.staging_areas[index] = try StagingArea.init(d.device,
                                                          d.options.staging_area_capacity_per_frame,
                                                          .UPLOAD);
        }

        var gpu_heap_total_capacity =
            d.options.shader_visible_permanent_cbv_srv_uav_heap_range_capacity
            + max_frames_in_flight * d.options.shader_visible_per_frame_cbv_srv_uav_heap_range_capacity;
        d.shader_visible_cbv_srv_uav_heap = try DescriptorHeap.init(d.device,
                                                                    gpu_heap_total_capacity,
                                                                    .CBV_SRV_UAV,
                                                                    d3d12.DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        errdefer d.shader_visible_cbv_srv_uav_heap.deinit();

        d.shader_visible_permanent_cbv_srv_uav_heap_range = DescriptorHeap.initWithExisting(
            d.shader_visible_cbv_srv_uav_heap,
            0,
            d.options.shader_visible_permanent_cbv_srv_uav_heap_range_capacity);

        var heap_range_offset = d.options.shader_visible_permanent_cbv_srv_uav_heap_range_capacity;
        var heap_range_size = d.options.shader_visible_per_frame_cbv_srv_uav_heap_range_capacity;
        for (d.shader_visible_per_frame_cbv_srv_uav_heap_ranges) |_, index| {
            d.shader_visible_per_frame_cbv_srv_uav_heap_ranges[index] = DescriptorHeap.initWithExisting(
                d.shader_visible_cbv_srv_uav_heap,
                heap_range_offset,
                heap_range_size);
            heap_range_offset += heap_range_size;
        }

        gpu_heap_total_capacity =
            d.options.shader_visible_permanent_sampler_heap_range_capacity
            + max_frames_in_flight * d.options.shader_visible_per_frame_sampler_heap_range_capacity;
        d.shader_visible_sampler_heap = try DescriptorHeap.init(d.device,
                                                                gpu_heap_total_capacity,
                                                                .SAMPLER,
                                                                d3d12.DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        errdefer d.shader_visible_sampler_heap.deinit();

        d.shader_visible_permanent_sampler_heap_range = DescriptorHeap.initWithExisting(
            d.shader_visible_sampler_heap,
            0,
            d.options.shader_visible_permanent_sampler_heap_range_capacity);

        heap_range_offset = d.options.shader_visible_permanent_sampler_heap_range_capacity;
        heap_range_size = d.options.shader_visible_per_frame_sampler_heap_range_capacity;
        for (d.shader_visible_per_frame_sampler_heap_ranges) |_, index| {
            d.shader_visible_per_frame_sampler_heap_ranges[index] = DescriptorHeap.initWithExisting(
                d.shader_visible_sampler_heap,
                heap_range_offset,
                heap_range_size);
            heap_range_offset += heap_range_size;
        }

        d.current_frame_slot = 0;

        d.depth_stencil_buffer = ObjectHandle.invalid();
        d.dsv = Descriptor.invalid();
        d.current_pipeline_handle = ObjectHandle.invalid();
        d.imgui_font_data = null;
        d.imgui_font_texture = ObjectHandle.invalid();
        d.imgui_font_srv = Descriptor.invalid();
        d.imgui_pipeline = ObjectHandle.invalid();
        d.imgui_vbuf = [_]HostVisibleBuffer { .{} } ** max_frames_in_flight;
        d.imgui_ibuf = [_]HostVisibleBuffer { .{} } ** max_frames_in_flight;
        d.imgui_wdata = ImguiWData { };
        d.camera_wdata = CameraWData.init();
        d.mipmapgen_pipeline = ObjectHandle.invalid();
        d.dummy_texture_handle = ObjectHandle.invalid();
        d.dummy_texture = null;

        d.format_work_area = std.ArrayList(u8).init(allocator);

        d.sampler_cache_map = std.HashMap(SamplerCacheKey, Descriptor, SamplerCacheHashContext, 80).init(allocator);
        d.sampler_cpu_pool = try CpuDescriptorPool.init(allocator, device, .SAMPLER);
        errdefer d.sampler_cpu_pool.deinit();

        d.mesh_arena = std.heap.ArenaAllocator.init(allocator);
        errdefer d.mesh_arena.deinit();
        zmesh.init(d.mesh_arena.allocator());
        errdefer zmesh.deinit();

        d.image_arena = std.heap.ArenaAllocator.init(allocator);
        errdefer d.image_arena.deinit();
        zstbi.init(d.image_arena.allocator());
        errdefer zstbi.deinit();

        d.frame_timer = std.time.Timer.start() catch unreachable;
        d.elapsed_begin_to_begin_ns = 0;

        var self = Fw {
            .allocator = allocator,
            .d = d
        };

        try self.acquireSwapchainBuffers();
        try self.ensureDepthStencil();

        return self;
    }

    pub fn deinit(self: *Fw) void {
        self.waitGpu();
        for (self.d.swapchain_buffers) |swapchain_buffer| {
            self.d.resource_pool.remove(swapchain_buffer.handle);
        }
        self.d.sampler_cpu_pool.deinit();
        self.d.sampler_cache_map.deinit();
        self.d.format_work_area.deinit();
        if (self.d.imgui_font_data) |imgui_font_data| {
            self.allocator.free(imgui_font_data);
        }
        self.d.shader_visible_sampler_heap.deinit();
        self.d.shader_visible_cbv_srv_uav_heap.deinit();
        for (self.d.staging_areas) |*staging_area| {
            staging_area.deinit();
        }
        self.allocator.free(self.d.transition_resource_barriers);
        self.d.release_queue.deinit();
        self.d.pipeline_cache.deinit();
        self.d.pipeline_pool.deinit();
        self.d.resource_pool.deinit();
        self.d.dsv_pool.deinit();
        self.d.rtv_pool.deinit();
        _ = self.d.cmd_list.Release();
        for (self.d.cmd_allocators) |cmd_allocator| {
            _ = cmd_allocator.Release();
        }
        w32.CloseHandle(self.d.frame_fence_event);
        _ = self.d.frame_fence.Release();
        _ = self.d.swapchain.Release();
        _ = self.d.cmdqueue.Release();
        _ = self.d.device.Release();
        _ = self.d.dxgiFactory.Release();
        imgui.igDestroyContext(null);
        zmesh.deinit();
        self.d.mesh_arena.deinit();
        zstbi.deinit();
        self.d.image_arena.deinit();
        self.allocator.destroy(self.d);
        w32.ole32.CoUninitialize();
    }

    fn acquireSwapchainBuffers(self: *Fw) !void {
        var swapchain_buffers: [swapchain_buffer_count]SwapchainBuffer = undefined;
        for (swapchain_buffers) |_, index| {
            var res: *d3d12.IResource = undefined;
            try zwin32.hrErrorOnFail(self.d.swapchain.GetBuffer(@intCast(u32, index),
                                                                &d3d12.IID_IResource,
                                                                @ptrCast(*?*anyopaque, &res)));
            const rtDesc = &d3d12.RENDER_TARGET_VIEW_DESC {
                .Format = .R8G8B8A8_UNORM,
                .ViewDimension = .TEXTURE2D,
                .u = .{
                    .Texture2D = .{
                        .MipSlice = 0,
                        .PlaneSlice = 0,
                    }
                }
            };
            const handle = blk: {
                errdefer _ = res.Release();
                break :blk try Resource.addToPool(&self.d.resource_pool, res, d3d12.RESOURCE_STATE_PRESENT);
            };
            swapchain_buffers[index] = SwapchainBuffer {
                .handle = handle,
                .descriptor = try self.d.rtv_pool.allocate(1)
            };
            self.d.device.CreateRenderTargetView(res,
                                                 rtDesc,
                                                 swapchain_buffers[index].descriptor.cpu_handle);
        }
        self.d.swapchain_buffers = swapchain_buffers;
        self.d.current_back_buffer_index = self.d.swapchain.GetCurrentBackBufferIndex();
    }

    fn ensureDepthStencil(self: *Fw) !void {
        if (self.d.resource_pool.lookupRef(self.d.depth_stencil_buffer)) |res| {
            if (res.desc.Width == self.d.swapchain_size.width and res.desc.Height == self.d.swapchain_size.height) {
                return;
            }
            self.d.resource_pool.remove(self.d.depth_stencil_buffer);
            self.d.depth_stencil_buffer = ObjectHandle.invalid();
            self.d.dsv_pool.release(self.d.dsv, 1);
            self.d.dsv = Descriptor.invalid();
        }
        self.d.depth_stencil_buffer = try self.createTexture2D(dsv_format,
                                                               self.d.swapchain_size,
                                                               1,
                                                               d3d12.RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | d3d12.RESOURCE_FLAG_DENY_SHADER_RESOURCE,
                                                               d3d12.RESOURCE_STATE_DEPTH_WRITE,
                                                               &.{
                                                                   .Format = Fw.dsv_format,
                                                                   .u = .{
                                                                       .DepthStencil = .{
                                                                           .Depth = 1.0,
                                                                           .Stencil = 0
                                                                       }
                                                                   }
        });
        self.d.dsv = try self.d.dsv_pool.allocate(1);
        self.d.device.CreateDepthStencilView(self.d.resource_pool.lookupRef(self.d.depth_stencil_buffer).?.resource,
                                             null,
                                             self.d.dsv.cpu_handle);
    }

    pub fn waitGpu(self: *Fw) void {
        self.d.frame_fence_counter += 1;
        zwin32.hrErrorOnFail(self.d.cmdqueue.Signal(self.d.frame_fence, self.d.frame_fence_counter)) catch unreachable;
        zwin32.hrErrorOnFail(self.d.frame_fence.SetEventOnCompletion(self.d.frame_fence_counter, self.d.frame_fence_event)) catch unreachable;
        w32.WaitForSingleObject(self.d.frame_fence_event, w32.INFINITE) catch unreachable;
    }

    fn waitGpuIfAhead(self: *Fw) void {
        self.d.frame_fence_counter += 1;
        zwin32.hrErrorOnFail(self.d.cmdqueue.Signal(self.d.frame_fence, self.d.frame_fence_counter)) catch unreachable;
        if (self.d.frame_fence_counter >= max_frames_in_flight) {
            const completed_value = self.d.frame_fence.GetCompletedValue();
            if (completed_value <= self.d.frame_fence_counter - max_frames_in_flight)  {
                zwin32.hrErrorOnFail(self.d.frame_fence.SetEventOnCompletion(completed_value + 1, self.d.frame_fence_event)) catch unreachable;
                w32.WaitForSingleObject(self.d.frame_fence_event, w32.INFINITE) catch unreachable;
            }
        }
    }

    pub fn addTransitionBarrier(self: *Fw, resource_handle: ObjectHandle, state_after: d3d12.RESOURCE_STATES) void {
        var res = self.d.resource_pool.lookupRef(resource_handle) orelse return;
        if (state_after != res.state) {
            if (self.d.trb_next == self.d.transition_resource_barriers.len)
                self.recordTransitionBarriers();

            self.d.transition_resource_barriers[self.d.trb_next] = TransitionResourceBarrier {
                .resource_handle = resource_handle,
                .state_before = res.state,
                .state_after = state_after
            };
            self.d.trb_next += 1;                
            res.state = state_after;
        }
    }

    pub fn recordTransitionBarriers(self: *Fw) void {
        var barriers: [transition_resource_barrier_buffer_size]d3d12.RESOURCE_BARRIER = undefined;
        var count: u32 = 0;
        var i: u32 = 0;
        while (i < self.d.trb_next) : (i += 1) {
            const trb = self.d.transition_resource_barriers[i];
            if (!self.d.resource_pool.isValid(trb.resource_handle))
                continue;
            barriers[count] = .{
                .Type = .TRANSITION,
                .Flags = d3d12.RESOURCE_BARRIER_FLAG_NONE,
                .u = .{
                    .Transition = .{
                        .pResource = self.d.resource_pool.lookupRef(trb.resource_handle).?.resource,
                        .Subresource = d3d12.RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        .StateBefore = trb.state_before,
                        .StateAfter = trb.state_after,
                    },
                },
            };
            count += 1;
        }
        if (count > 0) {
            self.d.cmd_list.ResourceBarrier(count, &barriers);
        }
        self.d.trb_next = 0;
    }

    pub fn recordSubresourceTransitionBarrier(self: *Fw,
                                              resource_handle: ObjectHandle,
                                              subresource: u32,
                                              state_before: d3d12.RESOURCE_STATES,
                                              state_after: d3d12.RESOURCE_STATES) void {
        var res = self.d.resource_pool.lookupRef(resource_handle) orelse return;
        self.d.cmd_list.ResourceBarrier(1, &[_]d3d12.RESOURCE_BARRIER {
            .{
                .Type = .TRANSITION,
                .Flags = d3d12.RESOURCE_BARRIER_FLAG_NONE,
                .u = .{
                    .Transition = .{
                        .pResource = res.resource,
                        .Subresource = subresource,
                        .StateBefore = state_before,
                        .StateAfter = state_after
                    }
                }
            }
        });
    }

    pub fn recordUavBarrier(self: *Fw, resource_handle: ObjectHandle) void {
        var res = self.d.resource_pool.lookupRef(resource_handle) orelse return;
        self.d.cmd_list.ResourceBarrier(1, &[_]d3d12.RESOURCE_BARRIER {
            .{
                .Type = .UAV,
                .Flags = d3d12.RESOURCE_BARRIER_FLAG_NONE,
                .u = .{
                    .UAV = .{
                        .pResource = res.resource
                    }
                }
            }
        });
    }

    pub fn getDevice(self: *const Fw) *d3d12.IDevice9 {
        return self.d.device;
    }

    pub fn getCommandList(self: *const Fw) *d3d12.IGraphicsCommandList6 {
        return self.d.cmd_list;
    }

    pub fn getResourcePool(self: *const Fw) *ObjectPool(Resource) {
        return &self.d.resource_pool;
    }

    pub fn getPipelinePool(self: *const Fw) *ObjectPool(Pipeline) {
        return &self.d.pipeline_pool;
    }

    pub fn getPipelineCache(self: *const Fw) *Pipeline.Cache {
        return &self.d.pipeline_cache;
    }

    pub fn getCurrentFrameSlot(self: *const Fw) u32 {
        return self.d.current_frame_slot;
    }

    pub fn getCurrentStagingArea(self: *const Fw) *StagingArea {
        return &self.d.staging_areas[self.d.current_frame_slot];
    }

    pub fn getShaderVisibleCbvSrvUavHeap(self: *const Fw) *d3d12.IDescriptorHeap {
        return self.d.shader_visible_cbv_srv_uav_heap.heap.?;
    }

    pub fn getPermanentShaderVisibleCbvSrvUavHeapRange(self: *const Fw) *DescriptorHeap {
        return &self.d.shader_visible_permanent_cbv_srv_uav_heap_range;
    }

    pub fn getCurrentShaderVisibleCbvSrvUavHeapRange(self: *const Fw) *DescriptorHeap {
        return &self.d.shader_visible_per_frame_cbv_srv_uav_heap_ranges[self.d.current_frame_slot];
    }

    pub fn getShaderVisibleSamplerHeap(self: *const Fw) *d3d12.IDescriptorHeap {
        return self.d.shader_visible_sampler_heap.heap.?;
    }

    pub fn getPermanentShaderVisibleSamplerHeapRange(self: *const Fw) *DescriptorHeap {
        return &self.d.shader_visible_permanent_sampler_heap_range;
    }

    pub fn getCurrentShaderVisibleSamplerHeapRange(self: *const Fw) *DescriptorHeap {
        return &self.d.shader_visible_per_frame_sampler_heap_ranges[self.d.current_frame_slot];
    }

    pub fn getDepthStencilBufferCpuDescriptorHandle(self: *const Fw) d3d12.CPU_DESCRIPTOR_HANDLE {
        return self.d.dsv.cpu_handle;
    }

    pub fn getBackBufferCpuDescriptorHandle(self: *const Fw) d3d12.CPU_DESCRIPTOR_HANDLE {
        return self.d.swapchain_buffers[self.d.current_back_buffer_index].descriptor.cpu_handle;
    }

    pub fn getBackBufferObjectHandle(self: *const Fw) ObjectHandle {
        return self.d.swapchain_buffers[self.d.current_back_buffer_index].handle;
    }

    pub fn getBackBufferPixelSize(self: *const Fw) Size {
        return self.d.swapchain_size;
    }

    pub const BeginFrameResult = enum {
        success,
        empty_output_size
    };

    pub fn beginFrame(self: *Fw) !BeginFrameResult {
        self.waitGpuIfAhead();
        self.drainReleaseQueue();

        if (self.d.window_size.isEmpty()) { // e.g. when minimized
            return BeginFrameResult.empty_output_size;
        }

        if (!std.meta.eql(self.d.swapchain_size, self.d.window_size)) {
            self.d.swapchain_size = self.d.window_size;
            std.debug.print("Resizing swapchain {}x{}\n", .{self.d.swapchain_size.width, self.d.swapchain_size.height});
            self.waitGpu();
            for (self.d.swapchain_buffers) |*swapchain_buffer| {
                self.d.resource_pool.remove(swapchain_buffer.handle);
                self.d.rtv_pool.release(swapchain_buffer.descriptor, 1);
            }
            try zwin32.hrErrorOnFail(self.d.swapchain.ResizeBuffers(swapchain_buffer_count,
                                                                    self.d.swapchain_size.width,
                                                                    self.d.swapchain_size.height,
                                                                    .R8G8B8A8_UNORM,
                                                                    self.d.swapchain_flags));
            try self.acquireSwapchainBuffers();
            try self.ensureDepthStencil();
        }

        const cmd_allocator = self.d.cmd_allocators[self.d.current_frame_slot];
        try zwin32.hrErrorOnFail(cmd_allocator.Reset());
        try zwin32.hrErrorOnFail(self.d.cmd_list.Reset(cmd_allocator, null));

        self.d.cmd_list.SetDescriptorHeaps(2, &[_]*d3d12.IDescriptorHeap {
            self.getShaderVisibleCbvSrvUavHeap(),
            self.getShaderVisibleSamplerHeap()
        });

        self.addTransitionBarrier(self.getBackBufferObjectHandle(), d3d12.RESOURCE_STATE_RENDER_TARGET);
        self.recordTransitionBarriers();

        self.resetTrackedState();

        self.d.staging_areas[self.d.current_frame_slot].reset();
        self.d.shader_visible_per_frame_cbv_srv_uav_heap_ranges[self.d.current_frame_slot].reset();
        self.d.shader_visible_per_frame_sampler_heap_ranges[self.d.current_frame_slot].reset();

        self.d.elapsed_begin_to_begin_ns = self.d.frame_timer.lap();

        return BeginFrameResult.success;
    }

    pub fn endFrame(self: *Fw) !void {
        self.addTransitionBarrier(self.getBackBufferObjectHandle(), d3d12.RESOURCE_STATE_PRESENT);
        self.recordTransitionBarriers();

        try zwin32.hrErrorOnFail(self.d.cmd_list.Close());

        const list = [_]*d3d12.ICommandList {
            @ptrCast(*d3d12.ICommandList, self.d.cmd_list)
        };
        self.d.cmdqueue.ExecuteCommandLists(1, &list);

        var present_flags: w32.UINT = 0;
        if (self.d.options.swap_interval == 0 and self.d.present_allow_tearing_supported)
            present_flags |= dxgi.PRESENT_ALLOW_TEARING;

        try zwin32.hrErrorOnFail(self.d.swapchain.Present(self.d.options.swap_interval, present_flags));

        self.switchToNextFrameSlot();
        self.d.current_back_buffer_index = self.d.swapchain.GetCurrentBackBufferIndex();
    }

    pub fn getBeginFrameTimeNsecs(self: *const Fw) u64 {
        return self.d.elapsed_begin_to_begin_ns;
    }

    pub fn setPipeline(self: *Fw, pipeline_handle: ObjectHandle) void {
        const pipeline = self.d.pipeline_pool.lookupRef(pipeline_handle) orelse return;
        if (pipeline_handle.index == self.d.current_pipeline_handle.index
                and pipeline_handle.generation == self.d.current_pipeline_handle.generation) {
            return;
        }
        self.d.cmd_list.SetPipelineState(pipeline.pso);
        switch (pipeline.ptype) {
            .Graphics => self.d.cmd_list.SetGraphicsRootSignature(pipeline.rs),
            .Compute => self.d.cmd_list.SetComputeRootSignature(pipeline.rs)
        }
        self.d.current_pipeline_handle = pipeline_handle;
    }

    pub fn resetTrackedState(self: *Fw) void {
        self.d.current_pipeline_handle = ObjectHandle.invalid();
    }

    /// Can be called inside and outside of begin-endFrame. Removes
    /// from the pool and releases the underlying native resource only
    /// in the max_frames_in_flight'th beginFrame() counted starting
    /// from the next endFrame().
    pub fn deferredReleaseResource(self: *Fw, resource_handle: ObjectHandle) void {
        self.d.release_queue.append(.{
            .rtype = .Resource,
            .handle = resource_handle,
        }) catch { };
    }

    pub fn deferredReleaseResourceWithViews(self: *Fw,
                                            resource_handle: ObjectHandle,
                                            view_pool: *CpuDescriptorPool,
                                            view: Descriptor,
                                            view_count: u32) void {
        self.d.release_queue.append(.{
            .rtype = .Resource,
            .handle = resource_handle,
            .view_pool = view_pool,
            .view_start = view,
            .view_count = view_count,
        }) catch { };
    }

    pub fn deferredReleasePipeline(self: *Fw, pipeline_handle: ObjectHandle) void {
        self.d.release_queue.append(.{
            .rtype = .Pipeline,
            .handle = pipeline_handle,
            .frame_slot_to_be_released_in = null
        }) catch { };
    }

    pub fn deferredReleaseCallback(self: *Fw, f: *const fn (*anyopaque) void, user_data: *anyopaque) void {
        self.d.release_queue.append(.{
            .rtype = .Callback,
            .callback_fn = f,
            .callback_user_data = user_data
        }) catch { };
    }

    fn switchToNextFrameSlot(self: *Fw) void {
        // "activate" the pending release requests
        for (self.d.release_queue.items) |*e| {
            if (e.frame_slot_to_be_released_in == null)
                e.frame_slot_to_be_released_in = self.d.current_frame_slot;
        }

        self.d.current_frame_slot = (self.d.current_frame_slot + 1) % max_frames_in_flight;
    }

    fn drainReleaseQueue(self: *Fw) void {
        if (self.d.release_queue.items.len == 0) {
            return;
        }
        var i: i32 = @intCast(i32, self.d.release_queue.items.len) - 1;
        while (i >= 0) : (i -= 1) {
            const idx: usize = @intCast(usize, i);
            if (self.d.release_queue.items[idx].frame_slot_to_be_released_in) |f| {
                if (f == self.d.current_frame_slot) {
                    const e = self.d.release_queue.orderedRemove(idx);
                    if (e.view_pool != null and e.view_start != null and e.view_count != null) {
                        e.view_pool.?.release(e.view_start.?, e.view_count.?);
                    }
                    switch (e.rtype) {
                        .Resource => {
                            self.d.resource_pool.remove(e.handle);
                        },
                        .Pipeline => {
                            self.d.pipeline_pool.remove(e.handle);
                        },
                        .Callback => {
                            e.callback_fn.?(e.callback_user_data.?);
                        }
                    }
                }
            }
        }
    }

    pub fn lookupOrCreatePipeline(self: *Fw,
                                  graphics_pso_desc: ?*d3d12.GRAPHICS_PIPELINE_STATE_DESC,
                                  compute_pso_desc: ?*d3d12.COMPUTE_PIPELINE_STATE_DESC,
                                  rs_desc: *const d3d12.VERSIONED_ROOT_SIGNATURE_DESC) !ObjectHandle {
        var sha: [Pipeline.sha_length]u8 = undefined;
        if (graphics_pso_desc != null) {
            Pipeline.getGraphicsPipelineSha(graphics_pso_desc.?, &sha);
        } else if (compute_pso_desc != null) {
            Pipeline.getComputePipelineSha(compute_pso_desc.?, &sha);
        } else {
            return ObjectHandle.invalid();
        }
        if (self.d.pipeline_cache.get(&sha)) |pipeline_handle| {
            return pipeline_handle;
        }
        var signature: *d3d.IBlob = undefined;
        try zwin32.hrErrorOnFail(d3d12.D3D12SerializeVersionedRootSignature(rs_desc,
                                                                            @ptrCast(*?*d3d.IBlob, &signature),
                                                                            null));
        defer _ = signature.Release();

        const pipeline_handle = blk: {
            var rs: *d3d12.IRootSignature = undefined;
            try zwin32.hrErrorOnFail(self.d.device.CreateRootSignature(0,
                                                                       signature.GetBufferPointer(),
                                                                       signature.GetBufferSize(),
                                                                       &d3d12.IID_IRootSignature,
                                                                       @ptrCast(*?*anyopaque, &rs)));
            errdefer _ = rs.Release();

            var pso: *d3d12.IPipelineState = undefined;
            if (graphics_pso_desc != null) {
                graphics_pso_desc.?.pRootSignature = rs;
                try zwin32.hrErrorOnFail(self.d.device.CreateGraphicsPipelineState(graphics_pso_desc.?,
                                                                                   &d3d12.IID_IPipelineState,
                                                                                   @ptrCast(*?*anyopaque, &pso)));
            } else {
                compute_pso_desc.?.pRootSignature = rs;
                try zwin32.hrErrorOnFail(self.d.device.CreateComputePipelineState(compute_pso_desc.?,
                                                                                  &d3d12.IID_IPipelineState,
                                                                                  @ptrCast(*?*anyopaque, &pso)));
            }
            errdefer _ = pso.Release();

            break :blk try Pipeline.addToPool(&self.d.pipeline_pool, pso, rs,
                                              if (graphics_pso_desc != null) Pipeline.Type.Graphics else Pipeline.Type.Compute);
        };

        try self.d.pipeline_cache.add(&sha, pipeline_handle);
        return pipeline_handle;
    }

    pub fn createBuffer(self: *Fw, heap_type: d3d12.HEAP_TYPE, size: u32) !ObjectHandle {
        var heap_properties = std.mem.zeroes(d3d12.HEAP_PROPERTIES);
        heap_properties.Type = heap_type;
        var resource: *d3d12.IResource = undefined;
        try zwin32.hrErrorOnFail(self.d.device.CreateCommittedResource(
            &heap_properties,
            d3d12.HEAP_FLAG_NONE,
            &.{
                .Dimension = .BUFFER,
                .Alignment = 0,
                .Width = std.math.max(1, size),
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = .UNKNOWN,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .ROW_MAJOR,
                .Flags = d3d12.RESOURCE_FLAG_NONE
            },
            d3d12.RESOURCE_STATE_COMMON,
            null,
            &d3d12.IID_IResource,
            @ptrCast(*?*anyopaque, &resource)));
        errdefer _ = resource.Release();
        return try Resource.addToPool(&self.d.resource_pool, resource, d3d12.RESOURCE_STATE_COMMON);
    }

    pub fn mapBuffer(self: *const Fw, comptime T: type, resource_handle: ObjectHandle) ![]T {
        const res = self.d.resource_pool.lookupRef(resource_handle) orelse return error.ResourceNotFoundInPool;
        var p: [*]u8 = undefined;
        try zwin32.hrErrorOnFail(res.resource.Map(0,  &.{ .Begin = 0, .End = 0 }, @ptrCast(*?*anyopaque, &p)));
        const slice = p[0..res.desc.Width];
        return std.mem.bytesAsSlice(T, @alignCast(@alignOf(T), slice));
    }

    pub fn createMappedHostVisibleBuffer(self: *Fw, size: u32) !HostVisibleBuffer {
        const resource_handle = try self.createBuffer(.UPLOAD, size);
        errdefer self.d.resource_pool.remove(resource_handle);
        const p = try self.mapBuffer(u8, resource_handle);
        return HostVisibleBuffer {
            .resource_handle = resource_handle,
            .p = p
        };
    }

    pub fn uploadBuffer(self: Fw, comptime T: type, resource_handle: ObjectHandle, data: []const T, staging: *StagingArea) !void {
        const res = self.d.resource_pool.lookupRef(resource_handle) orelse return;
        const byte_size = data.len * @sizeOf(T);
        const alloc = try staging.get(@intCast(u32, byte_size));
        std.mem.copy(T, alloc.castCpuSlice(T), data);
        self.d.cmd_list.CopyBufferRegion(res.resource, 0, alloc.buffer, alloc.buffer_offset, byte_size);
    }

    pub fn createTexture2D(self: *Fw,
                           format: dxgi.FORMAT,
                           pixel_size: Size,
                           mip_levels: u32,
                           flags: d3d12.RESOURCE_FLAGS,
                           state: d3d12.RESOURCE_STATES,
                           clear_value: ?*const d3d12.CLEAR_VALUE) !ObjectHandle {
        var heap_properties = std.mem.zeroes(d3d12.HEAP_PROPERTIES);
        heap_properties.Type = .DEFAULT;
        var resource: *d3d12.IResource = undefined;
        var resource_flags = flags;
        if (mip_levels > 1) {
            resource_flags |= d3d12.RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        try zwin32.hrErrorOnFail(self.d.device.CreateCommittedResource(
            &heap_properties,
            d3d12.HEAP_FLAG_NONE,
            &.{
                .Dimension = .TEXTURE2D,
                .Alignment = 0,
                .Width = pixel_size.width,
                .Height = pixel_size.height,
                .DepthOrArraySize = 1,
                .MipLevels = @intCast(u16, mip_levels),
                .Format = format,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .UNKNOWN,
                .Flags = resource_flags
            },
            state,
            clear_value,
            &d3d12.IID_IResource,
            @ptrCast(*?*anyopaque, &resource)));
        errdefer _ = resource.Release();
        return try Resource.addToPool(&self.d.resource_pool, resource, state);
    }

    pub fn createTexture2DSimple(self: *Fw,
                                 format: dxgi.FORMAT,
                                 pixel_size: Size,
                                 mip_levels: u32) !ObjectHandle {
        return self.createTexture2D(format, pixel_size, mip_levels, d3d12.RESOURCE_FLAG_NONE, d3d12.RESOURCE_STATE_COMMON, null);
    }

    pub fn uploadTexture2DSimple(self: *Fw,
                                 texture: ObjectHandle,
                                 data: []const u8,
                                 data_bytes_per_pixel: u32,
                                 data_bytes_per_line: u32,
                                 staging: *StagingArea) !void {
        const tex = self.d.resource_pool.lookupRef(texture) orelse return;
        var layout: [1]d3d12.PLACED_SUBRESOURCE_FOOTPRINT = undefined;
        var required_image_data_size: u64 = undefined;
        self.d.device.GetCopyableFootprints(&tex.desc, 0, 1, 0, &layout, null, null, &required_image_data_size);
        const required_bytes_per_line = layout[0].Footprint.RowPitch; // multiple of 256
        std.debug.assert(data_bytes_per_line <= required_bytes_per_line);
        const alloc = try staging.get(@intCast(u32, required_image_data_size));
        var y: u32 = 0;
        while (y < tex.desc.Height) : (y += 1) {
            const src_begin = y * data_bytes_per_line;
            const src_end = src_begin + (if (y < tex.desc.Height - 1) data_bytes_per_line else tex.desc.Width * data_bytes_per_pixel);
            const dst_begin = y * required_bytes_per_line;
            std.mem.copy(u8, alloc.cpu_slice[dst_begin..], data[src_begin..src_end]);
        }
        self.d.cmd_list.CopyTextureRegion(
            &.{
                .pResource = tex.resource,
                .Type = .SUBRESOURCE_INDEX,
                .u = .{
                    .SubresourceIndex = 0
                }
            },
            0, 0, 0,
            &.{
                .pResource = alloc.buffer,
                .Type = .PLACED_FOOTPRINT,
                .u = .{
                    .PlacedFootprint = .{
                        .Offset = alloc.buffer_offset,
                        .Footprint = .{
                            .Format = tex.desc.Format,
                            .Width = @intCast(u32, tex.desc.Width),
                            .Height = @intCast(u32, tex.desc.Height),
                            .Depth = 1,
                            .RowPitch = required_bytes_per_line
                        }
                    }
                }
            },
            null);
    }

    pub fn createSrv2D(self: *Fw, dst: *Descriptor, texture: ObjectHandle, most_detailed_mip: u32, mip_levels: u32) void {
        const tex = self.d.resource_pool.lookupRef(texture) orelse return;
        self.d.device.CreateShaderResourceView(
            tex.resource,
            &.{
                .Format = tex.desc.Format,
                .ViewDimension = .TEXTURE2D,
                .Shader4ComponentMapping = d3d12.DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .u = .{
                    .Texture2D = .{
                        .MostDetailedMip = most_detailed_mip,
                        .MipLevels = mip_levels,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0
                    }
                }
            },
            dst.cpu_handle);
    }

    pub fn createSrv2DNoMips(self: *Fw, dst: *Descriptor, texture: ObjectHandle) void {
        return self.createSrv2D(dst, texture, 0, 1);
    }

    pub fn createSrv2DAllMips(self: *Fw, dst: *Descriptor, texture: ObjectHandle) void {
        const tex = self.d.resource_pool.lookupRef(texture) orelse return;
        return self.createSrv2D(dst, texture, 0, tex.desc.MipLevels);
    }

    pub fn generateTexture2DMipmaps(self: *Fw, texture: ObjectHandle) !void {
        const texture_res = self.d.resource_pool.lookupRef(texture) orelse return;
        const texture_initial_state = texture_res.state;

        if (!self.d.pipeline_pool.isValid(self.d.mipmapgen_pipeline)) {
            var sampler_desc = std.mem.zeroes(d3d12.STATIC_SAMPLER_DESC);
            sampler_desc.Filter = .MIN_MAG_MIP_LINEAR;
            sampler_desc.AddressU = .CLAMP;
            sampler_desc.AddressV = .CLAMP;
            sampler_desc.AddressW = .CLAMP;
            sampler_desc.MaxLOD = std.math.floatMax(f32);
            sampler_desc.ShaderRegister = 0; // s0
            sampler_desc.ShaderVisibility = .ALL;
            var pso_desc = d3d12.COMPUTE_PIPELINE_STATE_DESC {
                .pRootSignature = null,
                .CS = .{
                    .pShaderBytecode = mipmapgen_cs,
                    .BytecodeLength = mipmapgen_cs.len,
                },
                .NodeMask = 0,
                .CachedPSO = .{
                    .pCachedBlob = null,
                    .CachedBlobSizeInBytes = 0
                },
                .Flags = d3d12.PIPELINE_STATE_FLAG_NONE
            };
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
                                .ShaderVisibility = .ALL
                            },
                            .{
                                .ParameterType = .DESCRIPTOR_TABLE,
                                .u = .{
                                    .DescriptorTable = .{
                                        .NumDescriptorRanges = 1,
                                        .pDescriptorRanges = &[_]d3d12.DESCRIPTOR_RANGE1 {
                                            .{
                                                .RangeType = .UAV,
                                                .NumDescriptors = 4,
                                                .BaseShaderRegister = 0, // u0..3
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
                        .NumStaticSamplers = 1,
                        .pStaticSamplers = &[_]d3d12.STATIC_SAMPLER_DESC {
                            sampler_desc
                        },
                        .Flags = d3d12.ROOT_SIGNATURE_FLAG_NONE
                    }
                }
            };
            self.d.mipmapgen_pipeline = try self.lookupOrCreatePipeline(null, &pso_desc, &rs_desc);
        }

        self.setPipeline(self.d.mipmapgen_pipeline);

        self.recordUavBarrier(texture);
        self.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_UNORDERED_ACCESS);
        self.recordTransitionBarriers();

        const descriptor_byte_size = self.getCurrentShaderVisibleCbvSrvUavHeapRange().descriptor_byte_size;
        var level: u32 = 0;
        while (level < texture_res.desc.MipLevels) {
            self.recordSubresourceTransitionBarrier(texture,
                                                    level,
                                                    d3d12.RESOURCE_STATE_UNORDERED_ACCESS,
                                                    d3d12.RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            var level_plus_one_mip_width = @intCast(u32, texture_res.desc.Width) >> @intCast(u5, level + 1);
            var level_plus_one_mip_height = texture_res.desc.Height >> @intCast(u5, level + 1);
            const dw = if (level_plus_one_mip_width == 1) level_plus_one_mip_height else level_plus_one_mip_width;
            const dh = if (level_plus_one_mip_height == 1) level_plus_one_mip_width else level_plus_one_mip_height;
            // number of times the size can be halved while still resulting in an even dimension
            const additional_mips = @ctz(dw | dh);
            const num_mips: u32 = std.math.min(1 + std.math.min(3, additional_mips), texture_res.desc.MipLevels - level);
            level_plus_one_mip_width = std.math.max(1, level_plus_one_mip_width);
            level_plus_one_mip_height = std.math.max(1, level_plus_one_mip_height);

            // std.debug.print("level={} num_mips={} level_{}_size={}x{}\n",
            //                 .{ level, num_mips, level + 1, level_plus_one_mip_width, level_plus_one_mip_height });

            const CBufData = struct {
                src_mip_level: u32,
                num_mip_levels: u32,
                texel_width: f32,
                texel_height: f32
            };
            const cbuf_data = [_]CBufData {
                .{
                    .src_mip_level = level,
                    .num_mip_levels = num_mips,
                    .texel_width = 1.0 / @intToFloat(f32, level_plus_one_mip_width),
                    .texel_height = 1.0 / @intToFloat(f32, level_plus_one_mip_height)
                }
            };
            const cbuf = try self.getCurrentStagingArea().get(@sizeOf(CBufData));
            std.mem.copy(CBufData, cbuf.castCpuSlice(CBufData), &cbuf_data);
            self.d.cmd_list.SetComputeRootConstantBufferView(0, cbuf.gpu_addr);

            var srv = try self.getCurrentShaderVisibleCbvSrvUavHeapRange().get(1);
            self.createSrv2D(&srv, texture, level, 1);
            self.d.cmd_list.SetComputeRootDescriptorTable(1, srv.gpu_handle);

            const uav_table_start = try self.getCurrentShaderVisibleCbvSrvUavHeapRange().get(4);
            var uav_cpu_handle = uav_table_start.cpu_handle;
            // if level is N, then need UAVs for levels N+1, ..., N+4
            var uav_idx: u32 = 0;
            while (uav_idx < 4) : (uav_idx += 1) {
                const uav_mip_level = std.math.min(level + 1 + uav_idx, texture_res.desc.MipLevels - 1);
                self.d.device.CreateUnorderedAccessView(
                    texture_res.resource,
                    null,
                    &.{
                        .Format = texture_res.desc.Format,
                        .ViewDimension = .TEXTURE2D,
                        .u = .{
                            .Texture2D = .{
                                .MipSlice = uav_mip_level,
                                .PlaneSlice = 0
                            }
                        }
                    },
                    uav_cpu_handle);
                uav_cpu_handle.ptr += descriptor_byte_size;
                //std.debug.print("  {}\n", .{ uav_mip_level });
            }
            self.d.cmd_list.SetComputeRootDescriptorTable(2, uav_table_start.gpu_handle);

            self.d.cmd_list.Dispatch(level_plus_one_mip_width, level_plus_one_mip_height, 1);

            self.recordUavBarrier(texture);
            self.recordSubresourceTransitionBarrier(texture,
                                                    level,
                                                    d3d12.RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                    d3d12.RESOURCE_STATE_UNORDERED_ACCESS);

            level += num_mips;
        }

        self.addTransitionBarrier(texture, texture_initial_state);
        self.recordTransitionBarriers();
    }

    pub fn beginGui(self: *Fw, cpu_cbv_srv_uav_pool: *CpuDescriptorPool) !void {
        if (!self.d.resource_pool.isValid(self.d.imgui_font_texture)) {
            var io = imgui.igGetIO().?;

            self.d.imgui_font_data = try self.allocator.alloc(u8, imgui_font.len);
            std.mem.copy(u8, self.d.imgui_font_data.?, imgui_font);
            var font_conf = imgui.ImFontConfig_ImFontConfig();
            font_conf.*.FontDataOwnedByAtlas = false;
            _ = imgui.ImFontAtlas_AddFontFromMemoryTTF(io.*.Fonts,
                                                       @ptrCast(?*anyopaque, self.d.imgui_font_data.?),
                                                       imgui_font.len,
                                                       20.0,
                                                       font_conf,
                                                       null);
            imgui.ImFontConfig_destroy(font_conf);

            var p: [*c]u8 = undefined;
            var w: i32 = 0;
            var h: i32 = 0;
            imgui.ImFontAtlas_GetTexDataAsRGBA32(io.*.Fonts, &p, &w, &h, null);

            var id: ?*u8 = null;
            imgui.ImFontAtlas_SetTexID(io.*.Fonts, @ptrCast(?*anyopaque, id));

            const texture = try self.createTexture2DSimple(.R8G8B8A8_UNORM,
                                                           .{ .width = @intCast(u32, w), .height = @intCast(u32, h) },
                                                           1);
            var srv = try cpu_cbv_srv_uav_pool.allocate(1);
            self.createSrv2DNoMips(&srv, texture);
            const pixels = p[0..@intCast(usize, w * h * 4)];
            self.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_COPY_DEST);
            self.recordTransitionBarriers();
            try self.uploadTexture2DSimple(texture, pixels, 4, @intCast(u32, w * 4), self.getCurrentStagingArea());
            self.addTransitionBarrier(texture, d3d12.RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            self.recordTransitionBarriers();

            self.d.imgui_font_texture = texture;
            self.d.imgui_font_srv = srv;
        }

        if (!self.d.pipeline_pool.isValid(self.d.imgui_pipeline)) {
            const input_element_descs = [_]d3d12.INPUT_ELEMENT_DESC {
                d3d12.INPUT_ELEMENT_DESC {
                    .SemanticName = "POSITION",
                    .SemanticIndex = 0,
                    .Format = .R32G32_FLOAT,
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
                    .AlignedByteOffset = 2 * @sizeOf(f32),
                    .InputSlotClass = .PER_VERTEX_DATA,
                    .InstanceDataStepRate = 0
                },
                d3d12.INPUT_ELEMENT_DESC {
                    .SemanticName = "TEXCOORD",
                    .SemanticIndex = 1,
                    .Format = .R8G8B8A8_UNORM,
                    .InputSlot = 0,
                    .AlignedByteOffset = 4 * @sizeOf(f32),
                    .InputSlotClass = .PER_VERTEX_DATA,
                    .InstanceDataStepRate = 0
                },
            };
            var pso_desc = std.mem.zeroes(d3d12.GRAPHICS_PIPELINE_STATE_DESC);
            pso_desc.InputLayout = .{
                .pInputElementDescs = &input_element_descs,
                .NumElements = input_element_descs.len
            };
            pso_desc.VS = .{
                .pShaderBytecode = imgui_vs,
                .BytecodeLength = imgui_vs.len
            };
            pso_desc.PS = .{
                .pShaderBytecode = imgui_ps,
                .BytecodeLength = imgui_ps.len
            };
            pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
            pso_desc.BlendState.RenderTarget[0].BlendEnable = w32.TRUE;
            pso_desc.BlendState.RenderTarget[0].SrcBlend = .SRC_ALPHA;
            pso_desc.BlendState.RenderTarget[0].DestBlend = .INV_SRC_ALPHA;
            pso_desc.BlendState.RenderTarget[0].BlendOp = .ADD;
            pso_desc.BlendState.RenderTarget[0].SrcBlendAlpha = .INV_SRC_ALPHA;
            pso_desc.BlendState.RenderTarget[0].DestBlendAlpha = .ZERO;
            pso_desc.BlendState.RenderTarget[0].BlendOpAlpha = .ADD;
            pso_desc.SampleMask = 0xFFFFFFFF;
            pso_desc.RasterizerState.FillMode = .SOLID;
            pso_desc.RasterizerState.CullMode = .NONE;
            pso_desc.PrimitiveTopologyType = .TRIANGLE;
            pso_desc.NumRenderTargets = 1;
            pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
            pso_desc.SampleDesc = .{ .Count = 1, .Quality = 0 };

            var sampler_desc = std.mem.zeroes(d3d12.STATIC_SAMPLER_DESC);
            sampler_desc.Filter = .MIN_MAG_MIP_LINEAR;
            sampler_desc.AddressU = .CLAMP;
            sampler_desc.AddressV = .CLAMP;
            sampler_desc.AddressW = .CLAMP;
            sampler_desc.ShaderRegister = 0; // s0
            sampler_desc.ShaderVisibility = .PIXEL;

            const rs_desc = d3d12.VERSIONED_ROOT_SIGNATURE_DESC {
                .Version = d3d12.ROOT_SIGNATURE_VERSION.VERSION_1_1,
                .u = .{
                    .Desc_1_1 = .{
                        .NumParameters = 2,
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
                                .ShaderVisibility = .VERTEX
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
                            }
                        },
                        .NumStaticSamplers = 1,
                        .pStaticSamplers = &[_]d3d12.STATIC_SAMPLER_DESC {
                            sampler_desc
                        },
                        .Flags = d3d12.ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                    }
                }
            };

            self.d.imgui_pipeline = try self.lookupOrCreatePipeline(&pso_desc, null, &rs_desc);
        }

        var io = imgui.igGetIO().?;
        io.*.DisplaySize = imgui.ImVec2 {
            .x = @intToFloat(f32, self.d.swapchain_size.width),
            .y = @intToFloat(f32, self.d.swapchain_size.height)
        };
        imgui.igNewFrame();
    }

    pub fn endGui(self: *Fw) !void {
        imgui.igRender();
        const draw_data = imgui.igGetDrawData();
        if (draw_data == null or draw_data.?.*.TotalVtxCount == 0) {
            return;
        }

        const num_vertices = @intCast(u32, draw_data.?.*.TotalVtxCount);
        const vbuf_byte_size = num_vertices * @sizeOf(imgui.ImDrawVert);
        var imgui_vbuf = &self.d.imgui_vbuf[self.d.current_frame_slot];
        const num_indices = @intCast(u32, draw_data.?.*.TotalIdxCount);
        const ibuf_byte_size = num_indices * @sizeOf(imgui.ImDrawIdx);
        var imgui_ibuf = &self.d.imgui_ibuf[self.d.current_frame_slot];
        if (self.d.resource_pool.lookupRef(imgui_vbuf.resource_handle)) |vbuf| {
            if (vbuf.desc.Width < vbuf_byte_size) {
                self.d.resource_pool.remove(imgui_vbuf.resource_handle);
            }
        }
        if (self.d.resource_pool.lookupRef(imgui_ibuf.resource_handle)) |ibuf| {
            if (ibuf.desc.Width < ibuf_byte_size) {
                self.d.resource_pool.remove(imgui_ibuf.resource_handle);
            }
        }
        if (!self.d.resource_pool.isValid(imgui_vbuf.resource_handle)) {
            const extra = 200 * @sizeOf(imgui.ImDrawVert);
            imgui_vbuf.* = try self.createMappedHostVisibleBuffer(vbuf_byte_size + extra);
        }
        if (!self.d.resource_pool.isValid(imgui_ibuf.resource_handle)) {
            const extra = 200 * @sizeOf(imgui.ImDrawIdx);
            imgui_ibuf.* = try self.createMappedHostVisibleBuffer(ibuf_byte_size + extra);
        }

        var vdata = imgui_vbuf.ptrAs(imgui.ImDrawVert);
        var idata = imgui_ibuf.ptrAs(imgui.ImDrawIdx);
        var voffset: u32 = 0;
        var ioffset: u32 = 0;
        var i: u32 = 0;
        while (i < @intCast(u32, draw_data.?.*.CmdListsCount)) : (i += 1) {
            const list = draw_data.?.*.CmdLists[i];
            const vcount = @intCast(u32, list.*.VtxBuffer.Size);
            std.mem.copy(imgui.ImDrawVert, vdata[voffset..voffset + vcount], list.*.VtxBuffer.Data[0..vcount]);
            const icount = @intCast(u32, list.*.IdxBuffer.Size);
            std.mem.copy(imgui.ImDrawIdx, idata[ioffset..ioffset + icount], list.*.IdxBuffer.Data[0..icount]);
            voffset += vcount;
            ioffset += icount;
        }

        self.setPipeline(self.d.imgui_pipeline);
        self.d.cmd_list.RSSetViewports(1, &[_]d3d12.VIEWPORT {
            .{
                .TopLeftX = 0.0,
                .TopLeftY = 0.0,
                .Width = @intToFloat(f32, self.d.swapchain_size.width),
                .Height = @intToFloat(f32, self.d.swapchain_size.height),
                .MinDepth = 0.0,
                .MaxDepth = 1.0
            }
        });
        self.d.cmd_list.IASetPrimitiveTopology(.TRIANGLELIST);
        self.d.cmd_list.IASetVertexBuffers(0, 1, &[_]d3d12.VERTEX_BUFFER_VIEW {
            .{
                .BufferLocation = self.d.resource_pool.lookupRef(imgui_vbuf.resource_handle).?.resource.GetGPUVirtualAddress(),
                .SizeInBytes = num_vertices * @sizeOf(imgui.ImDrawVert),
                .StrideInBytes = @sizeOf(imgui.ImDrawVert)
            }
        });
        self.d.cmd_list.IASetIndexBuffer(&.{
            .BufferLocation = self.d.resource_pool.lookupRef(imgui_ibuf.resource_handle).?.resource.GetGPUVirtualAddress(),
            .SizeInBytes = num_indices * @sizeOf(imgui.ImDrawIdx),
            .Format = if (@sizeOf(imgui.ImDrawIdx) == 2) .R16_UINT else .R32_UINT
        });

        const cbuf = try self.getCurrentStagingArea().get(64);
        const m = [_]f32 { // column major
            2.0 / @intToFloat(f32, self.d.swapchain_size.width), 0.0, 0.0, -1.0,
            0.0, 2.0 / -@intToFloat(f32, self.d.swapchain_size.height), 0.0, 1.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0
        };
        std.mem.copy(f32, cbuf.castCpuSlice(f32), &m);
        self.d.cmd_list.SetGraphicsRootConstantBufferView(0, cbuf.gpu_addr);
        const shader_visible_srv = try self.getCurrentShaderVisibleCbvSrvUavHeapRange().get(1);
        self.d.device.CopyDescriptorsSimple(1,
                                            shader_visible_srv.cpu_handle,
                                            self.d.imgui_font_srv.cpu_handle,
                                            .CBV_SRV_UAV);
        self.d.cmd_list.SetGraphicsRootDescriptorTable(1, shader_visible_srv.gpu_handle);

        voffset = 0;
        ioffset = 0;
        i = 0;
        while (i < @intCast(u32, draw_data.?.*.CmdListsCount)) : (i += 1) {
            const gui_cmd_list = draw_data.?.*.CmdLists[i];
            var j: u32 = 0;
            while (j < gui_cmd_list.*.CmdBuffer.Size) : (j += 1) {
                const cmd = &gui_cmd_list.*.CmdBuffer.Data[j];
                if (cmd.*.UserCallback == null) {
                    const scissor_rect = d3d12.RECT {
                        .left = @floatToInt(i32, cmd.*.ClipRect.x),
                        .top = @floatToInt(i32, cmd.*.ClipRect.y),
                        .right = @floatToInt(i32, cmd.*.ClipRect.z),
                        .bottom = @floatToInt(i32, cmd.*.ClipRect.w)
                    };
                    if (scissor_rect.right > scissor_rect.left and scissor_rect.bottom > scissor_rect.top) {
                        self.d.cmd_list.RSSetScissorRects(1, &[_]d3d12.RECT { scissor_rect });
                        self.d.cmd_list.DrawIndexedInstanced(cmd.*.ElemCount, 1, cmd.*.IdxOffset + ioffset,
                                                             @intCast(i32, cmd.*.VtxOffset + voffset), 0);
                    }
                }
            }
            voffset += @intCast(u32, gui_cmd_list.*.VtxBuffer.Size);
            ioffset += @intCast(u32, gui_cmd_list.*.IdxBuffer.Size);
        }
    }

    pub fn lookupOrCreateSampler(self: *Fw, key: SamplerCacheKey) !Descriptor {
        var v = try self.d.sampler_cache_map.getOrPut(key);
        if (v.found_existing) {
            return v.value_ptr.*;
        }
        const sampler_alloc = if (key.stype == .Cpu) try self.d.sampler_cpu_pool.allocate(1)
            else try self.getPermanentShaderVisibleSamplerHeapRange().get(1);
        self.d.device.CreateSampler(&key.desc, sampler_alloc.cpu_handle);
        v.key_ptr.* = key;
        v.value_ptr.* = sampler_alloc;
        return sampler_alloc;
    }

    pub fn getDummyTexture(self: *Fw, cpu_cbv_srv_uav_pool: *CpuDescriptorPool) !Descriptor {
        if (!self.d.resource_pool.isValid(self.d.dummy_texture_handle) or self.d.dummy_texture == null) {
            self.d.dummy_texture_handle = try self.createTexture2DSimple(.R8G8B8A8_UNORM, .{ .width = 64, .height = 64 }, 1);
            self.addTransitionBarrier(self.d.dummy_texture_handle, d3d12.RESOURCE_STATE_COPY_DEST);
            self.recordTransitionBarriers();
            const dummy_image = [_]u8 { 255 } ** (64 * 64 * 4);
            try self.uploadTexture2DSimple(self.d.dummy_texture_handle, &dummy_image, 4, 64 * 4, self.getCurrentStagingArea());
            self.addTransitionBarrier(self.d.dummy_texture_handle, d3d12.RESOURCE_STATE_ALL_SHADER_RESOURCE);
            self.recordTransitionBarriers();
            self.d.dummy_texture = try cpu_cbv_srv_uav_pool.allocate(1);
            self.createSrv2DNoMips(&self.d.dummy_texture.?, self.d.dummy_texture_handle);
        }
        return self.d.dummy_texture.?;
    }

    fn isVkKeyDown(vk: c_int) bool {
        return (@bitCast(u16, w32.GetKeyState(vk)) & 0x8000) != 0;
    }

    fn vkKeyToImGuiKey(wparam: w32.WPARAM) imgui.ImGuiKey {
        switch (wparam) {
            w32.VK_TAB => return imgui.ImGuiKey_Tab,
            w32.VK_LEFT => return imgui.ImGuiKey_LeftArrow,
            w32.VK_RIGHT => return imgui.ImGuiKey_RightArrow,
            w32.VK_UP => return imgui.ImGuiKey_UpArrow,
            w32.VK_DOWN => return imgui.ImGuiKey_DownArrow,
            w32.VK_PRIOR => return imgui.ImGuiKey_PageUp,
            w32.VK_NEXT => return imgui.ImGuiKey_PageDown,
            w32.VK_HOME => return imgui.ImGuiKey_Home,
            w32.VK_END => return imgui.ImGuiKey_End,
            w32.VK_INSERT => return imgui.ImGuiKey_Insert,
            w32.VK_DELETE => return imgui.ImGuiKey_Delete,
            w32.VK_BACK => return imgui.ImGuiKey_Backspace,
            w32.VK_SPACE => return imgui.ImGuiKey_Space,
            w32.VK_RETURN => return imgui.ImGuiKey_Enter,
            w32.VK_ESCAPE => return imgui.ImGuiKey_Escape,
            w32.VK_OEM_7 => return imgui.ImGuiKey_Apostrophe,
            w32.VK_OEM_COMMA => return imgui.ImGuiKey_Comma,
            w32.VK_OEM_MINUS => return imgui.ImGuiKey_Minus,
            w32.VK_OEM_PERIOD => return imgui.ImGuiKey_Period,
            w32.VK_OEM_2 => return imgui.ImGuiKey_Slash,
            w32.VK_OEM_1 => return imgui.ImGuiKey_Semicolon,
            w32.VK_OEM_PLUS => return imgui.ImGuiKey_Equal,
            w32.VK_OEM_4 => return imgui.ImGuiKey_LeftBracket,
            w32.VK_OEM_5 => return imgui.ImGuiKey_Backslash,
            w32.VK_OEM_6 => return imgui.ImGuiKey_RightBracket,
            w32.VK_OEM_3 => return imgui.ImGuiKey_GraveAccent,
            w32.VK_CAPITAL => return imgui.ImGuiKey_CapsLock,
            w32.VK_SCROLL => return imgui.ImGuiKey_ScrollLock,
            w32.VK_NUMLOCK => return imgui.ImGuiKey_NumLock,
            w32.VK_SNAPSHOT => return imgui.ImGuiKey_PrintScreen,
            w32.VK_PAUSE => return imgui.ImGuiKey_Pause,
            w32.VK_NUMPAD0 => return imgui.ImGuiKey_Keypad0,
            w32.VK_NUMPAD1 => return imgui.ImGuiKey_Keypad1,
            w32.VK_NUMPAD2 => return imgui.ImGuiKey_Keypad2,
            w32.VK_NUMPAD3 => return imgui.ImGuiKey_Keypad3,
            w32.VK_NUMPAD4 => return imgui.ImGuiKey_Keypad4,
            w32.VK_NUMPAD5 => return imgui.ImGuiKey_Keypad5,
            w32.VK_NUMPAD6 => return imgui.ImGuiKey_Keypad6,
            w32.VK_NUMPAD7 => return imgui.ImGuiKey_Keypad7,
            w32.VK_NUMPAD8 => return imgui.ImGuiKey_Keypad8,
            w32.VK_NUMPAD9 => return imgui.ImGuiKey_Keypad9,
            w32.VK_DECIMAL => return imgui.ImGuiKey_KeypadDecimal,
            w32.VK_DIVIDE => return imgui.ImGuiKey_KeypadDivide,
            w32.VK_MULTIPLY => return imgui.ImGuiKey_KeypadMultiply,
            w32.VK_SUBTRACT => return imgui.ImGuiKey_KeypadSubtract,
            w32.VK_ADD => return imgui.ImGuiKey_KeypadAdd,
            w32.IM_VK_KEYPAD_ENTER => return imgui.ImGuiKey_KeypadEnter,
            w32.VK_LSHIFT => return imgui.ImGuiKey_LeftShift,
            w32.VK_LCONTROL => return imgui.ImGuiKey_LeftCtrl,
            w32.VK_LMENU => return imgui.ImGuiKey_LeftAlt,
            w32.VK_LWIN => return imgui.ImGuiKey_LeftSuper,
            w32.VK_RSHIFT => return imgui.ImGuiKey_RightShift,
            w32.VK_RCONTROL => return imgui.ImGuiKey_RightCtrl,
            w32.VK_RMENU => return imgui.ImGuiKey_RightAlt,
            w32.VK_RWIN => return imgui.ImGuiKey_RightSuper,
            w32.VK_APPS => return imgui.ImGuiKey_Menu,
            '0' => return imgui.ImGuiKey_0,
            '1' => return imgui.ImGuiKey_1,
            '2' => return imgui.ImGuiKey_2,
            '3' => return imgui.ImGuiKey_3,
            '4' => return imgui.ImGuiKey_4,
            '5' => return imgui.ImGuiKey_5,
            '6' => return imgui.ImGuiKey_6,
            '7' => return imgui.ImGuiKey_7,
            '8' => return imgui.ImGuiKey_8,
            '9' => return imgui.ImGuiKey_9,
            'A' => return imgui.ImGuiKey_A,
            'B' => return imgui.ImGuiKey_B,
            'C' => return imgui.ImGuiKey_C,
            'D' => return imgui.ImGuiKey_D,
            'E' => return imgui.ImGuiKey_E,
            'F' => return imgui.ImGuiKey_F,
            'G' => return imgui.ImGuiKey_G,
            'H' => return imgui.ImGuiKey_H,
            'I' => return imgui.ImGuiKey_I,
            'J' => return imgui.ImGuiKey_J,
            'K' => return imgui.ImGuiKey_K,
            'L' => return imgui.ImGuiKey_L,
            'M' => return imgui.ImGuiKey_M,
            'N' => return imgui.ImGuiKey_N,
            'O' => return imgui.ImGuiKey_O,
            'P' => return imgui.ImGuiKey_P,
            'Q' => return imgui.ImGuiKey_Q,
            'R' => return imgui.ImGuiKey_R,
            'S' => return imgui.ImGuiKey_S,
            'T' => return imgui.ImGuiKey_T,
            'U' => return imgui.ImGuiKey_U,
            'V' => return imgui.ImGuiKey_V,
            'W' => return imgui.ImGuiKey_W,
            'X' => return imgui.ImGuiKey_X,
            'Y' => return imgui.ImGuiKey_Y,
            'Z' => return imgui.ImGuiKey_Z,
            w32.VK_F1 => return imgui.ImGuiKey_F1,
            w32.VK_F2 => return imgui.ImGuiKey_F2,
            w32.VK_F3 => return imgui.ImGuiKey_F3,
            w32.VK_F4 => return imgui.ImGuiKey_F4,
            w32.VK_F5 => return imgui.ImGuiKey_F5,
            w32.VK_F6 => return imgui.ImGuiKey_F6,
            w32.VK_F7 => return imgui.ImGuiKey_F7,
            w32.VK_F8 => return imgui.ImGuiKey_F8,
            w32.VK_F9 => return imgui.ImGuiKey_F9,
            w32.VK_F10 => return imgui.ImGuiKey_F10,
            w32.VK_F11 => return imgui.ImGuiKey_F11,
            w32.VK_F12 => return imgui.ImGuiKey_F12,
            else => return imgui.ImGuiKey_None
        }
    }

    const PAINTSTRUCT = extern struct {
        hdc: w32.HDC,
        fErase: w32.BOOL,
        rcPaint: w32.RECT,
        fRestore: w32.BOOL,
        fIncUpdate: w32.BOOL,
        rgbReserved: [32]u8
    };

    extern "user32" fn BeginPaint(hWnd: w32.HWND, lpPaint: *PAINTSTRUCT) w32.HDC;
    extern "user32" fn EndPaint(hWnd: w32.HWND, lpPaint: *PAINTSTRUCT) w32.BOOL;
    //extern "gdi32" fn FillRect(hDC: w32.HDC, lprc: *w32.RECT, hbr: w32.HBRUSH) w32.INT;

    fn processWindowMessage(
        window: w32.HWND,
        message: w32.UINT,
        wparam: w32.WPARAM,
        lparam: w32.LPARAM,
    ) callconv(w32.WINAPI) w32.LRESULT {
        if (imgui.igGetCurrentContext() == null) {
            return w32.user32.defWindowProcA(window, message, wparam, lparam);
        }
        var io = imgui.igGetIO().?;
        var d = @ptrCast(*Data, @alignCast(@alignOf(*Data), io.*.BackendPlatformUserData));

        switch (message) {
            w32.user32.WM_DESTROY => {
                w32.user32.PostQuitMessage(0);
            },
            w32.user32.WM_SIZE => {
                const new_size = Size {
                    .width = @intCast(u32, lparam & 0xFFFF),
                    .height = @intCast(u32, lparam >> 16)
                };
                if (!std.meta.eql(d.window_size, new_size)) {
                    d.window_size = new_size;
                    //std.debug.print("new width {} height {}\n", .{d.window_size.width, d.window_size.height});
                }
            },
            w32.user32.WM_PAINT => {
                var ps = std.mem.zeroes(PAINTSTRUCT);
                _ = BeginPaint(window, &ps);
                //_ = FillRect(ps.hdc, &ps.rcPaint, @intToPtr(w32.HBRUSH, 5)); // COLOR_WINDOW
                _ = EndPaint(window, &ps);
            },
            w32.user32.WM_LBUTTONDOWN,
            w32.user32.WM_RBUTTONDOWN,
            w32.user32.WM_MBUTTONDOWN,
            w32.user32.WM_LBUTTONDBLCLK,
            w32.user32.WM_RBUTTONDBLCLK,
            w32.user32.WM_MBUTTONDBLCLK => {
                var button: u32 = 0;
                if (message == w32.user32.WM_LBUTTONDOWN or message == w32.user32.WM_LBUTTONDBLCLK) button = 0;
                if (message == w32.user32.WM_RBUTTONDOWN or message == w32.user32.WM_RBUTTONDBLCLK) button = 1;
                if (message == w32.user32.WM_MBUTTONDOWN or message == w32.user32.WM_MBUTTONDBLCLK) button = 2;
                if (d.imgui_wdata.mouse_buttons_down == 0 and w32.GetCapture() == null) {
                    _ = w32.SetCapture(window);
                }
                d.imgui_wdata.mouse_buttons_down |= @as(u32, 1) << @intCast(u5, button);
                imgui.ImGuiIO_AddMouseButtonEvent(io, @intCast(i32, button), true);
            },
            w32.user32.WM_LBUTTONUP,
            w32.user32.WM_RBUTTONUP,
            w32.user32.WM_MBUTTONUP => {
                var button: u32 = 0;
                if (message == w32.user32.WM_LBUTTONUP) button = 0;
                if (message == w32.user32.WM_RBUTTONUP) button = 1;
                if (message == w32.user32.WM_MBUTTONUP) button = 2;
                d.imgui_wdata.mouse_buttons_down &= ~(@as(u32, 1) << @intCast(u5, button));
                if (d.imgui_wdata.mouse_buttons_down == 0 and w32.GetCapture() == window) {
                    _ = w32.ReleaseCapture();
                }
                imgui.ImGuiIO_AddMouseButtonEvent(io, @intCast(i32, button), false);
            },
            w32.user32.WM_MOUSEWHEEL => {
                const wheel_y = @intToFloat(f32, w32.GET_WHEEL_DELTA_WPARAM(wparam)) / @intToFloat(f32, w32.WHEEL_DELTA);
                imgui.ImGuiIO_AddMouseWheelEvent(io, 0.0, wheel_y);
            },
            w32.user32.WM_MOUSEMOVE => {
                d.imgui_wdata.mouse_window = window;
                if (!d.imgui_wdata.mouse_tracked) {
                    _ = w32.TrackMouseEvent(&w32.TRACKMOUSEEVENT {
                        .cbSize = @sizeOf(w32.TRACKMOUSEEVENT),
                        .dwFlags = w32.TME_LEAVE,
                        .hwndTrack = window,
                        .dwHoverTime = 0
                    });
                    d.imgui_wdata.mouse_tracked = true;
                }
                imgui.ImGuiIO_AddMousePosEvent(io,
                                               @intToFloat(f32, w32.GET_X_LPARAM(lparam)),
                                               @intToFloat(f32, w32.GET_Y_LPARAM(lparam)));
            },
            w32.user32.WM_MOUSELEAVE => {
                if (d.imgui_wdata.mouse_window == window) {
                    d.imgui_wdata.mouse_window = null;
                }
                d.imgui_wdata.mouse_tracked = false;
                imgui.ImGuiIO_AddMousePosEvent(io, -imgui.igGET_FLT_MAX(), -imgui.igGET_FLT_MAX());
            },
            w32.user32.WM_SETFOCUS,
            w32.user32.WM_KILLFOCUS => {
                imgui.ImGuiIO_AddFocusEvent(io, if (message == w32.user32.WM_SETFOCUS) true else false);
            },
            w32.user32.WM_CHAR => {
                if (wparam > 0 and wparam < 0x10000) {
                    imgui.ImGuiIO_AddInputCharacterUTF16(io, @intCast(u16, wparam & 0xFFFF));
                }
            },
            w32.user32.WM_KEYDOWN,
            w32.user32.WM_KEYUP,
            w32.user32.WM_SYSKEYDOWN,
            w32.user32.WM_SYSKEYUP => {
                const down = if (message == w32.user32.WM_KEYDOWN or message == w32.user32.WM_SYSKEYDOWN) true else false;
                if (wparam < 256) {
                    imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiMod_Ctrl, isVkKeyDown(w32.VK_CONTROL));
                    imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiMod_Shift, isVkKeyDown(w32.VK_SHIFT));
                    imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiMod_Alt, isVkKeyDown(w32.VK_MENU));
                    imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiMod_Super, isVkKeyDown(w32.VK_APPS));

                    var vk = @intCast(i32, wparam);
                    if (wparam == w32.VK_RETURN and (((lparam >> 16) & 0xFFFF) & w32.KF_EXTENDED) != 0) {
                        vk = w32.IM_VK_KEYPAD_ENTER;
                    }
                    const key = vkKeyToImGuiKey(wparam);
                    if (key != imgui.ImGuiKey_None)
                        imgui.ImGuiIO_AddKeyEvent(io, key, down);

                    if (vk == w32.VK_SHIFT) {
                        if (isVkKeyDown(w32.VK_LSHIFT) == down)
                            imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiKey_LeftShift, down);
                        if (isVkKeyDown(w32.VK_RSHIFT) == down)
                            imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiKey_RightShift, down);
                    } else if (vk == w32.VK_CONTROL) {
                        if (isVkKeyDown(w32.VK_LCONTROL) == down)
                            imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiKey_LeftCtrl, down);
                        if (isVkKeyDown(w32.VK_RCONTROL) == down)
                            imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiKey_RightCtrl, down);
                    } else if (vk == w32.VK_MENU) {
                        if (isVkKeyDown(w32.VK_LMENU) == down)
                            imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiKey_LeftAlt, down);
                        if (isVkKeyDown(w32.VK_RMENU) == down)
                            imgui.ImGuiIO_AddKeyEvent(io, imgui.ImGuiKey_RightAlt, down);
                    }
                }
                return w32.user32.defWindowProcA(window, message, wparam, lparam);
            },
            else => {
                return w32.user32.defWindowProcA(window, message, wparam, lparam);
            }
        }
        return 0;
    }

    pub fn guiHasFocus() bool {
        return imgui.igIsWindowFocused(imgui.ImGuiFocusedFlags_AnyWindow);
    }

    pub fn updateCamera(self: *Fw, camera: *Camera, movement_speed: f32, rotate_speed: f32) void {
        if (guiHasFocus()) {
            _ = w32.GetCursorPos(&self.d.camera_wdata.last_cursor_pos);
        } else {
            var cursor_pos: w32.POINT = undefined;
            _ = w32.GetCursorPos(&cursor_pos);
            const dx = cursor_pos.x - self.d.camera_wdata.last_cursor_pos.x;
            const dy = cursor_pos.y - self.d.camera_wdata.last_cursor_pos.y;
            self.d.camera_wdata.last_cursor_pos = cursor_pos;
            if (w32.GetAsyncKeyState(w32.VK_LBUTTON) < 0) {
                camera.rotate(@intToFloat(f32, dx), @intToFloat(f32, dy), rotate_speed);
            }
            if (w32.GetAsyncKeyState('W') < 0) {
                camera.moveForward(movement_speed);
            } else if (w32.GetAsyncKeyState('S') < 0) {
                camera.moveBackward(movement_speed);
            }
            if (w32.GetAsyncKeyState('D') < 0) {
                camera.moveRight(movement_speed);
            } else if (w32.GetAsyncKeyState('A') < 0) {
                camera.moveLeft(movement_speed);
            }
            if (w32.GetAsyncKeyState('R') < 0) {
                camera.moveUp(movement_speed);
            } else if (w32.GetAsyncKeyState('F') < 0) {
                camera.moveDown(movement_speed);
            }
        }
    }

    pub fn handleWindowEvents() bool {
        var message = std.mem.zeroes(w32.user32.MSG);
        while (w32.user32.peekMessageA(&message, null, 0, 0, w32.user32.PM_REMOVE) catch false) {
            _ = w32.user32.translateMessage(&message);
            _ = w32.user32.dispatchMessageA(&message);
            if (message.message == w32.user32.WM_QUIT) {
                return false;
            }
        }
        return true;
    }

    pub fn formatTemp(self: *Fw, comptime fmt: []const u8, args: anytype) []const u8 {
        const len = std.fmt.count(fmt, args);
        if (self.d.format_work_area.items.len < len) {
            self.d.format_work_area.resize(len + 64) catch unreachable;
        }
        return std.fmt.bufPrint(self.d.format_work_area.items, fmt, args) catch unreachable;
    }

    pub fn formatTempZ(self: *Fw, comptime fmt: []const u8, args: anytype) [:0]const u8 {
        const len = std.fmt.count(fmt ++ "\x00", args);
        if (self.d.format_work_area.items.len < len) {
            self.d.format_work_area.resize(len + 64) catch unreachable;
        }
        return std.fmt.bufPrintZ(self.d.format_work_area.items, fmt, args) catch unreachable;
    }

    pub fn getMeshArenaAllocator(self: *const Fw) std.mem.Allocator {
        return self.d.mesh_arena.allocator();
    }

    pub fn resetMeshArena(self: *Fw) void {
        zmesh.deinit();
        self.d.mesh_arena.deinit();
        self.d.mesh_arena = std.heap.ArenaAllocator.init(self.allocator);
        zmesh.init(self.d.mesh_arena.allocator());
    }

    pub fn getImageArenaAllocator(self: *const Fw) std.mem.Allocator {
        return self.d.image_arena.allocator();
    }

    pub fn resetImageArena(self: *Fw) void {
        zstbi.deinit();
        self.d.image_arena.deinit();
        self.d.image_arena = std.heap.ArenaAllocator.init(self.allocator);
        zstbi.init(self.d.image_arena.allocator());
    }

    const ImageLoadContext = struct {
        source: [:0]u8,
        result: *?zstbi.Image
    };

    fn imageLoaderFunc(ctx: ImageLoadContext) void {
        ctx.result.* = zstbi.Image.init(ctx.source, 4) catch null; // image arena
    }

    /// Allocates all work data + vertex and index data in the Mesh using the
    /// mesh arena. MeshTexture.image is on the image arena. The other Mesh
    /// data is allocated normally. Missing vertex attributes are filled out
    /// with zeroes.
    pub fn loadGltf(self: *Fw, gltf_path: [:0]const u8) !Mesh {
        const options = zmesh.gltf.Options {
            .memory = .{
                // mesh arena
                .alloc_func = zmesh.mem.zmeshAllocUser,
                .free_func = zmesh.mem.zmeshFreeUser
            }
        };
        const gltf = try zmesh.gltf.parseFile(options, gltf_path);
        try zmesh.gltf.loadBuffers(options, gltf, gltf_path);

        var arena = self.getMeshArenaAllocator();
        var mesh = Mesh.init(self.allocator, arena);
        errdefer mesh.deinit();
        mesh.source = try std.fmt.allocPrintZ(self.allocator, "{s}", .{ gltf_path });
        var indices = std.ArrayList(u32).init(arena);
        var positions = std.ArrayList([3]f32).init(arena);
        var normals = std.ArrayList([3]f32).init(arena);
        var texcoords0 = std.ArrayList([2]f32).init(arena);
        var tangents = std.ArrayList([4]f32).init(arena);

        const num_meshes = @intCast(u32, gltf.meshes_count);
        const num_materials = @intCast(u32, gltf.materials_count);
        var mesh_index: u32 = 0;
        while (mesh_index < num_meshes) : (mesh_index += 1) {
            const current_mesh = &gltf.meshes.?[mesh_index];
            const num_submeshes = @intCast(u32, gltf.meshes.?[mesh_index].primitives_count);
            var submesh_index: u32 = 0;
            while (submesh_index < num_submeshes) : (submesh_index += 1) {
                const indices_start_index = indices.items.len;
                const positions_start_index = positions.items.len;
                const current_submesh = &current_mesh.primitives[submesh_index];
                var had_position = false;
                var had_normal = false;
                var had_texcoord0 = false;
                var had_tangent = false;
                for (current_submesh.attributes[0..current_submesh.attributes_count]) |attrib| {
                    const accessor = attrib.data;
                    std.debug.assert(accessor.component_type == .r_32f);
                    const buffer_view = accessor.buffer_view.?;
                    std.debug.assert(buffer_view.buffer.data != null);
                    try positions.ensureTotalCapacity(positions.items.len + accessor.count);
                    try normals.ensureTotalCapacity(normals.items.len + accessor.count);
                    try texcoords0.ensureTotalCapacity(texcoords0.items.len + accessor.count);
                    try tangents.ensureTotalCapacity(tangents.items.len + accessor.count);
                    var offset = accessor.offset + buffer_view.offset;
                    var i: u32 = 0;
                    while (i < accessor.count) : (i += 1) {
                        const data_addr = @ptrCast([*]const u8, buffer_view.buffer.data) + offset;
                        if (attrib.type == .position) {
                            std.debug.assert(accessor.type == .vec3);
                            positions.appendAssumeCapacity(@ptrCast([*]const [3]f32, @alignCast(4, data_addr))[0]);
                            had_position = true;
                        } else if (attrib.type == .normal) {
                            std.debug.assert(accessor.type == .vec3);
                            normals.appendAssumeCapacity(@ptrCast([*]const [3]f32, @alignCast(4, data_addr))[0]);
                            had_normal = true;
                        } else if (attrib.type == .texcoord) {
                            std.debug.assert(accessor.type == .vec2);
                            texcoords0.appendAssumeCapacity(@ptrCast([*]const [2]f32, @alignCast(4, data_addr))[0]);
                            had_texcoord0 = true;
                        } else if (attrib.type == .tangent) {
                            std.debug.assert(accessor.type == .vec4);
                            tangents.appendAssumeCapacity(@ptrCast([*]const [4]f32, @alignCast(4, data_addr))[0]);
                            had_tangent = true;
                        }
                        offset += if (buffer_view.stride != 0) buffer_view.stride else accessor.stride;
                    }
                }
                std.debug.assert(had_position);
                {
                    var i: u32 = 0;
                    const count = positions.items.len - positions_start_index;
                    while (i < count) : (i += 1) {
                        if (!had_normal) {
                            normals.appendAssumeCapacity([3]f32 { 0.0, 0.0, 0.0 });
                        }
                        if (!had_texcoord0) {
                            texcoords0.appendAssumeCapacity([2]f32 { 0.0, 0.0 });
                        }
                        if (!had_tangent) {
                            tangents.appendAssumeCapacity([4]f32 { 0.0, 0.0, 0.0, 0.0 });
                        }
                    }
                }

                const num_indices: u32 = @intCast(u32, current_submesh.indices.?.count);
                try indices.ensureTotalCapacity(indices.items.len + num_indices);
                const accessor = current_submesh.indices.?;
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
                try mesh.submeshes.append(.{
                    .indices_start_index = @intCast(u32, indices_start_index),
                    .vertices_start_index = @intCast(u32, positions_start_index),
                    .index_count = @intCast(u32, indices.items.len - indices_start_index),
                    .vertex_count = @intCast(u32, positions.items.len - positions_start_index),
                    .material_index = assigned_material_index
                });
            }
        }

        try mesh.indices.ensureTotalCapacity(indices.items.len);
        for (indices.items) |index| {
            mesh.indices.appendAssumeCapacity(index);
        }

        try mesh.vertices.ensureTotalCapacity(positions.items.len);
        for (positions.items) |_, index| {
            mesh.vertices.appendAssumeCapacity(.{
                .position = positions.items[index],
                .normal = normals.items[index],
                .texcoord0 = texcoords0.items[index],
                .tangent = tangents.items[index]
            });
        }

        const num_images = @intCast(u32, gltf.images_count);
        try mesh.textures.ensureTotalCapacity(num_images);
        var image_index: u32 = 0;
        while (image_index < num_images) : (image_index += 1) {
            const image = &gltf.images.?[image_index];
            const asset_dir = std.fs.path.dirname(gltf_path) orelse ".";
            const texture_path = try std.fs.path.joinZ(self.allocator, &[_][]const u8 {
                asset_dir,
                try std.fmt.allocPrint(arena, "{s}", .{ image.uri.? })
            });
            var result = MeshTexture {
                .source = texture_path,
                .image = null
            };
            mesh.textures.appendAssumeCapacity(result);
        }

        var threads: std.ArrayList(std.Thread) = std.ArrayList(std.Thread).init(arena);
        const max_num_threads = @intCast(u32, std.Thread.getCpuCount() catch 4);
        try threads.ensureTotalCapacity(max_num_threads);
        image_index = 0;
        while (image_index < num_images) {
            var num_threads: u32 = max_num_threads;
            if (image_index + num_threads > num_images) {
                num_threads = num_images - image_index;
            }
            var t: u32 = 0;
            while (t < num_threads) : (t += 1) {
                var ctx = ImageLoadContext {
                    .source = mesh.textures.items[image_index + t].source,
                    .result = &mesh.textures.items[image_index + t].image
                };
                const thread = std.Thread.spawn(.{}, imageLoaderFunc, .{ ctx }) catch unreachable;
                threads.appendAssumeCapacity(thread);
            }
            for (threads.items) |thread| {
                thread.join();
            }
            threads.clearRetainingCapacity();
            image_index += num_threads;
        }

        for (mesh.textures.items) |*texture, index| {
            if (texture.image == null) {
                std.debug.print("Failed to load glTF texture {}: {s}\n", .{ index, texture.source });
            }
        }

        try mesh.materials.ensureTotalCapacity(num_materials);
        var material_index: u32 = 0;
        while (material_index < num_materials) : (material_index += 1) {
            const invalid_image_index = num_images;
            var base_color_tex_index: u32 = invalid_image_index;
            var metallic_roughness_tex_index: u32 = invalid_image_index;
            var normal_tex_index: u32 = invalid_image_index;
            const gltf_material = &gltf.materials.?[material_index];
            const mr = &gltf_material.pbr_metallic_roughness;
            if (gltf_material.has_pbr_metallic_roughness != 0) {
                image_index = 0;
                while (image_index < num_images) : (image_index += 1) {
                    const image = &gltf.images.?[image_index];
                    if (mr.base_color_texture.texture != null and mr.base_color_texture.texture.?.*.image.?.*.uri == image.uri) {
                        base_color_tex_index = image_index;
                    }
                    if (mr.metallic_roughness_texture.texture != null and mr.metallic_roughness_texture.texture.?.*.image.?.*.uri == image.uri) {
                        metallic_roughness_tex_index = image_index;
                    }
                    if (gltf_material.normal_texture.texture != null and gltf_material.normal_texture.texture.?.*.image.?.*.uri == image.uri) {
                        normal_tex_index = image_index;
                    }
                }
            }
            mesh.materials.appendAssumeCapacity(.{
                .base_color = mr.base_color_factor,
                .roughness = mr.roughness_factor,
                .metallic = mr.metallic_factor,
                .base_color_tex_index = if (base_color_tex_index != invalid_image_index) base_color_tex_index else null,
                .metallic_roughness_tex_index = if (metallic_roughness_tex_index != invalid_image_index) metallic_roughness_tex_index else null,
                .normal_tex_index = if (normal_tex_index != invalid_image_index) normal_tex_index else null,
            });
        }

        return mesh;
    }

    /// use with deferredReleaseCallback() to deferred-release and null out a ?StagingArea
    pub fn releaseStagingArea(area_opt_ptr: *anyopaque) void {
        var area_opt = @ptrCast(*?StagingArea, @alignCast(@alignOf(*?StagingArea), area_opt_ptr));
        area_opt.*.?.deinit();
        area_opt.* = null;
    }
};
