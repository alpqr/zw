const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const dxgi = zwin32.dxgi;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const d3d12d = zwin32.d3d12d;
const zm = @import("zmath");
const zstbi = @import("zstbi");
const imgui = @cImport({
    @cDefine("CIMGUI_DEFINE_ENUMS_AND_STRUCTS", "");
    @cDefine("CIMGUI_NO_EXPORT", "");
    @cInclude("cimgui.h");
});

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

    pub fn get(self: *DescriptorHeap, count: u32) Descriptor {
        std.debug.assert(self.size + count <= self.capacity);
        self.size += count;
        return self.at(self.size - count);
    }

    pub fn unget(self: *DescriptorHeap, count: u32) void {
        std.debug.assert(self.size >= count);
        self.size -= count;
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
                            last.map.set(first_index + j);
                            return last.heap.at(first_index);
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

    pub fn allocate(self: *StagingArea, size: u32) ?Allocation {
        const alloc_size = alignedSize(size, alignment);
        if (self.size + alloc_size > self.capacity) {
            return null;
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

    pub fn rotate(self: *Camera, dx: f32, dy: f32) void {
        self.pitch += 0.0025 * dy;
        self.yaw += 0.0025 * dx;
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

pub const Fw = struct {
    pub const Options = struct {
        window_size: Size = .{ .width = 1280, .height = 720 },
        window_name: [*:0]const u8 = "zigapp",
        enable_debug_layer: bool = false,
        swap_interval: u32 = 1,
        small_staging_area_capacity_per_frame: u32 = 16 * 1024 * 1024,
        shader_visible_cbv_srv_uav_heap_capacity_per_frame: u32 = 256,
        shader_visible_sampler_heap_capacity_per_frame: u32 = 16,
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
        transition_resource_barriers: []TransitionResourceBarrier,
        trb_next: u32,
        small_staging_areas: [max_frames_in_flight]StagingArea,
        shader_visible_cbv_srv_uav_heap: DescriptorHeap,
        shader_visible_cbv_srv_uav_heap_ranges: [max_frames_in_flight]DescriptorHeap,
        shader_visible_sampler_heap: DescriptorHeap,
        shader_visible_sampler_heap_ranges: [max_frames_in_flight]DescriptorHeap,
        depth_stencil_buffer: ObjectHandle,
        dsv: Descriptor,
        current_pipeline_handle: ObjectHandle,
    };
    d: *Data,

    fn fromWCHAR(work: []u8, src: []const u16) []const u8 {
        const len = std.unicode.utf16leToUtf8(work, src) catch 0;
        if (len > 0) {
            for (work) | c, idx | {
                if (c == 0)
                    return work[0..idx];
            }
        }
        return &.{};
    }

    pub fn init(allocator: std.mem.Allocator, options: Options) !Fw {
        var d = allocator.create(Data) catch unreachable;
        errdefer allocator.destroy(d);
        d.options = options;

        zstbi.init(allocator);
        errdefer zstbi.deinit();

        _ = imgui.igCreateContext(null);
        errdefer imgui.igDestroyContext(null);
        var io = imgui.igGetIO().?;
        io.*.BackendPlatformUserData = d;

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

        d.transition_resource_barriers = try allocator.alloc(TransitionResourceBarrier, transition_resource_barrier_buffer_size);
        errdefer allocator.free(d.transition_resource_barriers);
        d.trb_next = 0;

        for (d.small_staging_areas) |_, index| {
            d.small_staging_areas[index] = try StagingArea.init(d.device,
                                                                d.options.small_staging_area_capacity_per_frame,
                                                                .UPLOAD);
        }

        var heap_capacity = d.options.shader_visible_cbv_srv_uav_heap_capacity_per_frame;
        d.shader_visible_cbv_srv_uav_heap = try DescriptorHeap.init(d.device,
                                                                   max_frames_in_flight * heap_capacity,
                                                                   .CBV_SRV_UAV,
                                                                   d3d12.DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        errdefer d.shader_visible_cbv_srv_uav_heap.deinit();
        for (d.shader_visible_cbv_srv_uav_heap_ranges) |_, index| {
            d.shader_visible_cbv_srv_uav_heap_ranges[index] = DescriptorHeap.initWithExisting(
                d.shader_visible_cbv_srv_uav_heap,
                @intCast(u32, index * heap_capacity),
                heap_capacity);
        }

        heap_capacity = d.options.shader_visible_sampler_heap_capacity_per_frame;
        d.shader_visible_sampler_heap = try DescriptorHeap.init(d.device,
                                                                max_frames_in_flight * heap_capacity,
                                                                .SAMPLER,
                                                                d3d12.DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        errdefer d.shader_visible_sampler_heap.deinit();
        for (d.shader_visible_sampler_heap_ranges) |_, index| {
            d.shader_visible_sampler_heap_ranges[index] = DescriptorHeap.initWithExisting(
                d.shader_visible_sampler_heap,
                @intCast(u32, index * heap_capacity),
                heap_capacity);
        }

        d.current_frame_slot = 0;

        d.depth_stencil_buffer = ObjectHandle.invalid();
        d.dsv = Descriptor.invalid();

        d.current_pipeline_handle = ObjectHandle.invalid();

        var self = Fw {
            .d = d
        };

        try self.acquireSwapchainBuffers();
        try self.ensureDepthStencil();

        return self;
    }

    pub fn deinit(self: *Fw, allocator: std.mem.Allocator) void {
        self.waitGpu();
        for (self.d.swapchain_buffers) |swapchain_buffer| {
            self.d.resource_pool.remove(swapchain_buffer.handle);
        }
        self.d.shader_visible_sampler_heap.deinit();
        for (self.d.shader_visible_sampler_heap_ranges) |*h| {
            h.deinit();
        }
        self.d.shader_visible_cbv_srv_uav_heap.deinit();
        for (self.d.shader_visible_cbv_srv_uav_heap_ranges) |*h| {
            h.deinit();
        }
        for (self.d.small_staging_areas) |*staging_area| {
            staging_area.deinit();
        }
        allocator.free(self.d.transition_resource_barriers);
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
        zstbi.deinit();
        allocator.destroy(self.d);
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
        self.d.depth_stencil_buffer = try self.createDepthStencilBuffer(self.d.swapchain_size);
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

    pub fn getCurrentSmallStagingArea(self: *const Fw) *StagingArea {
        return &self.d.small_staging_areas[self.d.current_frame_slot];
    }

    pub fn getShaderVisibleCbvSrvUavHeap(self: *const Fw) *d3d12.IDescriptorHeap {
        return self.d.shader_visible_cbv_srv_uav_heap.heap.?;
    }

    pub fn getCurrentShaderVisibleCbvSrvUavHeapRange(self: *const Fw) *DescriptorHeap {
        return &self.d.shader_visible_cbv_srv_uav_heap_ranges[self.d.current_frame_slot];
    }

    pub fn getShaderVisibleSamplerHeap(self: *const Fw) *d3d12.IDescriptorHeap {
        return self.d.shader_visible_sampler_heap.heap.?;
    }

    pub fn getCurrentShaderVisibleSamplerHeapRange(self: *const Fw) *DescriptorHeap {
        return &self.d.shader_visible_sampler_heap_ranges[self.d.current_frame_slot];
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
            self.d.current_frame_slot = 0;
            try self.ensureDepthStencil();
        }

        const cmd_allocator = self.d.cmd_allocators[self.d.current_frame_slot];
        try zwin32.hrErrorOnFail(cmd_allocator.Reset());
        try zwin32.hrErrorOnFail(self.d.cmd_list.Reset(cmd_allocator, null));

        self.addTransitionBarrier(self.getBackBufferObjectHandle(), d3d12.RESOURCE_STATE_RENDER_TARGET);
        self.recordTransitionBarriers();

        self.resetTrackedState();

        self.d.small_staging_areas[self.d.current_frame_slot].reset();
        self.d.shader_visible_cbv_srv_uav_heap_ranges[self.d.current_frame_slot].reset();
        self.d.shader_visible_sampler_heap_ranges[self.d.current_frame_slot].reset();

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

        self.d.current_frame_slot = (self.d.current_frame_slot + 1) % max_frames_in_flight;
        self.d.current_back_buffer_index = self.d.swapchain.GetCurrentBackBufferIndex();
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

    pub fn lookupOrCreateGraphicsPipeline(self: *Fw,
                                          pso_desc: *d3d12.GRAPHICS_PIPELINE_STATE_DESC,
                                          rs_desc: *const d3d12.VERSIONED_ROOT_SIGNATURE_DESC) !ObjectHandle {
        var sha: [Pipeline.sha_length]u8 = undefined;
        Pipeline.getGraphicsPipelineSha(pso_desc, &sha);
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
            pso_desc.pRootSignature = rs;

            var pso: *d3d12.IPipelineState = undefined;
            try zwin32.hrErrorOnFail(self.d.device.CreateGraphicsPipelineState(pso_desc,
                                                                               &d3d12.IID_IPipelineState,
                                                                               @ptrCast(*?*anyopaque, &pso)));
            errdefer _ = pso.Release();

            break :blk try Pipeline.addToPool(&self.d.pipeline_pool, pso, rs, Pipeline.Type.Graphics);
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
        const res = self.d.resource_pool.lookupRef(resource_handle).?;
        var p: [*]u8 = undefined;
        try zwin32.hrErrorOnFail(res.resource.Map(0,  &.{ .Begin = 0, .End = 0 }, @ptrCast(*?*anyopaque, &p)));
        const slice = p[0..res.desc.Width];
        return std.mem.bytesAsSlice(T, @alignCast(@alignOf(T), slice));
    }

    fn createDepthStencilBuffer(self: *Fw, pixel_size: Size) !ObjectHandle {
        var heap_properties = std.mem.zeroes(d3d12.HEAP_PROPERTIES);
        heap_properties.Type = .DEFAULT;
        var resource: *d3d12.IResource = undefined;
        try zwin32.hrErrorOnFail(self.d.device.CreateCommittedResource(
            &heap_properties,
            d3d12.HEAP_FLAG_NONE,
            &.{
                .Dimension = .TEXTURE2D,
                .Alignment = 0,
                .Width = pixel_size.width,
                .Height = pixel_size.height,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = Fw.dsv_format,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .UNKNOWN,
                .Flags = d3d12.RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | d3d12.RESOURCE_FLAG_DENY_SHADER_RESOURCE
            },
            d3d12.RESOURCE_STATE_DEPTH_WRITE,
            &.{
                .Format = Fw.dsv_format,
                .u = .{
                    .DepthStencil = .{
                        .Depth = 1.0,
                        .Stencil = 0
                    }
                }
            },
            &d3d12.IID_IResource,
            @ptrCast(*?*anyopaque, &resource)));
        errdefer _ = resource.Release();
        return try Resource.addToPool(&self.d.resource_pool, resource, d3d12.RESOURCE_STATE_DEPTH_WRITE);
    }

    pub fn createTexture2DSimple(self: *Fw, format: dxgi.FORMAT, pixel_size: Size) !ObjectHandle {
        var heap_properties = std.mem.zeroes(d3d12.HEAP_PROPERTIES);
        heap_properties.Type = .DEFAULT;
        var resource: *d3d12.IResource = undefined;
        try zwin32.hrErrorOnFail(self.d.device.CreateCommittedResource(
            &heap_properties,
            d3d12.HEAP_FLAG_NONE,
            &.{
                .Dimension = .TEXTURE2D,
                .Alignment = 0,
                .Width = pixel_size.width,
                .Height = pixel_size.height,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = format,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .UNKNOWN,
                .Flags = d3d12.RESOURCE_FLAG_NONE
            },
            d3d12.RESOURCE_STATE_COMMON,
            null,
            &d3d12.IID_IResource,
            @ptrCast(*?*anyopaque, &resource)));
        errdefer _ = resource.Release();
        return try Resource.addToPool(&self.d.resource_pool, resource, d3d12.RESOURCE_STATE_COMMON);
    }

    pub fn uploadTexture2DSimple(self: *Fw,
                                 texture: ObjectHandle,
                                 data: []u8,
                                 data_bytes_per_line: u32,
                                 staging: *StagingArea) void {
        var tex_opt = self.d.resource_pool.lookupRef(texture);
        if (tex_opt == null)
            return;
        var tex = tex_opt.?;
        var layout: [1]d3d12.PLACED_SUBRESOURCE_FOOTPRINT = undefined;
        var required_image_data_size: u64 = undefined;
        self.d.device.GetCopyableFootprints(&tex.desc, 0, 1, 0, &layout, null, null, &required_image_data_size);
        const required_bytes_per_line = layout[0].Footprint.RowPitch; // multiple of 256
        std.debug.assert(data_bytes_per_line <= required_bytes_per_line);
        const alloc = staging.allocate(@intCast(u32, required_image_data_size)).?;
        var y: u32 = 0;
        while (y < tex.desc.Height) : (y += 1) {
            const src_begin = y * data_bytes_per_line;
            const src_end = src_begin + data_bytes_per_line;
            const dst_begin = y * required_bytes_per_line;
            const dst_end = dst_begin + required_bytes_per_line;
            std.mem.copy(u8, alloc.cpu_slice[dst_begin..dst_end], data[src_begin..src_end]);
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
        var dd: ?*Data = null;
        if (imgui.igGetCurrentContext() != null) {
            var io = imgui.igGetIO().?;
            dd = @ptrCast(*Data, @alignCast(@alignOf(*Data), io.*.BackendPlatformUserData));
        }
        switch (message) {
            w32.user32.WM_DESTROY => {
                w32.user32.PostQuitMessage(0);
            },
            w32.user32.WM_SIZE => {
                if (dd) |d| {
                    const new_size = Size {
                        .width = @intCast(u32, lparam & 0xFFFF),
                        .height = @intCast(u32, lparam >> 16)
                    };
                    if (!std.meta.eql(d.window_size, new_size)) {
                        d.window_size = new_size;
                        std.debug.print("new width {} height {}\n", .{d.window_size.width, d.window_size.height});
                    }
                }
            },
            w32.user32.WM_PAINT => {
                var ps = std.mem.zeroes(PAINTSTRUCT);
                _ = BeginPaint(window, &ps);
                //_ = FillRect(ps.hdc, &ps.rcPaint, @intToPtr(w32.HBRUSH, 5)); // COLOR_WINDOW
                _ = EndPaint(window, &ps);
            },
            else => {
                return w32.user32.defWindowProcA(window, message, wparam, lparam);
            },
        }
        return 0;
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
};
