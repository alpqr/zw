const std = @import("std");
const zwin32 = @import("zwin32");
const w32 = zwin32.base;
const dxgi = zwin32.dxgi;
const d3d = zwin32.d3d;
const d3d12 = zwin32.d3d12;
const d3d12d = zwin32.d3d12d;

const imgui = @cImport({
    @cDefine("CIMGUI_DEFINE_ENUMS_AND_STRUCTS", "");
    @cDefine("CIMGUI_NO_EXPORT", "");
    @cInclude("cimgui.h");
});

pub const Descriptor = struct {
    cpu_handle: d3d12.CPU_DESCRIPTOR_HANDLE,
    gpu_handle: d3d12.GPU_DESCRIPTOR_HANDLE
};

pub const DescriptorHeap = struct {
    heap: *d3d12.IDescriptorHeap,
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
        if ((heap_flags & d3d12.DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) != 0)
            gpu_handle = heap.GetGPUDescriptorHandleForHeapStart();

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

    pub fn deinit(self: *DescriptorHeap) void {
        _ = self.heap.Release();
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

        const cpu_handle = d3d12.CPU_DESCRIPTOR_HANDLE { .ptr = self.base.cpu_handle.ptr + start_offset };
        var gpu_handle = d3d12.GPU_DESCRIPTOR_HANDLE { .ptr = 0 };
        if (self.base.gpu_handle.ptr != 0)
            gpu_handle = d3d12.GPU_DESCRIPTOR_HANDLE { .ptr = self.base.gpu_handle.ptr + start_offset };

        return Descriptor {
            .cpu_handle = cpu_handle,
            .gpu_handle = gpu_handle
        };
    }

    pub fn reset(self: *DescriptorHeap) void {
        self.size = 0;
    }
};

pub const DescriptorPool = struct {
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
                heap_type: d3d12.DESCRIPTOR_HEAP_TYPE,
                heap_flags: d3d12.DESCRIPTOR_HEAP_FLAGS) !DescriptorPool {
        var firstHeap = try DescriptorHeap.init(device, descriptors_per_heap, heap_type, heap_flags);
        errdefer firstHeap.deinit();
        const h = HeapWithMap {
            .heap = firstHeap,
            .map = std.StaticBitSet(descriptors_per_heap).initEmpty()
        };
        var heaps = std.ArrayList(HeapWithMap).init(allocator);
        try heaps.append(h);
        return DescriptorPool {
            .heaps = heaps,
            .descriptor_byte_size = firstHeap.descriptor_byte_size,
            .device = device,
        };
    }

    pub fn deinit(self: *DescriptorPool) void {
        for (self.heaps.items) |*h| {
            h.heap.deinit();
        }
        self.heaps.deinit();
    }

    pub fn allocate(self: *DescriptorPool, count: u32) !Descriptor {
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

    pub fn release(self: *DescriptorPool, descriptor: Descriptor, count: u32) void {
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
    pub const alignment: u32 = 512;

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
};

pub const Fw = struct {
    pub const Options = struct {
        window_width: u32 = 1280,
        window_height: u32 = 720,
        window_name: [*:0]const u8 = "zigapp",
        enable_debug_layer: bool = false,
        swap_interval: u32 = 1,
        small_upload_staging_area_capacity_per_frame: u32 = 16 * 1024 * 1024,
    };

    pub const max_frames_in_flight = 2;
    const swapchain_buffer_count = 3;
    const transition_resource_barrier_pool_size = 64;

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
        window_width: u32,
        window_height: u32,
        dxgiFactory: *dxgi.IFactory6,
        device: *d3d12.IDevice9,
        cmdqueue: *d3d12.ICommandQueue,
        swapchain: *dxgi.ISwapChain3,
        swapchain_flags: u32,
        swapchain_width: u32,
        swapchain_height: u32,
        frame_fence: *d3d12.IFence,
        frame_fence_event: w32.HANDLE,
        frame_fence_counter: u64,
        cmdallocators: [max_frames_in_flight]*d3d12.ICommandAllocator,
        cmdlist: *d3d12.IGraphicsCommandList6,
        rtv_pool: DescriptorPool,
        swapchain_buffers: [swapchain_buffer_count]SwapchainBuffer,
        current_frame_slot: u32,
        current_back_buffer_index: u32,
        present_allow_tearing_supported: bool,
        resource_pool: ObjectPool(Resource),
        pipeline_pool: ObjectPool(Pipeline),
        pipeline_cache: Pipeline.Cache,
        transition_resource_barriers: []TransitionResourceBarrier,
        trb_next: u32,
        small_upload_staging_areas: [max_frames_in_flight]StagingArea,
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
        d.window_width = 0;
        d.window_height = 0;

        const style = w32.user32.WS_OVERLAPPEDWINDOW; // resizable, minimize, maximize
        var rect = w32.RECT {
            .left = 0,
            .top = 0,
            .right = @intCast(i32, options.window_width),
            .bottom = @intCast(i32, options.window_height)
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

        d.swapchain_width = d.window_width;
        d.swapchain_height = d.window_height;

        var swapchain: *dxgi.ISwapChain1 = undefined;
        d.swapchain_flags = 0;
        if (options.swap_interval == 0 and d.present_allow_tearing_supported)
            d.swapchain_flags |= dxgi.SWAP_CHAIN_FLAG_ALLOW_TEARING;
        const swapchainDesc = &dxgi.SWAP_CHAIN_DESC1 {
            .Width = std.math.max(1, d.window_width),
            .Height = std.math.max(1, d.window_height),
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

        var cmdallocators: [max_frames_in_flight]*d3d12.ICommandAllocator = undefined;
        for (cmdallocators) |_, index| {
            try zwin32.hrErrorOnFail(device.CreateCommandAllocator(.DIRECT,
                                                                   &d3d12.IID_ICommandAllocator,
                                                                   @ptrCast(*?*anyopaque, &cmdallocators[index])));
        }
        errdefer for (cmdallocators) |cmdallocator| { _ = cmdallocator.Release(); };
        d.cmdallocators = cmdallocators;

        var cmdlist: *d3d12.IGraphicsCommandList6 = undefined;
        try zwin32.hrErrorOnFail(device.CreateCommandList(0, .DIRECT, cmdallocators[0], null, 
                                                          &d3d12.IID_IGraphicsCommandList6,
                                                          @ptrCast(*?*anyopaque, &cmdlist)));
        errdefer _ = cmdlist.Release();
        d.cmdlist = cmdlist;
        try zwin32.hrErrorOnFail(cmdlist.Close());

        d.rtv_pool = try DescriptorPool.init(allocator, device, .RTV, d3d12.DESCRIPTOR_HEAP_FLAG_NONE);
        errdefer d.rtv_pool.deinit();

        d.resource_pool = try ObjectPool(Resource).init(allocator);
        errdefer d.resource_pool.deinit();

        d.pipeline_pool = try ObjectPool(Pipeline).init(allocator);
        errdefer d.pipeline_pool.deinit();

        d.pipeline_cache = Pipeline.Cache.init(allocator);
        errdefer d.pipeline_cache.deinit();

        d.transition_resource_barriers = try allocator.alloc(TransitionResourceBarrier, transition_resource_barrier_pool_size);
        errdefer allocator.free(d.transition_resource_barriers);
        d.trb_next = 0;

        for (d.small_upload_staging_areas) |_, index| {
            d.small_upload_staging_areas[index] = try StagingArea.init(d.device,
                                                                       d.options.small_upload_staging_area_capacity_per_frame,
                                                                       .UPLOAD);
        }

        d.current_frame_slot = 0;

        d.current_pipeline_handle = ObjectHandle.invalid();

        var self = Fw {
            .d = d
        };
        try self.acquireSwapchainBuffers();
        return self;
    }

    pub fn deinit(self: *Fw, allocator: std.mem.Allocator) void {
        self.waitGpu();
        for (self.d.swapchain_buffers) |swapchain_buffer| {
            self.d.resource_pool.remove(swapchain_buffer.handle);
        }
        for (self.d.small_upload_staging_areas) |*upload_staging_area| {
            upload_staging_area.deinit();
        }
        allocator.free(self.d.transition_resource_barriers);
        self.d.pipeline_cache.deinit();
        self.d.pipeline_pool.deinit();
        self.d.resource_pool.deinit();
        self.d.rtv_pool.deinit();
        _ = self.d.cmdlist.Release();
        for (self.d.cmdallocators) |cmdallocator| {
            _ = cmdallocator.Release();
        }
        w32.CloseHandle(self.d.frame_fence_event);
        _ = self.d.frame_fence.Release();
        _ = self.d.swapchain.Release();
        _ = self.d.cmdqueue.Release();
        _ = self.d.device.Release();
        _ = self.d.dxgiFactory.Release();
        imgui.igDestroyContext(null);
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
        var barriers: [transition_resource_barrier_pool_size]d3d12.RESOURCE_BARRIER = undefined;
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
            self.d.cmdlist.ResourceBarrier(count, &barriers);
        }
        self.d.trb_next = 0;
    }

    pub fn getDevice(self: *const Fw) *d3d12.IDevice9 {
        return self.d.device;
    }

    pub fn getCommandList(self: *const Fw) *d3d12.IGraphicsCommandList6 {
        return self.d.cmdlist;
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

    pub fn getCurrentSmallUploadStagingArea(self: *const Fw) *StagingArea {
        return &self.d.small_upload_staging_areas[self.d.current_frame_slot];
    }

    pub fn getBackBufferCpuDescriptorHandle(self: *const Fw) d3d12.CPU_DESCRIPTOR_HANDLE {
        return self.d.swapchain_buffers[self.d.current_back_buffer_index].descriptor.cpu_handle;
    }

    pub fn getBackBufferObjectHandle(self: *const Fw) ObjectHandle {
        return self.d.swapchain_buffers[self.d.current_back_buffer_index].handle;
    }

    pub fn getBackBufferPixelSize(self: *const Fw) struct { width: u32, height: u32 } {
        return .{
            .width = self.d.swapchain_width,
            .height = self.d.swapchain_height
        };
    }

    pub fn beginFrame(self: *Fw) !void {
        self.waitGpuIfAhead();

        if (self.d.swapchain_width != self.d.window_width or self.d.swapchain_height != self.d.window_height) {
            self.d.swapchain_width = self.d.window_width;
            self.d.swapchain_height = self.d.window_height;
            std.debug.print("Resizing swapchain {}x{}\n", .{self.d.swapchain_width, self.d.swapchain_height});
            self.waitGpu();
            for (self.d.swapchain_buffers) |*swapchain_buffer| {
                self.d.resource_pool.remove(swapchain_buffer.handle);
                self.d.rtv_pool.release(swapchain_buffer.descriptor, 1);
            }
            try zwin32.hrErrorOnFail(self.d.swapchain.ResizeBuffers(swapchain_buffer_count,
                                                                    std.math.max(1, self.d.window_width),
                                                                    std.math.max(1, self.d.window_height),
                                                                    .R8G8B8A8_UNORM,
                                                                    self.d.swapchain_flags));
            try self.acquireSwapchainBuffers();
            self.d.current_frame_slot = 0;
        }

        const ca = self.d.cmdallocators[self.d.current_frame_slot];
        try zwin32.hrErrorOnFail(ca.Reset());
        try zwin32.hrErrorOnFail(self.d.cmdlist.Reset(ca, null));

        self.addTransitionBarrier(self.getBackBufferObjectHandle(), d3d12.RESOURCE_STATE_RENDER_TARGET);
        self.recordTransitionBarriers();

        self.resetTrackedState();

        self.d.small_upload_staging_areas[self.d.current_frame_slot].reset();
    }

    pub fn endFrame(self: *Fw) !void {
        self.addTransitionBarrier(self.getBackBufferObjectHandle(), d3d12.RESOURCE_STATE_PRESENT);
        self.recordTransitionBarriers();

        try zwin32.hrErrorOnFail(self.d.cmdlist.Close());

        const list = [_]*d3d12.ICommandList {
            @ptrCast(*d3d12.ICommandList, self.d.cmdlist)
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
        self.d.cmdlist.SetPipelineState(pipeline.pso);
        switch (pipeline.ptype) {
            .Graphics => self.d.cmdlist.SetGraphicsRootSignature(pipeline.rs),
            .Compute => self.d.cmdlist.SetComputeRootSignature(pipeline.rs)
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
                    const newWidth = @intCast(u32, lparam & 0xFFFF);
                    const newHeight = @intCast(u32, lparam >> 16);
                    if (d.window_width != newWidth or d.window_height != newHeight) {
                        d.window_width = newWidth;
                        d.window_height = newHeight;
                        std.debug.print("new width {} height {}\n", .{d.window_width, d.window_height});
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
