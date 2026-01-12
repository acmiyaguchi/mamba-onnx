//! Mamba Selective Scan - Zig implementation
//!
//! AVX2-optimized selective scan kernel for ONNX Runtime.
//! Features a warm thread pool for low-latency parallel execution.
//! Build: zig build -Dbackend=zig -Doptimize=ReleaseFast

const std = @import("std");
const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

// =============================================================================
// SIMD Types
// =============================================================================

const Vec8 = @Vector(8, f32);
const Vec8i = @Vector(8, i32);
const Vec8u = @Vector(8, u32);

// =============================================================================
// Thread Pool
// =============================================================================

const MAX_THREADS = 32;

/// Work function type - processes indices [start, end) with given context
const WorkFn = *const fn (start: usize, end: usize, ctx: *anyopaque) void;

/// Global thread pool for warm thread reuse
/// Uses a generation-based synchronization to avoid races between work batches
const ThreadPool = struct {
    threads: [MAX_THREADS]std.Thread = undefined,
    num_workers: usize = 0,

    // Synchronization
    mutex: std.Thread.Mutex = .{},
    work_available: std.Thread.Condition = .{},
    work_done: std.Thread.Condition = .{},

    // Work specification (protected by mutex)
    work_fn: ?WorkFn = null,
    work_ctx: ?*anyopaque = null,
    total_work: usize = 0,

    // Generation counter to track work batches
    generation: usize = 0,

    // Atomic work counter for work-stealing
    next_idx: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    // Barrier for completion - counts down to 0
    workers_remaining: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    shutdown: bool = false,
    initialized: bool = false,

    fn init(self: *ThreadPool) void {
        if (self.initialized) return;

        const cpu_count = std.Thread.getCpuCount() catch 4;
        self.num_workers = @min(cpu_count, MAX_THREADS) - 1; // -1 because main thread participates

        // Spawn worker threads
        for (0..self.num_workers) |i| {
            self.threads[i] = std.Thread.spawn(.{}, workerLoop, .{self}) catch {
                self.num_workers = i;
                break;
            };
        }

        self.initialized = true;
    }

    fn deinit(self: *ThreadPool) void {
        if (!self.initialized) return;

        // Signal shutdown
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.shutdown = true;
        }
        self.work_available.broadcast();

        // Join all workers
        for (0..self.num_workers) |i| {
            self.threads[i].join();
        }

        self.initialized = false;
    }

    fn parallelFor(self: *ThreadPool, total: usize, work_fn: WorkFn, ctx: *anyopaque) void {
        if (!self.initialized) self.init();

        if (total == 0) return;

        // For small work or single thread, run directly
        if (self.num_workers == 0) {
            work_fn(0, total, ctx);
            return;
        }

        const num_participants = self.num_workers + 1; // +1 for main thread

        // Setup work under lock
        {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.work_fn = work_fn;
            self.work_ctx = ctx;
            self.total_work = total;
            self.next_idx.store(0, .release);
            self.workers_remaining.store(num_participants, .release);
            self.generation +%= 1;
        }

        // Wake workers
        self.work_available.broadcast();

        // Main thread participates in work
        self.processWork(work_fn, ctx, total);

        // Main thread signals completion
        const remaining = self.workers_remaining.fetchSub(1, .acq_rel);
        if (remaining == 1) {
            // Main thread was the last one - we're done
            self.mutex.lock();
            defer self.mutex.unlock();
            self.work_fn = null;
            return;
        }

        // Wait for workers to finish
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.workers_remaining.load(.acquire) > 0) {
            self.work_done.wait(&self.mutex);
        }
        self.work_fn = null;
    }

    fn workerLoop(self: *ThreadPool) void {
        var my_gen: usize = 0;

        while (true) {
            var work_fn: WorkFn = undefined;
            var work_ctx: *anyopaque = undefined;
            var total: usize = undefined;

            // Wait for new work
            {
                self.mutex.lock();
                defer self.mutex.unlock();

                while ((self.generation == my_gen or self.work_fn == null) and !self.shutdown) {
                    self.work_available.wait(&self.mutex);
                }

                if (self.shutdown) return;

                // Capture work params and update our generation
                my_gen = self.generation;
                work_fn = self.work_fn.?;
                work_ctx = self.work_ctx.?;
                total = self.total_work;
            }

            // Process work
            self.processWork(work_fn, work_ctx, total);

            // Signal completion
            const remaining = self.workers_remaining.fetchSub(1, .acq_rel);
            if (remaining == 1) {
                // Last worker - signal main thread
                self.mutex.lock();
                defer self.mutex.unlock();
                self.work_done.signal();
            }
        }
    }

    fn processWork(self: *ThreadPool, work_fn: WorkFn, ctx: *anyopaque, total: usize) void {
        // Work-stealing loop - grab one item at a time for good load balancing
        while (true) {
            const idx = self.next_idx.fetchAdd(1, .acq_rel);
            if (idx >= total) break;
            work_fn(idx, idx + 1, ctx);
        }
    }
};

var global_pool: ThreadPool = .{};

// =============================================================================
// Fast Math
// =============================================================================

/// Fast exp approximation using polynomial + IEEE 754 bit tricks
fn fast_exp(x: Vec8) Vec8 {
    const max_val: Vec8 = @splat(88.376);
    const min_val: Vec8 = @splat(-88.376);
    const clamped = @min(max_val, @max(min_val, x));

    const log2e: Vec8 = @splat(1.44269504089);
    const ln2: Vec8 = @splat(0.6931471805599453);

    const t = clamped * log2e;
    const t_floor = @floor(t);
    const f = clamped - t_floor * ln2;

    const c0: Vec8 = @splat(1.0);
    const c1: Vec8 = @splat(1.0);
    const c2: Vec8 = @splat(0.5);
    const c3: Vec8 = @splat(0.166666666666);
    const c4: Vec8 = @splat(0.041666666666);
    const c5: Vec8 = @splat(0.008333333333);

    var p = @mulAdd(Vec8, c5, f, c4);
    p = @mulAdd(Vec8, p, f, c3);
    p = @mulAdd(Vec8, p, f, c2);
    p = @mulAdd(Vec8, p, f, c1);
    p = @mulAdd(Vec8, p, f, c0);

    const n: Vec8i = @intFromFloat(t_floor);
    const bias: Vec8i = @splat(127);
    const exp_bits: Vec8u = @bitCast(n +% bias);
    const pow2n: Vec8 = @bitCast(exp_bits << @splat(23));

    return p * pow2n;
}

fn fast_softplus(x: f32) f32 {
    if (x > 20.0) return x;
    if (x < -20.0) return @exp(x);
    return std.math.log1p(@exp(x));
}

fn fast_silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

// =============================================================================
// Selective Scan Kernel (Thread Pool)
// =============================================================================

/// Work context for parallel kernel execution
fn KernelContext(comptime use_fused: bool, comptime use_exact: bool) type {
    return struct {
        dim: usize,
        seqlen: usize,
        u_data: [*]const f32,
        delta_data: [*]const f32,
        A_data: [*]const f32,
        B_data: [*]const f32,
        C_data: [*]const f32,
        D_data: [*]const f32,
        z_data: ?[*]const f32,
        out_data: [*]f32,

        const Self = @This();

        fn workFn(start: usize, end: usize, ctx_ptr: *anyopaque) void {
            const self: *const Self = @ptrCast(@alignCast(ctx_ptr));
            var idx = start;
            while (idx < end) : (idx += 1) {
                const b = idx / self.dim;
                const d = idx % self.dim;
                processOneDim(
                    use_fused,
                    use_exact,
                    b,
                    d,
                    self.dim,
                    self.seqlen,
                    self.u_data,
                    self.delta_data,
                    self.A_data,
                    self.B_data,
                    self.C_data,
                    self.D_data,
                    self.z_data,
                    self.out_data,
                );
            }
        }
    };
}

/// Process a single (batch, dim) pair - the inner sequential scan
fn processOneDim(
    comptime use_fused: bool,
    comptime use_exact: bool,
    b: usize,
    d: usize,
    dim: usize,
    seqlen: usize,
    u_data: [*]const f32,
    delta_data: [*]const f32,
    A_data: [*]const f32,
    B_data: [*]const f32,
    C_data: [*]const f32,
    D_data: [*]const f32,
    z_data: ?[*]const f32,
    out_data: [*]f32,
) void {
    const A_ptr = A_data + d * 16;
    const D_val = D_data[d];

    const A_0: Vec8 = A_ptr[0..8].*;
    const A_1: Vec8 = (A_ptr + 8)[0..8].*;

    var h_0: Vec8 = @splat(0);
    var h_1: Vec8 = @splat(0);

    for (0..seqlen) |l| {
        const u_idx = b * dim * seqlen + d * seqlen + l;
        const bc_idx = b * seqlen * 16 + l * 16;

        const u_t = u_data[u_idx];
        var delta_t = delta_data[u_idx];

        if (use_fused) {
            delta_t = fast_softplus(delta_t);
        }

        const u_vec: Vec8 = @splat(u_t);
        const delta_vec: Vec8 = @splat(delta_t);

        const B_ptr = B_data + bc_idx;
        const B_t_0: Vec8 = B_ptr[0..8].*;
        const B_t_1: Vec8 = (B_ptr + 8)[0..8].*;

        if (use_exact) {
            const A_bar_0 = fast_exp(delta_vec * A_0);
            const A_bar_1 = fast_exp(delta_vec * A_1);
            const B_bar_0 = delta_vec * B_t_0;
            const B_bar_1 = delta_vec * B_t_1;

            h_0 = @mulAdd(Vec8, A_bar_0, h_0, B_bar_0 * u_vec);
            h_1 = @mulAdd(Vec8, A_bar_1, h_1, B_bar_1 * u_vec);
        } else {
            const tmp_0 = @mulAdd(Vec8, A_0, h_0, B_t_0 * u_vec);
            const tmp_1 = @mulAdd(Vec8, A_1, h_1, B_t_1 * u_vec);
            h_0 = @mulAdd(Vec8, delta_vec, tmp_0, h_0);
            h_1 = @mulAdd(Vec8, delta_vec, tmp_1, h_1);
        }

        const C_ptr = C_data + bc_idx;
        const C_t_0: Vec8 = C_ptr[0..8].*;
        const C_t_1: Vec8 = (C_ptr + 8)[0..8].*;

        const y_0 = C_t_0 * h_0;
        const y_1 = C_t_1 * h_1;

        var y_scalar = @reduce(.Add, y_0) + @reduce(.Add, y_1);
        y_scalar += D_val * u_t;

        if (use_fused) {
            if (z_data) |z| {
                y_scalar *= fast_silu(z[u_idx]);
            }
        }

        out_data[u_idx] = y_scalar;
    }
}

fn selectiveScanKernel(
    comptime use_fused: bool,
    comptime use_exact: bool,
    batch: usize,
    dim: usize,
    seqlen: usize,
    u_data: [*]const f32,
    delta_data: [*]const f32,
    A_data: [*]const f32,
    B_data: [*]const f32,
    C_data: [*]const f32,
    D_data: [*]const f32,
    z_data: ?[*]const f32,
    out_data: [*]f32,
) void {
    const total_work = batch * dim;

    // Create kernel context on stack
    const Ctx = KernelContext(use_fused, use_exact);
    var ctx = Ctx{
        .dim = dim,
        .seqlen = seqlen,
        .u_data = u_data,
        .delta_data = delta_data,
        .A_data = A_data,
        .B_data = B_data,
        .C_data = C_data,
        .D_data = D_data,
        .z_data = z_data,
        .out_data = out_data,
    };

    // Use thread pool for parallel execution
    // Each (batch, dim) pair is a work unit for good load balancing
    global_pool.parallelFor(total_work, Ctx.workFn, @ptrCast(&ctx));
}

// =============================================================================
// ONNX Runtime Integration
// =============================================================================

const KernelState = struct {
    api: *const c.OrtApi,
    use_fused: bool,
    use_exact: bool,
};

fn createKernelFn(
    comptime use_fused: bool,
    comptime use_exact: bool,
) *const fn (?*const c.OrtCustomOp, [*c]const c.OrtApi, ?*const c.OrtKernelInfo) callconv(.c) ?*anyopaque {
    return &struct {
        fn create(_: ?*const c.OrtCustomOp, api_ptr: [*c]const c.OrtApi, _: ?*const c.OrtKernelInfo) callconv(.c) ?*anyopaque {
            const state = std.heap.c_allocator.create(KernelState) catch return null;
            state.* = .{
                .api = @ptrCast(api_ptr), // Cast C pointer to Zig pointer
                .use_fused = use_fused,
                .use_exact = use_exact,
            };
            return state;
        }
    }.create;
}

fn computeKernel(kernel: ?*anyopaque, context: ?*c.OrtKernelContext) callconv(.c) void {
    const state: *KernelState = @ptrCast(@alignCast(kernel orelse return));
    const api = state.api;
    const ctx = context orelse return;

    // Get inputs
    var u_val: ?*const c.OrtValue = null;
    var delta_val: ?*const c.OrtValue = null;
    var A_val: ?*const c.OrtValue = null;
    var B_val: ?*const c.OrtValue = null;
    var C_val: ?*const c.OrtValue = null;
    var D_val: ?*const c.OrtValue = null;
    var z_val: ?*const c.OrtValue = null;

    _ = api.KernelContext_GetInput.?(ctx, 0, &u_val);
    _ = api.KernelContext_GetInput.?(ctx, 1, &delta_val);
    _ = api.KernelContext_GetInput.?(ctx, 2, &A_val);
    _ = api.KernelContext_GetInput.?(ctx, 3, &B_val);
    _ = api.KernelContext_GetInput.?(ctx, 4, &C_val);
    _ = api.KernelContext_GetInput.?(ctx, 5, &D_val);
    if (state.use_fused) {
        _ = api.KernelContext_GetInput.?(ctx, 6, &z_val);
    }

    // Get shape
    var u_info: ?*c.OrtTensorTypeAndShapeInfo = null;
    _ = api.GetTensorTypeAndShape.?(u_val, &u_info);
    var u_dims: [3]i64 = undefined;
    _ = api.GetDimensions.?(u_info, &u_dims, 3);
    api.ReleaseTensorTypeAndShapeInfo.?(u_info);

    const batch: usize = @intCast(u_dims[0]);
    const dim: usize = @intCast(u_dims[1]);
    const seqlen: usize = @intCast(u_dims[2]);

    // Get data pointers
    var u_data: ?*anyopaque = null;
    var delta_data: ?*anyopaque = null;
    var A_data: ?*anyopaque = null;
    var B_data: ?*anyopaque = null;
    var C_data: ?*anyopaque = null;
    var D_data: ?*anyopaque = null;
    var z_data: ?*anyopaque = null;

    _ = api.GetTensorMutableData.?(@constCast(u_val), &u_data);
    _ = api.GetTensorMutableData.?(@constCast(delta_val), &delta_data);
    _ = api.GetTensorMutableData.?(@constCast(A_val), &A_data);
    _ = api.GetTensorMutableData.?(@constCast(B_val), &B_data);
    _ = api.GetTensorMutableData.?(@constCast(C_val), &C_data);
    _ = api.GetTensorMutableData.?(@constCast(D_val), &D_data);
    if (state.use_fused) {
        _ = api.GetTensorMutableData.?(@constCast(z_val), &z_data);
    }

    // Create output
    var out_shape = [_]i64{ @intCast(batch), @intCast(dim), @intCast(seqlen) };
    var out_val: ?*c.OrtValue = null;
    _ = api.KernelContext_GetOutput.?(ctx, 0, &out_shape, 3, &out_val);
    var out_data: ?*anyopaque = null;
    _ = api.GetTensorMutableData.?(out_val, &out_data);

    // Cast pointers
    const u_ptr: [*]const f32 = @ptrCast(@alignCast(u_data));
    const delta_ptr: [*]const f32 = @ptrCast(@alignCast(delta_data));
    const A_ptr: [*]const f32 = @ptrCast(@alignCast(A_data));
    const B_ptr: [*]const f32 = @ptrCast(@alignCast(B_data));
    const C_ptr: [*]const f32 = @ptrCast(@alignCast(C_data));
    const D_ptr: [*]const f32 = @ptrCast(@alignCast(D_data));
    const out_ptr: [*]f32 = @ptrCast(@alignCast(out_data));

    const z_ptr: ?[*]const f32 = if (z_data) |z| @ptrCast(@alignCast(z)) else null;

    // Run kernel
    if (state.use_fused and state.use_exact) {
        selectiveScanKernel(true, true, batch, dim, seqlen, u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr);
    } else if (state.use_fused) {
        selectiveScanKernel(true, false, batch, dim, seqlen, u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr);
    } else if (state.use_exact) {
        selectiveScanKernel(false, true, batch, dim, seqlen, u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, null, out_ptr);
    } else {
        selectiveScanKernel(false, false, batch, dim, seqlen, u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, null, out_ptr);
    }
}

fn destroyKernel(kernel: ?*anyopaque) callconv(.c) void {
    if (kernel) |k| {
        const state: *KernelState = @ptrCast(@alignCast(k));
        std.heap.c_allocator.destroy(state);
    }
}

// Op name functions
fn nameSS(_: ?*const c.OrtCustomOp) callconv(.c) [*c]const u8 {
    return "SelectiveScan";
}
fn nameSSE(_: ?*const c.OrtCustomOp) callconv(.c) [*c]const u8 {
    return "SelectiveScanExact";
}
fn nameSSF(_: ?*const c.OrtCustomOp) callconv(.c) [*c]const u8 {
    return "SelectiveScanFused";
}
fn nameSSFE(_: ?*const c.OrtCustomOp) callconv(.c) [*c]const u8 {
    return "SelectiveScanFusedExact";
}

fn getExecutionProvider(_: ?*const c.OrtCustomOp) callconv(.c) [*c]const u8 {
    return "CPUExecutionProvider";
}
fn getInputType(_: ?*const c.OrtCustomOp, _: usize) callconv(.c) c.ONNXTensorElementDataType {
    return c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
fn getInputCount6(_: ?*const c.OrtCustomOp) callconv(.c) usize {
    return 6;
}
fn getInputCount7(_: ?*const c.OrtCustomOp) callconv(.c) usize {
    return 7;
}
fn getOutputCount(_: ?*const c.OrtCustomOp) callconv(.c) usize {
    return 1;
}
fn getOutputType(_: ?*const c.OrtCustomOp, _: usize) callconv(.c) c.ONNXTensorElementDataType {
    return c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
fn getInputCharacteristic(_: ?*const c.OrtCustomOp, _: usize) callconv(.c) c.OrtCustomOpInputOutputCharacteristic {
    return c.INPUT_OUTPUT_REQUIRED;
}
fn getOutputCharacteristic(_: ?*const c.OrtCustomOp, _: usize) callconv(.c) c.OrtCustomOpInputOutputCharacteristic {
    return c.INPUT_OUTPUT_REQUIRED;
}
fn getInputMemoryType(_: ?*const c.OrtCustomOp, _: usize) callconv(.c) c.OrtMemType {
    return c.OrtMemTypeDefault;
}
fn getStartVersion(_: ?*const c.OrtCustomOp) callconv(.c) c_int {
    return 1;
}
fn getEndVersion(_: ?*const c.OrtCustomOp) callconv(.c) c_int {
    return std.math.maxInt(c_int);
}

// Global op definitions
var op_ss: c.OrtCustomOp = undefined;
var op_sse: c.OrtCustomOp = undefined;
var op_ssf: c.OrtCustomOp = undefined;
var op_ssfe: c.OrtCustomOp = undefined;

fn initOp(
    op: *c.OrtCustomOp,
    create_fn: *const fn (?*const c.OrtCustomOp, [*c]const c.OrtApi, ?*const c.OrtKernelInfo) callconv(.c) ?*anyopaque,
    name_fn: *const fn (?*const c.OrtCustomOp) callconv(.c) [*c]const u8,
    input_count_fn: *const fn (?*const c.OrtCustomOp) callconv(.c) usize,
) void {
    op.* = std.mem.zeroes(c.OrtCustomOp);
    op.version = c.ORT_API_VERSION;
    op.CreateKernel = create_fn;
    op.KernelCompute = computeKernel;
    op.KernelDestroy = destroyKernel;
    op.GetName = name_fn;
    op.GetExecutionProviderType = getExecutionProvider;
    op.GetInputTypeCount = input_count_fn;
    op.GetInputType = getInputType;
    op.GetOutputTypeCount = getOutputCount;
    op.GetOutputType = getOutputType;
    op.GetInputCharacteristic = getInputCharacteristic;
    op.GetOutputCharacteristic = getOutputCharacteristic;
    op.GetInputMemoryType = getInputMemoryType;
    op.GetStartVersion = getStartVersion;
    op.GetEndVersion = getEndVersion;
}

export fn RegisterCustomOps(options: ?*c.OrtSessionOptions, api_base: ?*const c.OrtApiBase) callconv(.c) ?*c.OrtStatus {
    const base = api_base orelse return null;
    const api_ptr = base.GetApi.?(c.ORT_API_VERSION) orelse return null;
    const api: *const c.OrtApi = @ptrCast(api_ptr);

    // Initialize ops
    initOp(&op_ss, createKernelFn(false, false), nameSS, getInputCount6);
    initOp(&op_sse, createKernelFn(false, true), nameSSE, getInputCount6);
    initOp(&op_ssf, createKernelFn(true, false), nameSSF, getInputCount7);
    initOp(&op_ssfe, createKernelFn(true, true), nameSSFE, getInputCount7);

    // Create domain and register ops
    var domain: ?*c.OrtCustomOpDomain = null;
    if (api.CreateCustomOpDomain.?("mamba", &domain) != null) return null;
    if (api.CustomOpDomain_Add.?(domain, &op_ss) != null) return null;
    if (api.CustomOpDomain_Add.?(domain, &op_sse) != null) return null;
    if (api.CustomOpDomain_Add.?(domain, &op_ssf) != null) return null;
    if (api.CustomOpDomain_Add.?(domain, &op_ssfe) != null) return null;
    if (api.AddCustomOpDomain.?(options, domain) != null) return null;

    return null;
}
