//! Smoke tests for ForkJoin parallel dispatch
//!
//! Run: zig test tests/test_forkjoin.zig -OReleaseFast

const std = @import("std");

const MAX_WORKERS = 31;

const ForkJoin = struct {
    threads: [MAX_WORKERS]std.Thread = undefined,
    num_workers: u32 = 0,
    work_fn: WorkFn = undefined,
    work_ctx: *anyopaque = undefined,
    total_work: usize = 0,
    next_idx: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    gen: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    done: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    ready: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    stop: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    const WorkFn = *const fn (start: usize, end: usize, ctx: *anyopaque) void;

    fn init(self: *ForkJoin, num_threads: u32) void {
        self.num_workers = if (num_threads > 1) num_threads - 1 else 0;
        if (self.num_workers > MAX_WORKERS) self.num_workers = MAX_WORKERS;

        for (0..self.num_workers) |i| {
            self.threads[i] = std.Thread.spawn(.{}, workerLoop, .{ self, i }) catch {
                self.num_workers = @intCast(i);
                break;
            };
        }

        // Wait for all workers to reach their initial park point
        while (self.ready.load(.acquire) < self.num_workers) {
            std.atomic.spinLoopHint();
        }
    }

    fn deinit(self: *ForkJoin) void {
        self.stop.store(1, .release);
        _ = self.gen.fetchAdd(1, .release);
        std.Thread.Futex.wake(&self.gen, MAX_WORKERS);
        for (0..self.num_workers) |i| {
            self.threads[i].join();
        }
    }

    fn dispatch(self: *ForkJoin, total: usize, work_fn: WorkFn, ctx: *anyopaque) void {
        if (total == 0) return;
        if (self.num_workers == 0) {
            work_fn(0, total, ctx);
            return;
        }
        const num_participants = self.num_workers + 1;

        self.work_fn = work_fn;
        self.work_ctx = ctx;
        self.total_work = total;
        self.next_idx.store(0, .monotonic);
        self.done.store(0, .monotonic);

        _ = self.gen.fetchAdd(1, .release);
        std.Thread.Futex.wake(&self.gen, MAX_WORKERS);

        self.doWork(work_fn, ctx, total);

        const prev_done = self.done.fetchAdd(1, .release);
        if (prev_done + 1 < num_participants) {
            while (true) {
                const current_done = self.done.load(.acquire);
                if (current_done >= num_participants) break;
                std.Thread.Futex.wait(&self.done, current_done);
            }
        }
    }

    fn doWork(self: *ForkJoin, work_fn: WorkFn, ctx: *anyopaque, total: usize) void {
        while (true) {
            const idx = self.next_idx.fetchAdd(1, .monotonic);
            if (idx >= total) break;
            work_fn(idx, idx + 1, ctx);
        }
    }

    fn workerLoop(self: *ForkJoin, _: usize) void {
        var my_gen: u32 = self.gen.load(.acquire);
        _ = self.ready.fetchAdd(1, .release);
        while (true) {
            std.Thread.Futex.wait(&self.gen, my_gen);
            const new_gen = self.gen.load(.acquire);
            if (new_gen == my_gen) continue;
            my_gen = new_gen;

            if (self.stop.load(.acquire) != 0) return;

            const work_fn = self.work_fn;
            const ctx = self.work_ctx;
            const total = self.total_work;

            self.doWork(work_fn, ctx, total);

            const num_participants = self.num_workers + 1;
            const prev_done = self.done.fetchAdd(1, .release);
            if (prev_done + 1 >= num_participants) {
                std.Thread.Futex.wake(&self.done, 1);
            }
        }
    }
};

// =============================================================================
// Test 1: 100 rapid dispatches of 256 items, verify exact counts
// =============================================================================

test "rapid dispatches" {
    var fj: ForkJoin = .{};
    fj.init(4);
    defer fj.deinit();

    const N = 256;
    const ITERS = 100;

    var slots: [N]std.atomic.Value(u32) = undefined;
    for (&slots) |*s| s.* = std.atomic.Value(u32).init(0);

    const Ctx = struct {
        s: *[N]std.atomic.Value(u32),
        fn work(start: usize, end: usize, ctx_ptr: *anyopaque) void {
            const self: *const @This() = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |i| {
                _ = self.s[i].fetchAdd(1, .monotonic);
            }
        }
    };
    var ctx = Ctx{ .s = &slots };

    for (0..ITERS) |_| {
        fj.dispatch(N, Ctx.work, @ptrCast(&ctx));
    }

    for (slots, 0..) |s, i| {
        const val = s.load(.monotonic);
        if (val != ITERS) {
            std.debug.print("slot {} = {} (expected {})\n", .{ i, val, ITERS });
            return error.SlotMismatch;
        }
    }
}

// =============================================================================
// Test 2: Dispatch sizes 0, 1, and a larger value all produce correct counts
// =============================================================================

test "varying sizes" {
    var fj: ForkJoin = .{};
    fj.init(4);
    defer fj.deinit();

    const Ctx = struct {
        c: *std.atomic.Value(u32),
        fn work(_: usize, _: usize, ctx_ptr: *anyopaque) void {
            const self: *const @This() = @ptrCast(@alignCast(ctx_ptr));
            _ = self.c.fetchAdd(1, .monotonic);
        }
    };

    const sizes = [_]usize{ 0, 1, 3, 4, 5, 64 };
    for (sizes) |size| {
        var counter = std.atomic.Value(u32).init(0);
        var ctx = Ctx{ .c = &counter };
        fj.dispatch(size, Ctx.work, @ptrCast(&ctx));
        const val = counter.load(.monotonic);
        if (val != @as(u32, @intCast(size))) {
            std.debug.print("size {} => counter {} (expected {})\n", .{ size, val, size });
            return error.CountMismatch;
        }
    }
}

// =============================================================================
// Test 3: Uneven work — slot 0 spins, verify all slots visited exactly once
// =============================================================================

test "uneven work" {
    var fj: ForkJoin = .{};
    fj.init(4);
    defer fj.deinit();

    const N = 32;
    var visited: [N]std.atomic.Value(u32) = undefined;
    for (&visited) |*v| v.* = std.atomic.Value(u32).init(0);

    const Ctx = struct {
        v: *[N]std.atomic.Value(u32),
        fn work(start: usize, end: usize, ctx_ptr: *anyopaque) void {
            const self: *const @This() = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |idx| {
                if (idx == 0) {
                    var i: u32 = 0;
                    while (i < 100_000) : (i += 1) {
                        std.atomic.spinLoopHint();
                    }
                }
                _ = self.v[idx].fetchAdd(1, .monotonic);
            }
        }
    };
    var ctx = Ctx{ .v = &visited };

    fj.dispatch(N, Ctx.work, @ptrCast(&ctx));

    for (visited, 0..) |v, i| {
        const val = v.load(.monotonic);
        if (val != 1) {
            std.debug.print("visited[{}] = {} (expected 1)\n", .{ i, val });
            return error.VisitMismatch;
        }
    }
}
