const std = @import("std");
const eigencube = @import("eigencube");

const Vec3 = eigencube.Vec3;
const Mat3 = eigencube.Mat3;
const Move = eigencube.Move;
const Cube = eigencube.Cube;
const FastCube = eigencube.FastCube;
const CUBELETS = eigencube.CUBELETS;
const UNIT_VECTORS = eigencube.UNIT_VECTORS;
const MOVES = eigencube.MOVES;
const SOLVED_CUBE = eigencube.SOLVED_CUBE;
const tables = eigencube.tables;
const isCubeletSolved = eigencube.isCubeletSolved;
const astar = eigencube.astar;
const shuffle = eigencube.shuffle;
const solve = eigencube.solve;

fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        .x = a.y * b.z - a.z * b.y,
        .y = a.z * b.x - a.x * b.z,
        .z = a.x * b.y - a.y * b.x,
    };
}

fn applyCanonicalMove(m: Move, cube: *const Cube) Cube {
    const n = m.normal;
    const r_m = m.rotMat();
    var res: Cube = undefined;
    for (cube, 0..) |item, i| {
        const c = item.pos;
        const r = item.rot;
        res[i] = .{
            .pos = c,
            .rot = if (n.dot(r.mulVec(c)) > 0) r_m.mulMat(r) else r,
        };
    }
    return res;
}

test "counts and group size" {
    try std.testing.expectEqual(@as(usize, 26), CUBELETS.len);
    try std.testing.expectEqual(@as(usize, 12), MOVES.len);
    try std.testing.expectEqual(@as(usize, 24), tables.rotations.len);
    try std.testing.expect(tables.rotations[0].eql(Mat3.id));
    for (0..26) |c_idx| {
        for (0..24) |r_idx| {
            try std.testing.expect(tables.sqrt_dist_solved[c_idx][r_idx] <= 1.733);
            try std.testing.expect(tables.sqrt_dist_pos[c_idx][r_idx] <= 1.733);
        }
    }
}

test "linear algebra invariants" {
    const c = Vec3{ .x = 1, .y = 1, .z = 1 };
    try std.testing.expect(isCubeletSolved(c, Mat3.id));
    const rot_x = MOVES[0].rotMat();
    try std.testing.expect(!isCubeletSolved(c, rot_x));
    try std.testing.expectEqual(@as(i32, 0), (Vec3{ .x = 1, .y = 0, .z = 0 }).dot(.{ .x = 0, .y = 1, .z = 0 }));
    try std.testing.expectEqual(@as(i32, 32), (Vec3{ .x = 1, .y = 2, .z = 3 }).dot(.{ .x = 4, .y = 5, .z = 6 }));
}

test "moves turn the way they are named" {
    for (MOVES) |m| {
        for (UNIT_VECTORS) |p| {
            if (p.dot(m.normal) == 0) {
                const rotated = m.rotMat().mulVec(p);
                const triple_prod = m.normal.dot(cross(p, rotated));
                try std.testing.expectEqual(-@as(i32, m.dir), triple_prod);
            }
        }
    }
}

test "opposite face turns the other way" {
    for (MOVES) |m| {
        const opp = Move{ .normal = .{ .x = -m.normal.x, .y = -m.normal.y, .z = -m.normal.z }, .dir = -m.dir };
        try std.testing.expect(m.rotMat().eql(opp.rotMat()));
    }
}

test "standard notation" {
    const Case = struct { face: Vec3, home: Vec3, after: Vec3 };
    const clockwise = [_]Case{
        .{ .face = .{ .x = 0, .y = 0, .z = 1 }, .home = .{ .x = 1, .y = 0, .z = 1 }, .after = .{ .x = 0, .y = -1, .z = 1 } },
        .{ .face = .{ .x = 0, .y = 0, .z = -1 }, .home = .{ .x = 1, .y = 0, .z = -1 }, .after = .{ .x = 0, .y = 1, .z = -1 } },
        .{ .face = .{ .x = 1, .y = 0, .z = 0 }, .home = .{ .x = 1, .y = 0, .z = 1 }, .after = .{ .x = 1, .y = 1, .z = 0 } },
        .{ .face = .{ .x = -1, .y = 0, .z = 0 }, .home = .{ .x = -1, .y = 0, .z = 1 }, .after = .{ .x = -1, .y = -1, .z = 0 } },
        .{ .face = .{ .x = 0, .y = 1, .z = 0 }, .home = .{ .x = 1, .y = 1, .z = 0 }, .after = .{ .x = 0, .y = 1, .z = 1 } },
        .{ .face = .{ .x = 0, .y = -1, .z = 0 }, .home = .{ .x = 1, .y = -1, .z = 0 }, .after = .{ .x = 0, .y = -1, .z = -1 } },
    };
    for (clockwise) |tc| {
        const res = (Move{ .normal = tc.face, .dir = 1 }).rotMat().mulVec(tc.home);
        try std.testing.expect(res.eql(tc.after));
    }
}

test "fast cube roundtrip" {
    const c0 = SOLVED_CUBE;
    const fast_cube = FastCube.fromCube(&c0);
    try std.testing.expect(std.mem.eql(u8, &fast_cube.data, &FastCube.SOLVED.data));
    const back0 = fast_cube.toCube();
    for (0..26) |i| {
        try std.testing.expect(back0[i].pos.eql(c0[i].pos));
        try std.testing.expect(back0[i].rot.eql(c0[i].rot));
    }

    var permuted = c0;
    std.mem.reverse(eigencube.Cubelet, &permuted);
    const fast_permuted = FastCube.fromCube(&permuted);
    try std.testing.expect(std.mem.eql(u8, &fast_permuted.data, &FastCube.SOLVED.data));

    const scrambled = shuffle(50, 42, c0);
    const fast_scrambled = FastCube.fromCube(&scrambled);
    const back_scrambled = fast_scrambled.toCube();
    for (0..26) |i| {
        try std.testing.expect(back_scrambled[i].pos.eql(scrambled[i].pos));
        try std.testing.expect(back_scrambled[i].rot.eql(scrambled[i].rot));
    }
}

test "4x single move identity" {
    const c0 = FastCube.SOLVED;
    for (0..12) |m| {
        const c1 = c0.applyMove(m);
        const c2 = c1.applyMove(m);
        const c3 = c2.applyMove(m);
        const c4 = c3.applyMove(m);
        try std.testing.expect(c4.isSolved());
        try std.testing.expect(!c1.isSolved());
        try std.testing.expect(!c2.isSolved());
        try std.testing.expect(!c3.isSolved());
    }
}

test "inverse move cancellation" {
    const c0 = FastCube.SOLVED;
    for (0..12) |m| {
        const inv_m = m ^ 1;
        const c_after = c0.applyMove(m).applyMove(inv_m);
        try std.testing.expect(c_after.isSolved());
    }
}

test "6x sexy move identity" {
    var r_idx: usize = 0;
    var u_idx: usize = 0;
    for (MOVES, 0..) |m, i| {
        if (m.normal.eql(.{ .x = 0, .y = 1, .z = 0 }) and m.dir == 1) r_idx = i;
        if (m.normal.eql(.{ .x = 0, .y = 0, .z = 1 }) and m.dir == 1) u_idx = i;
    }
    const sexy = [_]usize{ r_idx, u_idx, r_idx ^ 1, u_idx ^ 1 };

    var c = FastCube.SOLVED;
    for (0..6) |_| {
        for (sexy) |m| c = c.applyMove(m);
    }
    try std.testing.expect(c.isSolved());
}

test "deterministic budget exhaustion" {
    const c0 = FastCube.SOLVED;
    const Dummy = struct {
        fn goal(_: void, _: *const FastCube) bool {
            return false;
        }
        fn h(_: void, _: *const FastCube) f64 {
            return 0.0;
        }
    };
    var bufs = try eigencube.SearchBuffers.init(std.testing.allocator, 1024, 1024);
    defer bufs.deinit();
    const res = try astar(std.testing.allocator, &bufs, c0, {}, Dummy.goal, Dummy.h, 0.0, 10);
    try std.testing.expect(res == null);
}

test "full scramble solve" {
    const scrambled = shuffle(100, 42, SOLVED_CUBE);
    const moves = try solve(std.testing.allocator, scrambled);
    defer std.testing.allocator.free(moves);

    var curr = FastCube.fromCube(&scrambled);
    for (moves) |m| {
        var m_idx: usize = 0;
        for (MOVES, 0..) |cand, i| {
            if (cand.normal.eql(m.normal) and cand.dir == m.dir) {
                m_idx = i;
                break;
            }
        }
        curr = curr.applyMove(m_idx);
    }
    try std.testing.expect(curr.isSolved());
}

test "canonical cube solve interoperability" {
    const scrambled = shuffle(100, 42, SOLVED_CUBE);
    const moves = try solve(std.testing.allocator, scrambled);
    defer std.testing.allocator.free(moves);

    var canonical = scrambled;
    for (moves) |m| {
        canonical = applyCanonicalMove(m, &canonical);
    }
    for (canonical) |item| {
        try std.testing.expect(isCubeletSolved(item.pos, item.rot));
    }
}

test "successive solves independent" {
    for ([_]u64{ 1, 2 }) |seed| {
        const scrambled = shuffle(50, seed, SOLVED_CUBE);
        const moves = try solve(std.testing.allocator, scrambled);
        defer std.testing.allocator.free(moves);

        var curr = FastCube.fromCube(&scrambled);
        for (moves) |m| {
            var m_idx: usize = 0;
            for (MOVES, 0..) |cand, i| {
                if (cand.normal.eql(m.normal) and cand.dir == m.dir) {
                    m_idx = i;
                    break;
                }
            }
            curr = curr.applyMove(m_idx);
        }
        try std.testing.expect(curr.isSolved());
    }
}

test "search buffer dynamic growth" {
    var bufs = try eigencube.SearchBuffers.init(std.testing.allocator, 4, 4);
    defer bufs.deinit();

    // Insert multiple distinct states to trigger dynamic growth
    for (0..20) |step| {
        var c = FastCube.SOLVED;
        c.data[0] = @intCast(step);
        _ = try bufs.putOrUpdate(&c.data, @intCast(step + 1), 1);
    }
    try std.testing.expect(bufs.visited.len >= 32);
    try std.testing.expect(bufs.count == 20);
}

test "full scramble solve with budget restarts" {
    // Seed 0 specifically triggers budget expansion beyond 225k simulated moves
    const scrambled = shuffle(100, 0, SOLVED_CUBE);
    const moves = try solve(std.testing.allocator, scrambled);
    defer std.testing.allocator.free(moves);

    var curr = FastCube.fromCube(&scrambled);
    for (moves) |m| {
        var m_idx: usize = 0;
        for (MOVES, 0..) |cand, i| {
            if (cand.normal.eql(m.normal) and cand.dir == m.dir) {
                m_idx = i;
                break;
            }
        }
        curr = curr.applyMove(m_idx);
    }
    try std.testing.expect(curr.isSolved());
}

test "search buffer init allocation failure cleanup" {
    for ([_]usize{ 0, 1, 2 }) |fail_idx| {
        var failing_alloc = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_idx });
        try std.testing.expectError(error.OutOfMemory, eigencube.SearchBuffers.init(failing_alloc.allocator(), 16, 16));
    }
}

test "search buffer zero capacity init" {
    var bufs = try eigencube.SearchBuffers.init(std.testing.allocator, 0, 0);
    defer bufs.deinit();
    try std.testing.expect(bufs.visited.len >= 4);
    try std.testing.expect(bufs.frontier.len >= 4);
}

test "min heap push with zero initial capacity" {
    const empty_slice: []eigencube.HeapNode = &[_]eigencube.HeapNode{};
    var heap = eigencube.MinHeap{ .items = empty_slice, .len = 0, .allocator = std.testing.allocator };
    defer std.testing.allocator.free(heap.items);

    try heap.push(.{ .prio = 10, .cost = 1, .last_move = 0, .state = FastCube.SOLVED });
    try std.testing.expect(heap.len == 1);
    try std.testing.expect(heap.items.len >= 4);
    const popped = heap.pop();
    try std.testing.expect(popped != null and popped.?.prio == 10);
}
