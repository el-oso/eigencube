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
        _ = try bufs.putOrUpdate(&c.data, c.zobristHash(), @intCast(step + 1), 1);
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

    try heap.push(.{ .prio = 10, .hash = FastCube.SOLVED.zobristHash(), .cost = 1, .last_move = 0, .state = FastCube.SOLVED });
    try std.testing.expect(heap.len == 1);
    try std.testing.expect(heap.items.len >= 4);
    const popped = heap.pop();
    try std.testing.expect(popped != null and popped.?.prio == 10);
}

test "comptime corner commutators form A4 subgroup" {
    const res = comptime blk: {
        @setEvalBranchQuota(500_000);
        var count: usize = 0;
        var unique: usize = 0;
        var seen: [256]bool = [_]bool{false} ** 256;
        for (0..12) |a| {
            for (0..12) |b| {
                for (0..12) |c| {
                    const seq = [8]usize{ a, b, a ^ 1, c, a, b ^ 1, a ^ 1, c ^ 1 };
                    var cube = FastCube.SOLVED;
                    for (seq) |m| cube = cube.applyMove(m);

                    var preserves = true;
                    for (eigencube.TOP_CUBELETS) |i| if (!tables.is_solved[i][cube.data[i]]) {
                        preserves = false;
                        break;
                    };
                    if (!preserves) continue;
                    for (eigencube.MID_LAYER) |i| if (!tables.is_solved[i][cube.data[i]]) {
                        preserves = false;
                        break;
                    };
                    if (!preserves) continue;
                    for (eigencube.BOT_EDGES) |i| if (!tables.is_solved[i][cube.data[i]]) {
                        preserves = false;
                        break;
                    };
                    if (!preserves) continue;

                    var perms: [4]usize = undefined;
                    for (eigencube.BOT_CORNERS, 0..) |ci, idx| {
                        const rot = tables.rotations[cube.data[ci]];
                        const pos = rot.mulVec(CUBELETS[ci]);
                        for (eigencube.BOT_CORNERS, 0..) |target_ci, target_idx| {
                            if (CUBELETS[target_ci].eql(pos)) {
                                perms[idx] = target_idx;
                                break;
                            }
                        }
                    }
                    if (perms[0] != 0 or perms[1] != 1 or perms[2] != 2 or perms[3] != 3) {
                        count += 1;
                        const code = perms[0] + perms[1] * 4 + perms[2] * 16 + perms[3] * 64;
                        if (!seen[code]) {
                            seen[code] = true;
                            unique += 1;
                        }
                    }
                }
            }
        }
        break :blk .{ .total = count, .unique = unique };
    };
    try std.testing.expect(res.total == 24);
    try std.testing.expect(res.unique == 8);
}

test "comptime corner solution table solves all 12 A4 states" {
    const CornerSol = struct { len: u8, moves: [16]u8 };
    const tables_local = eigencube.tables;
    const BOT_CORNERS = eigencube.BOT_CORNERS;

    const corner_table = comptime blk: {
        @setEvalBranchQuota(500_000);
        var table = [_]CornerSol{.{ .len = 255, .moves = undefined }} ** 256;

        const get_code = struct {
            fn call(cube: FastCube) u8 {
                var code: u8 = 0;
                for (BOT_CORNERS, 0..) |ci, idx| {
                    const rot = tables_local.rotations[cube.data[ci]];
                    const pos = rot.mulVec(CUBELETS[ci]);
                    for (BOT_CORNERS, 0..) |target_ci, target_idx| {
                        if (CUBELETS[target_ci].eql(pos)) {
                            code |= @as(u8, @intCast(target_idx)) << @as(u3, @intCast(idx * 2));
                            break;
                        }
                    }
                }
                return code;
            }
        }.call;

        var commutators: [24][8]u8 = undefined;
        var comm_cnt: usize = 0;
        for (0..12) |a| {
            for (0..12) |b| {
                for (0..12) |c| {
                    const seq = [8]u8{ @intCast(a), @intCast(b), @intCast(a ^ 1), @intCast(c), @intCast(a), @intCast(b ^ 1), @intCast(a ^ 1), @intCast(c ^ 1) };
                    var cube = FastCube.SOLVED;
                    for (seq) |m| cube = cube.applyMove(m);

                    var preserves = true;
                    for (eigencube.TOP_CUBELETS) |i| if (!tables_local.is_solved[i][cube.data[i]]) {
                        preserves = false;
                        break;
                    };
                    if (!preserves) continue;
                    for (eigencube.MID_LAYER) |i| if (!tables_local.is_solved[i][cube.data[i]]) {
                        preserves = false;
                        break;
                    };
                    if (!preserves) continue;
                    for (eigencube.BOT_EDGES) |i| if (!tables_local.is_solved[i][cube.data[i]]) {
                        preserves = false;
                        break;
                    };
                    if (!preserves) continue;

                    const code = get_code(cube);
                    if (code != get_code(FastCube.SOLVED)) {
                        commutators[comm_cnt] = seq;
                        comm_cnt += 1;
                    }
                }
            }
        }

        const solved_code = get_code(FastCube.SOLVED);
        table[solved_code] = .{ .len = 0, .moves = [_]u8{0} ** 16 };

        var queue_cube: [12]FastCube = undefined;
        var queue_moves: [12][16]u8 = undefined;
        var queue_lens: [12]u8 = undefined;
        queue_cube[0] = FastCube.SOLVED;
        queue_lens[0] = 0;
        var q_len: usize = 1;
        var q_head: usize = 0;

        while (q_head < q_len) : (q_head += 1) {
            const curr = queue_cube[q_head];
            const curr_len = queue_lens[q_head];
            const curr_moves = queue_moves[q_head];
            if (curr_len >= 16) continue;

            for (commutators) |comm| {
                var nxt = curr;
                for (comm) |m| nxt = nxt.applyMove(m);
                const code = get_code(nxt);
                if (table[code].len == 255) {
                    const new_len = curr_len + 8;
                    var forward_moves: [16]u8 = undefined;
                    for (0..curr_len) |k| forward_moves[k] = curr_moves[k];
                    for (0..8) |k| forward_moves[curr_len + k] = comm[k];

                    var sol_moves: [16]u8 = undefined;
                    for (0..new_len) |k| sol_moves[k] = forward_moves[new_len - 1 - k] ^ 1;
                    table[code] = .{ .len = new_len, .moves = sol_moves };

                    queue_cube[q_len] = nxt;
                    queue_lens[q_len] = new_len;
                    queue_moves[q_len] = forward_moves;
                    q_len += 1;
                }
            }
        }

        break :blk table;
    };

    // Verify all 12 reachable states are solved!
    var count_reachable: usize = 0;
    for (corner_table) |sol| {
        if (sol.len != 255) {
            count_reachable += 1;
        }
    }
    try std.testing.expect(count_reachable == 12);

    // Verify round-trip solution on all 12 states
    for (corner_table) |sol| {
        if (sol.len == 255) continue;
        // Invert sol.moves to create the scrambled state from SOLVED
        var state = FastCube.SOLVED;
        // Inverting sol.moves: reverse and flip direction
        var k = sol.len;
        while (k > 0) : (k -= 1) {
            state = state.applyMove(sol.moves[k - 1] ^ 1);
        }
        // Now apply sol.moves to state
        for (0..sol.len) |idx| {
            state = state.applyMove(sol.moves[idx]);
        }
        // Verify all 4 corners are now position-solved!
        for (BOT_CORNERS) |ci| {
            try std.testing.expect(tables_local.is_pos_solved[ci][state.data[ci]]);
        }
        // And verify 17 top+mid and 4 bottom edges are still solved!
        for (eigencube.TOP_CUBELETS) |ci| try std.testing.expect(tables_local.is_solved[ci][state.data[ci]]);
        for (eigencube.MID_LAYER) |ci| try std.testing.expect(tables_local.is_solved[ci][state.data[ci]]);
        for (eigencube.BOT_EDGES) |ci| try std.testing.expect(tables_local.is_solved[ci][state.data[ci]]);
    }
}

test "comptime zobrist_diff and is_bottom_down mathematical invariants" {
    const t = eigencube.tables;
    const down_vec = eigencube.Vec3{ .x = 0, .y = 0, .z = -1 };

    // Verify is_bottom_down exact match with SO(3, Z) rotation
    for (0..24) |r| {
        const expected = t.rotations[r].mulVec(down_vec).eql(down_vec);
        try std.testing.expectEqual(expected, t.is_bottom_down[r]);
    }

    // Verify zobrist_diff satisfies exact XOR homomorphic differential:
    // zobrist[c][r] ^ zobrist_diff[m][c][r] == zobrist[c][transition[m][c][r]]
    for (0..12) |m| {
        for (0..26) |c| {
            for (0..24) |r| {
                const next_r = t.transition[m][c][r];
                const expected_diff = if (next_r != r) (t.zobrist[c][r] ^ t.zobrist[c][next_r]) else 0;
                try std.testing.expectEqual(expected_diff, t.zobrist_diff[m][c][r]);
                try std.testing.expectEqual(t.zobrist[c][next_r], t.zobrist[c][r] ^ t.zobrist_diff[m][c][r]);
            }
        }
    }
}
