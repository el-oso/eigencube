const std = @import("std");

pub var total_moves_simulated = std.atomic.Value(usize).init(0);

// -- Discrete 3D Linear Algebra: SO(3, ℤ) rotations & Euclidean vectors -------

pub const Vec3 = struct {
    x: i8,
    y: i8,
    z: i8,

    pub fn dot(self: Vec3, o: Vec3) i32 {
        return @as(i32, self.x) * @as(i32, o.x) + @as(i32, self.y) * @as(i32, o.y) + @as(i32, self.z) * @as(i32, o.z);
    }
    pub fn eql(self: Vec3, o: Vec3) bool {
        return self.x == o.x and self.y == o.y and self.z == o.z;
    }
    pub fn gt(self: Vec3, o: Vec3) bool {
        if (self.x != o.x) return self.x > o.x;
        if (self.y != o.y) return self.y > o.y;
        return self.z > o.z;
    }
};

pub const Mat3 = struct {
    row0: Vec3,
    row1: Vec3,
    row2: Vec3,

    pub const id = Mat3{ .row0 = .{ .x = 1, .y = 0, .z = 0 }, .row1 = .{ .x = 0, .y = 1, .z = 0 }, .row2 = .{ .x = 0, .y = 0, .z = 1 } };

    pub fn diag(v: Vec3) Mat3 {
        return .{ .row0 = .{ .x = v.x, .y = 0, .z = 0 }, .row1 = .{ .x = 0, .y = v.y, .z = 0 }, .row2 = .{ .x = 0, .y = 0, .z = v.z } };
    }
    pub fn transpose(self: Mat3) Mat3 {
        return .{ .row0 = .{ .x = self.row0.x, .y = self.row1.x, .z = self.row2.x }, .row1 = .{ .x = self.row0.y, .y = self.row1.y, .z = self.row2.y }, .row2 = .{ .x = self.row0.z, .y = self.row1.z, .z = self.row2.z } };
    }
    pub fn mulVec(self: Mat3, v: Vec3) Vec3 {
        return .{ .x = @intCast(self.row0.dot(v)), .y = @intCast(self.row1.dot(v)), .z = @intCast(self.row2.dot(v)) };
    }
    pub fn mulMat(self: Mat3, o: Mat3) Mat3 {
        const t = o.transpose();
        return (Mat3{ .row0 = self.mulVec(t.row0), .row1 = self.mulVec(t.row1), .row2 = self.mulVec(t.row2) }).transpose();
    }
    pub fn eql(self: Mat3, o: Mat3) bool {
        return self.row0.eql(o.row0) and self.row1.eql(o.row1) and self.row2.eql(o.row2);
    }
};

pub const Move = struct {
    normal: Vec3,
    dir: i8,

    /// Exact closed-form Rodrigues rotation formula for 90-degree turns of unit axis v:
    /// R = v vᵀ - d [v]ₓ, where [v]ₓ is the cross-product skew-symmetric matrix.
    /// https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    pub fn rotMat(self: Move) Mat3 {
        const x = self.normal.x;
        const y = self.normal.y;
        const z = self.normal.z;
        const d = self.dir;
        return .{ .row0 = .{ .x = x * x, .y = d * z, .z = -d * y }, .row1 = .{ .x = -d * z, .y = y * y, .z = d * x }, .row2 = .{ .x = d * y, .y = -d * x, .z = z * z } };
    }
};

pub const Cubelet = struct { pos: Vec3, rot: Mat3 };
pub const Cube = [26]Cubelet;

pub const CUBELETS: [26]Vec3 = init_c: {
    var c: [26]Vec3 = undefined;
    for (0..26) |i| {
        const j: usize = if (i < 13) i else i + 1;
        c[i] = .{ .x = @intCast(@as(i32, @intCast(j / 9)) - 1), .y = @intCast(@as(i32, @intCast((j / 3) % 3)) - 1), .z = @intCast(@as(i32, @intCast(j % 3)) - 1) };
    }
    break :init_c c;
};

pub const UNIT_VECTORS = [6]Vec3{ .{ .x = -1, .y = 0, .z = 0 }, .{ .x = 0, .y = -1, .z = 0 }, .{ .x = 0, .y = 0, .z = -1 }, .{ .x = 0, .y = 0, .z = 1 }, .{ .x = 0, .y = 1, .z = 0 }, .{ .x = 1, .y = 0, .z = 0 } };

pub const MOVES: [12]Move = init_m: {
    var moves: [12]Move = undefined;
    for (0..12) |i| moves[i] = .{ .normal = UNIT_VECTORS[i / 2], .dir = if (i % 2 == 0) -1 else 1 };
    break :init_m moves;
};

pub const SOLVED_CUBE: Cube = init_s: {
    var c: Cube = undefined;
    for (0..26) |i| c[i] = .{ .pos = CUBELETS[i], .rot = Mat3.id };
    break :init_s c;
};

pub const TOP_EDGES = [_]usize{ 5, 11, 16, 22 };
pub const TOP_CUBELETS = [_]usize{ 2, 5, 8, 11, 13, 16, 19, 22, 25 };
pub const MID_CUBELETS = [_]usize{ 1, 2, 4, 5, 7, 8, 10, 11, 13, 15, 16, 18, 19, 21, 22, 24, 25 };
pub const MID_LAYER = [_]usize{ 1, 4, 7, 10, 15, 18, 21, 24 };
pub const BOT_EDGES = [_]usize{ 3, 9, 14, 20 };
pub const BOT_CORNERS = [_]usize{ 0, 6, 17, 23 };
pub const NON_BOT_CORNERS = [_]usize{ 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25 };

pub fn isCubeletSolved(c: Vec3, r: Mat3) bool {
    return (r.mulMat(Mat3.diag(c))).eql(Mat3.diag(c));
}
pub fn isCubeletPosSolved(c: Vec3, r: Mat3) bool {
    return (r.mulVec(c)).eql(c);
}

// -- Comptime Cayley Group Precomputation: L1-resident action & distance tables -

fn computeDistanceTable(
    rots: *const [24]Mat3,
    trans: *const [12][24]u8,
    comptime goal: fn (Vec3, Mat3) bool,
    out_d: *[26][24]f32,
    out_s: *[26][24]bool,
) void {
    const sqrt_vals = [4]f32{ 0.0, 1.0, 1.41421356, 1.7320508 };
    for (CUBELETS, 0..) |pos, c| {
        var dist: [24]u8 = undefined;
        @memset(&dist, 255);
        var q: [24]usize = undefined;
        var q_len: usize = 0;
        var q_head: usize = 0;
        for (rots, 0..) |rot, r| if (goal(pos, rot)) {
            dist[r] = 0;
            q[q_len] = r;
            q_len += 1;
        };
        while (q_head < q_len) : (q_head += 1) {
            const curr = q[q_head];
            for (trans) |trans_m| {
                const nxt = trans_m[curr];
                if (dist[nxt] == 255) {
                    dist[nxt] = dist[curr] + 1;
                    q[q_len] = nxt;
                    q_len += 1;
                }
            }
        }
        for (0..24) |r| {
            out_d[c][r] = sqrt_vals[dist[r]];
            out_s[c][r] = (dist[r] == 0);
        }
    }
}

pub const Tables = struct {
    rotations: [24]Mat3,
    transition: [12][26][24]u8,
    sqrt_dist_solved: [26][24]f32,
    sqrt_dist_pos: [26][24]f32,
    is_solved: [26][24]bool,
    is_pos_solved: [26][24]bool,
    prune_move: [13][12]bool,

    pub fn init() Tables {
        @setEvalBranchQuota(200_000);
        var rotations: [24]Mat3 = undefined;
        rotations[0] = Mat3.id;
        var rot_len: usize = 1;
        var i: usize = 0;
        while (i < rot_len) : (i += 1) {
            for (MOVES) |m| {
                const nxt = m.rotMat().mulMat(rotations[i]);
                var found = false;
                for (rotations[0..rot_len]) |r| if (r.eql(nxt)) {
                    found = true;
                    break;
                };
                if (!found and rot_len < 24) {
                    rotations[rot_len] = nxt;
                    rot_len += 1;
                }
            }
        }

        var transition: [12][26][24]u8 = undefined;
        var rot_trans: [12][24]u8 = undefined;
        for (MOVES, 0..) |mv, m| {
            const n = mv.normal;
            const r_m = mv.rotMat();
            for (rotations, 0..) |r, r_idx| {
                const prod = r_m.mulMat(r);
                var next_r: u8 = 0;
                for (rotations, 0..) |cand, cand_idx| if (cand.eql(prod)) {
                    next_r = @intCast(cand_idx);
                    break;
                };
                rot_trans[m][r_idx] = next_r;
                for (CUBELETS, 0..) |c, c_idx| transition[m][c_idx][r_idx] = if (n.dot(r.mulVec(c)) > 0) next_r else @intCast(r_idx);
            }
        }

        var sqrt_dist_solved: [26][24]f32 = undefined;
        var is_solved: [26][24]bool = undefined;
        var sqrt_dist_pos: [26][24]f32 = undefined;
        var is_pos_solved: [26][24]bool = undefined;

        computeDistanceTable(&rotations, &rot_trans, isCubeletSolved, &sqrt_dist_solved, &is_solved);
        computeDistanceTable(&rotations, &rot_trans, isCubeletPosSolved, &sqrt_dist_pos, &is_pos_solved);

        var prune_move: [13][12]bool = undefined;
        for (&prune_move) |*row| @memset(row, false);
        for (1..13) |last| {
            for (0..12) |next| {
                const p = MOVES[last - 1];
                const m = MOVES[next];
                prune_move[last][next] = (p.normal.eql(m.normal) and p.dir == -m.dir) or (p.normal.dot(m.normal) == -1 and p.normal.gt(m.normal));
            }
        }

        return .{ .rotations = rotations, .transition = transition, .sqrt_dist_solved = sqrt_dist_solved, .sqrt_dist_pos = sqrt_dist_pos, .is_solved = is_solved, .is_pos_solved = is_pos_solved, .prune_move = prune_move };
    }
};

pub const tables = Tables.init();

// -- Flat 26-Byte State Representation & Heuristics ---------------------------

pub const FastCube = struct {
    data: [26]u8,

    pub const SOLVED = init_s: {
        var res: [26]u8 = undefined;
        @memset(&res, 0);
        break :init_s FastCube{ .data = res };
    };

    pub fn fromCube(cube: *const Cube) FastCube {
        var res: [26]u8 = undefined;
        for (CUBELETS, 0..) |c, i| {
            var r = Mat3.id;
            for (cube) |cubelet| if (cubelet.pos.eql(c)) {
                r = cubelet.rot;
                break;
            };
            var idx: u8 = 0;
            for (tables.rotations, 0..) |rot, rot_idx| if (rot.eql(r)) {
                idx = @intCast(rot_idx);
                break;
            };
            res[i] = idx;
        }
        return .{ .data = res };
    }

    pub fn toCube(self: FastCube) Cube {
        var cube: Cube = undefined;
        for (0..26) |i| cube[i] = .{ .pos = CUBELETS[i], .rot = tables.rotations[self.data[i]] };
        return cube;
    }

    pub inline fn applyMove(self: FastCube, m: usize) FastCube {
        var res: [26]u8 = undefined;
        for (0..26) |i| res[i] = tables.transition[m][i][self.data[i]];
        return .{ .data = res };
    }

    pub inline fn step(self: *FastCube, m: usize) usize {
        self.* = self.applyMove(m);
        return m;
    }

    pub inline fn isSolved(self: FastCube) bool {
        for (0..26) |i| if (!tables.is_solved[i][self.data[i]]) return false;
        return true;
    }

    pub inline fn sumDist(self: FastCube, indices: []const usize, dist: *const [26][24]f32) f32 {
        var s: f32 = 0;
        for (indices) |i| s += dist[i][self.data[i]];
        return s;
    }

    pub inline fn layerH(self: FastCube, indices: []const usize, dist: *const [26][24]f32, div: f32) f64 {
        const s = self.sumDist(indices, dist);
        return @as(f64, @floatCast((s * s) / div));
    }
};

pub fn log(comptime fmt: []const u8, args: anytype) void {
    const s: u64 = @intCast(@max(0, std.time.timestamp()));
    std.debug.print("[{d:0>2}:{d:0>2}:{d:0>2}] " ++ fmt ++ "\n", .{ (s / 3600) % 24, (s / 60) % 60, s % 60 } ++ args);
}

pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        return .{ .state = if (seed == 0) 1 else seed };
    }

    pub fn nextF64(self: *Rng) f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        return @as(f64, @floatFromInt(self.state >> 11)) * (1.0 / @as(f64, @floatFromInt(@as(u64, 1) << 53)));
    }

    pub fn randomGauss(self: *Rng, mean: f64, stdev: f64) f64 {
        var sum: f64 = 0;
        for (0..12) |_| sum += self.nextF64();
        return mean + stdev * (sum - 6.0);
    }
};

// -- Pruned A* Search with Stochastic Restarts --------------------------------

pub const VisitedEntry = struct { cube: [26]u8, cost: u16, move_code: u8 };
pub const HeapNode = struct { prio: f64, cost: u16, last_move: u8, state: FastCube };

pub const SearchBuffers = struct {
    ctrl: []u8,
    visited: []VisitedEntry,
    frontier: []HeapNode,
    mask: usize,
    count: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, v_cap: usize, f_cap: usize) !SearchBuffers {
        const cap = try std.math.ceilPowerOfTwo(usize, @max(v_cap, 4));
        const ctrl = try allocator.alloc(u8, cap);
        errdefer allocator.free(ctrl);
        @memset(ctrl, 0);
        const v = try allocator.alloc(VisitedEntry, cap);
        errdefer allocator.free(v);
        const frontier = try allocator.alloc(HeapNode, @max(f_cap, 4));
        return .{
            .ctrl = ctrl,
            .visited = v,
            .frontier = frontier,
            .mask = cap - 1,
            .count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SearchBuffers) void {
        self.allocator.free(self.ctrl);
        self.allocator.free(self.visited);
        self.allocator.free(self.frontier);
    }

    pub fn clear(self: *SearchBuffers) void {
        @memset(self.ctrl, 0);
        self.count = 0;
    }

    inline fn hash(cube: *const [26]u8) u64 {
        var h: u64 = 0;
        inline for (0..3) |k| {
            h = std.math.rotl(u64, h, 5) ^ std.mem.readInt(u64, cube[k * 8 .. k * 8 + 8], .little);
            h = h *% 0x517cc1b727220a95;
        }
        return (std.math.rotl(u64, h, 5) ^ @as(u64, std.mem.readInt(u16, cube[24..26], .little))) *% 0x517cc1b727220a95;
    }

    pub fn get(self: *const SearchBuffers, cube: *const [26]u8) ?struct { cost: u16, move_code: u8 } {
        const h = hash(cube);
        const tag: u8 = @as(u8, @truncate(h >> 56)) | 0x80;
        var idx: usize = @truncate(h & self.mask);
        while (true) : (idx = (idx + 1) & self.mask) {
            const c = self.ctrl[idx];
            if (c == 0) return null;
            if (c != tag) continue;
            const entry = &self.visited[idx];
            if (std.mem.eql(u8, &entry.cube, cube)) return .{ .cost = entry.cost, .move_code = entry.move_code };
        }
    }

    fn grow(self: *SearchBuffers) !void {
        const old_ctrl = self.ctrl;
        const old_entries = self.visited;
        const new_cap = old_ctrl.len * 2;
        const new_ctrl = try self.allocator.alloc(u8, new_cap);
        errdefer self.allocator.free(new_ctrl);
        @memset(new_ctrl, 0);
        const new_entries = try self.allocator.alloc(VisitedEntry, new_cap);
        const new_mask = new_cap - 1;

        for (old_ctrl, old_entries) |c, entry| {
            if (c == 0) continue;
            var idx: usize = @truncate(hash(&entry.cube) & new_mask);
            while (new_ctrl[idx] != 0) : (idx = (idx + 1) & new_mask) {}
            new_ctrl[idx] = c;
            new_entries[idx] = entry;
        }

        self.ctrl = new_ctrl;
        self.visited = new_entries;
        self.mask = new_mask;
        self.allocator.free(old_ctrl);
        self.allocator.free(old_entries);
    }

    pub fn putOrUpdate(self: *SearchBuffers, cube: *const [26]u8, cost: u16, move_code: u8) !enum { inserted, updated, skipped } {
        if (self.count * 2 >= self.ctrl.len) try self.grow();
        const h = hash(cube);
        const tag: u8 = @as(u8, @truncate(h >> 56)) | 0x80;
        var idx: usize = @truncate(h & self.mask);
        while (true) : (idx = (idx + 1) & self.mask) {
            const c = self.ctrl[idx];
            if (c == 0) {
                self.ctrl[idx] = tag;
                self.visited[idx] = .{ .cube = cube.*, .cost = cost, .move_code = move_code };
                self.count += 1;
                return .inserted;
            }
            if (c != tag) continue;
            const entry = &self.visited[idx];
            if (std.mem.eql(u8, &entry.cube, cube)) {
                if (cost < entry.cost) {
                    entry.cost = cost;
                    entry.move_code = move_code;
                    return .updated;
                }
                return .skipped;
            }
        }
    }
};

pub const MinHeap = struct {
    items: []HeapNode,
    len: usize = 0,
    allocator: std.mem.Allocator,

    pub fn push(self: *MinHeap, node: HeapNode) !void {
        if (self.len >= self.items.len) {
            const new_cap = if (self.items.len == 0) 4 else self.items.len * 2;
            self.items = try self.allocator.realloc(self.items, new_cap);
        }
        var i = self.len;
        self.len += 1;
        while (i > 0) {
            const p = (i - 1) / 2;
            if (self.items[p].prio <= node.prio) break;
            self.items[i] = self.items[p];
            i = p;
        }
        self.items[i] = node;
    }

    pub fn pop(self: *MinHeap) ?HeapNode {
        if (self.len == 0) return null;
        const res = self.items[0];
        self.len -= 1;
        if (self.len > 0) {
            const last = self.items[self.len];
            var i: usize = 0;
            while (2 * i + 1 < self.len) {
                var child = 2 * i + 1;
                if (child + 1 < self.len and self.items[child + 1].prio < self.items[child].prio) child += 1;
                if (last.prio <= self.items[child].prio) break;
                self.items[i] = self.items[child];
                i = child;
            }
            self.items[i] = last;
        }
        return res;
    }
};

pub const AStarResult = struct { cube: FastCube, path: []usize };

pub fn astar(
    allocator: std.mem.Allocator,
    bufs: *SearchBuffers,
    start: FastCube,
    ctx: anytype,
    comptime is_goal: fn (@TypeOf(ctx), *const FastCube) bool,
    comptime heuristic: fn (@TypeOf(ctx), *const FastCube) f64,
    random_weight: f64,
    max_moves: usize,
) !?AStarResult {
    if (is_goal(ctx, &start)) return .{ .cube = start, .path = try allocator.alloc(usize, 0) };

    var budget = max_moves;
    var rng = Rng.init(42);
    var frontier = MinHeap{ .items = bufs.frontier, .len = 0, .allocator = bufs.allocator };
    defer bufs.frontier = frontier.items;

    while (true) {
        bufs.clear();
        frontier.len = 0;
        _ = try bufs.putOrUpdate(&start.data, 0, 0);
        try frontier.push(.{ .prio = 0, .cost = 0, .last_move = 0, .state = start });

        var simulated: usize = 0;
        var found_goal: ?AStarResult = null;

        while (simulated < budget) {
            const node = frontier.pop() orelse break;
            const visited_cost = bufs.get(&node.state.data) orelse continue;
            if (node.cost > visited_cost.cost) continue;

            for (0..12) |m| {
                if (simulated >= budget) break;
                if (tables.prune_move[node.last_move][m]) continue;

                simulated += 1;
                const next_cost = node.cost + 1;
                const dst = node.state.applyMove(m);
                const mv_code: u8 = @intCast(m + 1);

                if ((try bufs.putOrUpdate(&dst.data, next_cost, mv_code)) == .skipped) continue;

                if (is_goal(ctx, &dst)) {
                    var path_list = std.ArrayList(usize).init(allocator);
                    errdefer path_list.deinit();
                    var curr = dst;
                    while (bufs.get(&curr.data)) |entry| {
                        if (entry.move_code == 0) break;
                        const code = entry.move_code - 1;
                        try path_list.append(code);
                        curr = curr.applyMove(code ^ 1);
                    }
                    std.mem.reverse(usize, path_list.items);
                    found_goal = .{ .cube = dst, .path = try path_list.toOwnedSlice() };
                    break;
                }

                const hw = if (random_weight > 0.0) @max(0.01, rng.randomGauss(1.0, random_weight)) else 1.0;
                const prio = @as(f64, @floatFromInt(next_cost)) + hw * heuristic(ctx, &dst);
                try frontier.push(.{ .prio = prio, .cost = next_cost, .last_move = mv_code, .state = dst });
            }
            if (found_goal != null) break;
        }
        _ = total_moves_simulated.fetchAdd(simulated, .monotonic);

        if (found_goal) |res| return res;
        if (simulated < budget or random_weight <= 0.0) return null;

        log("search budget of {d} moves exceeded; restarting", .{budget});
        budget = @max(budget + 1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(budget)) * 1.5)));
    }
}

// -- Canonical Reduction Phases & 3D Solver -----------------------------------

fn countSolved(c: *const FastCube, indices: []const usize) usize {
    var cnt: usize = 0;
    for (indices) |j| if (tables.is_solved[j][c.data[j]]) {
        cnt += 1;
    };
    return cnt;
}

fn topDone(i: usize, c: *const FastCube) bool {
    return countSolved(c, &TOP_EDGES) >= @min(4, i + 1) and
        countSolved(c, &TOP_CUBELETS) >= @min(9, i + 1) and
        countSolved(c, &MID_CUBELETS) >= @min(17, i + 1);
}

fn crossDone(i: usize, c: *const FastCube) bool {
    const d = Vec3{ .x = 0, .y = 0, .z = -1 };
    var ok: usize = 0;
    for (BOT_EDGES) |j| if (tables.rotations[c.data[j]].mulVec(d).eql(d)) {
        ok += 1;
    };
    const req_bot = if (i >= 3) @min(4, i - 3) else 0;
    return topDone(16, c) and ok >= @min(4, i + 1) and countSolved(c, &BOT_EDGES) >= req_bot;
}

fn cornersDone(i: usize, c: *const FastCube) bool {
    var pos: usize = 0;
    for (BOT_CORNERS) |j| if (tables.is_pos_solved[j][c.data[j]]) {
        pos += 1;
    };
    return crossDone(7, c) and pos >= @min(4, i + 1);
}

fn cornerOriented(c: usize, r_init: usize, bot: usize) bool {
    var r = r_init;
    for (0..4) |_| {
        if (tables.is_solved[c][r]) return true;
        r = tables.transition[bot][c][r];
    }
    return false;
}

fn findMove(n: Vec3) usize {
    for (MOVES, 0..) |m, i| if (m.normal.eql(n) and m.dir == 1) return i;
    unreachable;
}

pub fn solveEndgame(cube: *FastCube, moves: *std.ArrayList(usize)) !void {
    const left = findMove(.{ .x = 0, .y = -1, .z = 0 });
    const top = findMove(.{ .x = 0, .y = 0, .z = 1 });
    const bot = findMove(.{ .x = 0, .y = 0, .z = -1 });
    const routine = [8]usize{ left ^ 1, top ^ 1, left, top, left ^ 1, top ^ 1, left, top };

    for (0..4) |_| {
        while (true) {
            var target_c: usize = 0;
            const target_pos = Vec3{ .x = 1, .y = -1, .z = -1 };
            for (0..26) |i| if (tables.rotations[cube.data[i]].mulVec(CUBELETS[i]).eql(target_pos)) {
                target_c = i;
                break;
            };
            if (cornerOriented(target_c, cube.data[target_c], bot)) break;
            for (routine) |m| try moves.append(cube.step(m));
        }
        try moves.append(cube.step(bot));
    }
    while (!cube.isSolved()) try moves.append(cube.step(bot));
}

fn solvePhase(
    allocator: std.mem.Allocator,
    bufs: *SearchBuffers,
    name: []const u8,
    total: usize,
    rw: f64,
    comptime goal: anytype,
    comptime h: anytype,
    c: *FastCube,
    moves: *std.ArrayList(usize),
) !void {
    for (0..total) |i| {
        log("{s} #{d}", .{ name, i + 1 });
        const res = (try astar(allocator, bufs, c.*, i, goal, h, rw, 100_000)) orelse return error.PhaseFailed;
        defer allocator.free(res.path);
        log("-> found solution with {d} moves", .{res.path.len});
        c.* = res.cube;
        try moves.appendSlice(res.path);
    }
    log("--------------------------------------------------", .{});
}

pub fn shuffle(iters: usize, seed: u64, cube: Cube) Cube {
    var rng = Rng.init(seed);
    var c = FastCube.fromCube(&cube);
    for (0..iters) |_| c = c.applyMove(@as(usize, @intFromFloat(rng.nextF64() * 12.0)) % 12);
    return c.toCube();
}

pub fn solve(allocator: std.mem.Allocator, cube: Cube) ![]Move {
    const t0 = std.time.nanoTimestamp();
    const start_sim = total_moves_simulated.load(.monotonic);
    var moves = std.ArrayList(usize).init(allocator);
    errdefer moves.deinit();
    var c = FastCube.fromCube(&cube);

    var bufs = try SearchBuffers.init(allocator, 65_536, 65_536);
    defer bufs.deinit();

    const H1 = struct {
        fn h(idx: usize, state: *const FastCube) f64 {
            const s = if (idx < 9) TOP_CUBELETS[0..] else MID_CUBELETS[0..];
            return state.layerH(s, &tables.sqrt_dist_solved, if (idx < 9) 8.0 else 4.0);
        }
    };
    const HCross = struct {
        fn h(_: usize, state: *const FastCube) f64 {
            return state.layerH(NON_BOT_CORNERS[0..], &tables.sqrt_dist_solved, 3.0);
        }
    };
    const HCorners = struct {
        fn h(_: usize, state: *const FastCube) f64 {
            const t_s = state.sumDist(TOP_CUBELETS[0..], &tables.sqrt_dist_solved);
            const m_s = state.sumDist(MID_LAYER[0..], &tables.sqrt_dist_solved);
            const b_s = state.sumDist(BOT_EDGES[0..], &tables.sqrt_dist_solved) + state.sumDist(BOT_CORNERS[0..], &tables.sqrt_dist_pos);
            return @as(f64, @floatCast((t_s * t_s) / 5.0 + (m_s * m_s) / 3.0 + (b_s * b_s) / 8.0));
        }
    };

    try solvePhase(allocator, &bufs, "solving cubelet", 17, 0.25, topDone, H1.h, &c, &moves);
    try solvePhase(allocator, &bufs, "solving bottom cross", 8, 0.25, crossDone, HCross.h, &c, &moves);
    try solvePhase(allocator, &bufs, "positioning bottom corners", 4, 0.30, cornersDone, HCorners.h, &c, &moves);
    try solveEndgame(&c, &moves);

    const t1 = std.time.nanoTimestamp();
    const elapsed = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const sim = total_moves_simulated.load(.monotonic) - start_sim;

    log("Solved cube in {d} moves.", .{moves.items.len});
    log("is_cube_solved: {}", .{c.isSolved()});
    log("- time elapsed: {d:.3} sec", .{elapsed});
    const moves_per_sec = @as(f64, @floatFromInt(sim)) / @max(0.0001, elapsed);
    log("- moves simulated: {d} ({d:.0} moves/sec)", .{ sim, moves_per_sec });

    const final_moves = try allocator.alloc(Move, moves.items.len);
    for (moves.items, 0..) |m, i| final_moves[i] = MOVES[m];
    moves.deinit();
    return final_moves;
}
