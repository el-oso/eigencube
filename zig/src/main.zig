const std = @import("std");
const eigencube = @import("eigencube");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // skip exe name
    var seed: u64 = 42;
    if (args.next()) |seed_str| {
        seed = std.fmt.parseInt(u64, seed_str, 10) catch |err| {
            std.debug.print("Error: invalid random seed '{s}' ({s}). Expected unsigned integer.\n", .{ seed_str, @errorName(err) });
            return err;
        };
    }

    eigencube.log("Solving scrambled cube (seed={d})...", .{seed});
    const scrambled = eigencube.shuffle(100_000, seed, eigencube.SOLVED_CUBE);
    const moves = try eigencube.solve(allocator, scrambled);
    defer allocator.free(moves);

    std.debug.print("Solution length: {d}\n", .{moves.len});
}
