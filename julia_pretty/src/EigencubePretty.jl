"""
A Rubik's cube solver written as plain linear algebra.

A [`Cubelet`](@ref) is its home vector `c ∈ {-1,0,1}³` and a rotation matrix
`R`. It sits at `R * c`, its stickers point along the columns of
`R * Diagonal(c)`, and it is solved when `R * Diagonal(c) == Diagonal(c)`. A
[`Move`](@ref) is a unit vector `v` and a direction `d = ±1`: it turns the
cubelets with `v ⋅ (R * c) > 0` by the quarter-turn matrix `v * v' - d * [v]ₓ`.

The cube is solved layer by layer with A* searches that restart with a larger
move budget when they run out. Their random numbers come from a fixed
xorshift64 stream, so the same scramble always yields the same solution.
"""
module EigencubePretty

using LinearAlgebra: Diagonal, ⋅
using StaticArrays: SVector, SMatrix, @SMatrix, deleteat
using DataStructures: BinaryMinHeap
using Memoization: @memoize
using Random: Random, AbstractRNG
using PrecompileTools: @compile_workload

export solve, scramble, solved_cube, is_solved

# -- Geometry ---------------------------------------------------------------------

const Vec3 = SVector{3, Int8}
const Mat3 = SMatrix{3, 3, Int8, 9}

norm1(v) = sum(abs, v)

const identity3 = Mat3(Diagonal(Vec3(1, 1, 1)))

"""
    Cubelet(home, rotation)

The cubelet whose solved position is `home`, turned by `rotation`.
"""
struct Cubelet
    home::Vec3
    rotation::Mat3
end

position(c::Cubelet) = c.rotation * c.home
is_solved(c::Cubelet) = c.rotation * Diagonal(c.home) == Diagonal(c.home)

"""
    Cube

The 26 visible cubelets; the hidden center at the origin is omitted.
"""
struct Cube
    cubelets::SVector{26, Cubelet}
end

Base.iterate(cube::Cube, state...) = iterate(cube.cubelets, state...)
Base.length(::Cube) = 26
Base.eltype(::Type{Cube}) = Cubelet

is_solved(cube::Cube) = all(is_solved, cube)

# x, y, z each range over {-1, 0, 1}, with x varying slowest.
const vectors = SVector{27}(vec([Vec3(x, y, z) for z in -1:1, y in -1:1, x in -1:1]))
const unit_vectors = SVector{6}(filter(v -> norm1(v) == 1, vectors))

const solved_cube = Cube(map(v -> Cubelet(v, identity3), deleteat(vectors, 14)))  # vectors[14] is the origin

# -- Moves ------------------------------------------------------------------------

"""
    Move(normal, dir)

A quarter turn of the slice `normal` points at, clockwise (`dir = 1`) or
counterclockwise (`dir = -1`) as seen looking at that face. A move is applied by
calling it: `move(cube)`, `move(cubelet)`, or `move(R)` for a lone rotation.
"""
struct Move
    normal::Vec3
    dir::Int8
end

# Sorted by (normal, dir): a move's index in `moves` is its rank in the search's
# tie-breaking order.
const moves = SVector{12}([Move(v, d) for v in unit_vectors for d in (-1, 1)])

inverse(move::Move) = Move(move.normal, -move.dir)

# The part of a vector along `v` is kept (`v * v'`) and the rest is turned by
# the cross product with `v`.
# https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
function rotation_matrix(move::Move)
    v, d = move.normal, move.dir
    x, y, z = v
    cross_with_v = @SMatrix Int8[0 -z y; z 0 -x; -y x 0]  # cross_with_v * p is v × p
    return v * v' - d * cross_with_v
end

# A move turns exactly the cubelets whose current position lies on its side of the cube.
is_in_slice(move::Move, c::Cubelet) = move.normal ⋅ position(c) > 0

(move::Move)(R::Mat3) = rotation_matrix(move) * R
(move::Move)(c::Cubelet) = is_in_slice(move, c) ? Cubelet(c.home, move(c.rotation)) : c
(move::Move)(cube::Cube) = Cube(map(move, cube.cubelets))

apply_moves(state, ms) = foldl((s, move) -> move(s), ms; init = state)

# -- Random numbers ---------------------------------------------------------------

"""
    Xorshift64(seed)

The xorshift64 generator of eigencube's reference implementation, as a
`Random.AbstractRNG`, so that `rand(rng)` reproduces the reference's stream.
"""
mutable struct Xorshift64 <: AbstractRNG
    state::UInt64
    Xorshift64(seed::Integer) = new(UInt64(max(seed, 1)))
end

function Random.rand(rng::Xorshift64, ::Type{Float64})
    rng.state ⊻= rng.state << 13
    rng.state ⊻= rng.state >> 7
    rng.state ⊻= rng.state << 17
    return (rng.state >> 11) * 0x1.0p-53
end

# A normal variate approximated by the sum of 12 uniforms, accumulated left to right.
gaussian(rng, mean, stdev) = mean + stdev * (sum(_ -> rand(rng), 1:12) - 6)

random_move(rng) = moves[floor(Int, rand(rng) * length(moves)) + 1]

"""
    scramble(iterations = 100_000, seed = 42, cube = solved_cube)

Apply `iterations` random moves, drawn from the xorshift64 stream seeded with
`seed`, to `cube`.
"""
function scramble(iterations::Integer = 100_000, seed::Integer = 42, cube::Cube = solved_cube)
    rng = Xorshift64(seed)
    return foldl((c, _) -> random_move(rng)(c), 1:iterations; init = cube)
end

# -- A* with move-budgeted restarts ------------------------------------------------

# Total order on states: a matrix by rows; a cube cubelet by cubelet, home
# vector first.
order_key(R::Mat3) = vec(R')
order_key(c::Cubelet) = vcat(c.home, order_key(c.rotation))
order_key(cube::Cube) = map(order_key, cube.cubelets)

"""
    Node(priority, cost, move, state)

A frontier entry: `state` was reached at `cost` by `moves[move]` (`move = 0` at
the start). The frontier pops the lowest `priority` first; ties go to the
higher cost, then the later move, then the greater state.
"""
struct Node{S}
    priority::Float64
    cost::Int
    move::Int
    state::S
end

function Base.isless(a::Node, b::Node)
    a.priority != b.priority && return a.priority < b.priority
    a.cost != b.cost && return a.cost > b.cost
    a.move != b.move && return a.move > b.move
    return order_key(b.state) < order_key(a.state)
end

# Never undo the previous move; of two commuting opposite-face moves, keep one
# order. `last` is the previous move's index in `moves`, or 0 for none.
function should_prune(last::Int, move::Move)
    iszero(last) && return false
    prev = moves[last]
    return prev == inverse(move) || (prev.normal ⋅ move.normal == -1 && prev.normal > move.normal)
end

# The moves worth trying after `moves[last]`, with their indices.
successors(last::Int) = ((k, move) for (k, move) in pairs(moves) if !should_prune(last, move))

"""
    astar(start, is_goal, heuristic = Returns(0.0); random_weight = 0.0, max_moves = 100_000)

Best-first search from `start` over the states reachable with `moves`. The
priority of a state is its cost plus `w * heuristic(state)`; with a positive
`random_weight`, `w` is drawn per state from a Gaussian with mean 1 and that
standard deviation (floored at 0.01), otherwise `w = 1`.

At most `max_moves` successor states are generated per attempt. A search that
exhausts the budget with a positive `random_weight` restarts, continuing the
same random stream, on a budget 1.5 times larger.

Returns `(goal_state, path)`, where `path` is a `Vector{Move}`, or `nothing` if
the frontier empties, or the budget runs out with a zero `random_weight`.
"""
function astar(start::S, is_goal, heuristic = Returns(0.0); random_weight = 0.0, max_moves = 100_000) where {S}
    is_goal(start) && return (start, Move[])
    rng = Xorshift64(42)
    weight() = random_weight > 0 ? max(gaussian(rng, 1.0, random_weight), 0.01) : 1.0
    # state => how it was first reached at its lowest cost; restarts reuse its capacity
    visited = Dict{S, @NamedTuple{cost::Int, move::Int}}()
    budget = max_moves
    while true
        result = bounded_search!(visited, start, is_goal, heuristic, weight, budget)
        result === out_of_budget || return result
        random_weight > 0 || return nothing
        budget = max(budget + 1, floor(Int, 1.5 * budget))
    end
    return
end

struct OutOfBudget end
const out_of_budget = OutOfBudget()
const unvisited = (cost = typemax(Int), move = 0)

# One A* attempt that generates at most `budget` successor states, recording
# reached states in `visited`. Returns `(goal_state, path)`, `nothing` if the
# frontier empties, or `out_of_budget`.
function bounded_search!(visited, start, is_goal, heuristic, weight, budget)
    empty!(visited)
    visited[start] = (cost = 0, move = 0)
    frontier = BinaryMinHeap([Node(0.0, 0, 0, start)])
    simulated = 0
    while !isempty(frontier)
        node = pop!(frontier)
        node.cost > visited[node.state].cost && continue
        for (k, move) in Iterators.take(successors(node.move), budget - simulated)
            simulated += 1
            state, cost = move(node.state), node.cost + 1
            get(visited, state, unvisited).cost <= cost && continue
            visited[state] = (; cost, move = k)
            is_goal(state) && return (state, path_to(visited, state))
            push!(frontier, Node(cost + weight() * heuristic(state), cost, k, state))
        end
        simulated == budget && return out_of_budget
    end
    return nothing
end

# The moves from the search's start to `state`, found by undoing each state's last move.
function path_to(visited, state)
    k = visited[state].move
    iszero(k) && return Move[]
    return [path_to(visited, inverse(moves[k])(state)); moves[k]]
end

# -- Heuristics -------------------------------------------------------------------

# Fewest moves that bring a lone rotation to a goal: a breadth-first search,
# one layer of rotations per move. Memoized below: a cubelet has only 24
# distinct rotations. This must not call `astar`: `astar` reaches these
# distances through its heuristic, and that recursion keeps inference from
# compiling the memoized lookup without boxing its arguments.
function fewest_moves(is_goal, rotation)
    layer, seen, depth = [rotation], Set([rotation]), 0
    while !any(is_goal, layer)
        layer = unique!([move(R) for R in layer for move in moves if move(R) ∉ seen])
        union!(seen, layer)
        depth += 1
    end
    return depth
end

# Memoization keys a call by its (positional, keyword) argument tuples; a cache
# typed for that key avoids boxing every lookup.
const DistanceCache = Dict{Tuple{Tuple{Cubelet}, Tuple{}}, Int}
@memoize DistanceCache min_moves_to_solved(c::Cubelet) = fewest_moves(R -> is_solved(Cubelet(c.home, R)), c.rotation)
@memoize DistanceCache min_moves_to_position(c::Cubelet) = fewest_moves(R -> R * c.home == c.home, c.rotation)

# The p-norm with p = 1/2 of the distances `f(c)` over the cubelets selected by
# `criterion(c)`: (Σ dᵢ^p)^(1/p) = (Σ √dᵢ)². Written with `sqrt` and a square
# because `d^0.5` is not always bit-equal to `sqrt(d)`, and the summation order
# fixes the floating-point result.
pnorm_half(f, cube, criterion) = sum(sqrt(f(c)) for c in cube if criterion(c); init = 0.0)^2

is_top(c::Cubelet) = c.home[3] == 1
is_middle(c::Cubelet) = iszero(c.home[3])
is_bottom(c::Cubelet) = c.home[3] == -1
is_top_or_middle(c::Cubelet) = c.home[3] >= 0
is_edge(c::Cubelet) = norm1(c.home) == 2
is_corner(c::Cubelet) = norm1(c.home) == 3
is_top_edge(c::Cubelet) = is_top(c) && is_edge(c)
is_bottom_edge(c::Cubelet) = is_bottom(c) && is_edge(c)
is_bottom_corner(c::Cubelet) = is_bottom(c) && is_corner(c)

top_layer_heuristic(cube) = pnorm_half(min_moves_to_solved, cube, is_top) / 8
middle_layer_heuristic(cube) = pnorm_half(min_moves_to_solved, cube, is_top_or_middle) / 4
bottom_layer_edge_heuristic(cube) = pnorm_half(min_moves_to_solved, cube, !is_bottom_corner) / 3

# Bottom corners only need to reach their position; everything else must be solved.
min_moves_to_target(c::Cubelet) = is_bottom_corner(c) ? min_moves_to_position(c) : min_moves_to_solved(c)

bottom_layer_corner_heuristic(cube) =
    pnorm_half(min_moves_to_solved, cube, is_top) / 5 +
    pnorm_half(min_moves_to_solved, cube, is_middle) / 3 +
    pnorm_half(min_moves_to_target, cube, is_bottom) / 8

# -- Reduction to a solved cube, layer by layer --------------------------------------

num_solved(cube, criterion) = count(c -> criterion(c) && is_solved(c), cube)

const down = Vec3(0, 0, -1)
has_yellow_bottom(c::Cubelet) = c.rotation * down == down
is_in_right_place(c::Cubelet) = position(c) == c.home
num_bottom_edges_positioned(cube) = count(c -> is_bottom_edge(c) && has_yellow_bottom(c), cube)
num_bottom_corners_positioned(cube) = count(c -> is_bottom_corner(c) && is_in_right_place(c), cube)

# Each phase repeats an A* search `num_steps` times; the goal of step `i`
# (counting from 0) is one more solved cubelet than before.
abstract type Phase end
struct TopAndMiddleLayer <: Phase end
struct BottomLayerEdges <: Phase end
struct BottomLayerCorners <: Phase end

num_steps(::TopAndMiddleLayer) = 17
num_steps(::BottomLayerEdges) = 8
num_steps(::BottomLayerCorners) = 4

random_weight(::TopAndMiddleLayer) = 0.25
random_weight(::BottomLayerEdges) = 0.25
random_weight(::BottomLayerCorners) = 0.3

is_goal(::TopAndMiddleLayer, i, cube) =
    num_solved(cube, is_top_edge) >= min(4, i + 1) &&
    num_solved(cube, is_top) >= min(9, i + 1) &&
    num_solved(cube, is_top_or_middle) >= min(17, i + 1)
is_goal(::BottomLayerEdges, i, cube) =
    num_solved(cube, is_top_or_middle) == 17 &&
    num_bottom_edges_positioned(cube) >= min(4, i + 1) &&
    num_solved(cube, is_bottom_edge) >= min(4, i - 3)
is_goal(::BottomLayerCorners, i, cube) =
    num_solved(cube, is_top_or_middle) == 17 &&
    num_solved(cube, is_bottom_edge) == 4 &&
    num_bottom_corners_positioned(cube) >= min(4, i + 1)

heuristic(::TopAndMiddleLayer, i, cube) = i < 9 ? top_layer_heuristic(cube) : middle_layer_heuristic(cube)
heuristic(::BottomLayerEdges, i, cube) = bottom_layer_edge_heuristic(cube)
heuristic(::BottomLayerCorners, i, cube) = bottom_layer_corner_heuristic(cube)

function solve_step(phase::Phase, i, cube)
    result = astar(cube, c -> is_goal(phase, i, c), c -> heuristic(phase, i, c); random_weight = random_weight(phase))
    isnothing(result) && error("$(nameof(typeof(phase))) step $(i + 1) found no solution")
    return result
end

# Returns the cube after the phase and the moves that got it there.
function solve_phase(phase::Phase, cube)
    steps = accumulate((step, i) -> solve_step(phase, i, first(step)), 0:(num_steps(phase) - 1); init = (cube, Move[]))
    return first(last(steps)), reduce(vcat, last.(steps))
end

# -- Endgame -----------------------------------------------------------------------

const left, top, bottom = Move(Vec3(0, -1, 0), 1), Move(Vec3(0, 0, 1), 1), Move(Vec3(0, 0, -1), 1)

# (L' U' L U)² twists the front-left-bottom corner in place.
const twist = SVector(inverse(left), inverse(top), left, top, inverse(left), inverse(top), left, top)
twists(n) = collect(Iterators.flatten(Iterators.repeated(twist, n)))

bottom_left_front_corner(cube) = first(c for c in cube if position(c) == Vec3(1, -1, -1))

# Whether turning the bottom face alone can solve the front-left-bottom corner.
function is_corner_oriented(cube)
    c = bottom_left_front_corner(cube)
    return any(n -> is_solved(Cubelet(c.home, rotation_matrix(bottom)^n * c.rotation)), 0:3)
end

# The fewest `n` in `range` for which `moves_for(n)` takes `cube` to `goal`.
function fewest(goal, moves_for, cube, range)
    n = findfirst(n -> goal(apply_moves(cube, moves_for(n))), range)
    isnothing(n) && error("no move count in $range reaches the goal")
    return range[n]
end

# Orient the front-left-bottom corner, then turn the bottom face to bring up the
# next corner.
corner_moves(cube) = [twists(fewest(is_corner_oriented, twists, cube, 0:2)); bottom]

# Orient the four bottom corners in turn, then align the bottom face. Returns
# the final cube and the moves.
function solve_endgame(cube)
    cube, solution = foldl(1:4; init = (cube, Move[])) do (cube, solution), _
        ms = corner_moves(cube)
        (apply_moves(cube, ms), [solution; ms])
    end
    align = fill(bottom, fewest(is_solved, n -> fill(bottom, n), cube, 0:3))
    return apply_moves(cube, align), [solution; align]
end

"""
    solve(cube) -> Vector{Move}

Solve `cube` layer by layer: the top layer and middle edges, the bottom cross,
the bottom corner positions, then the corner twists. Returns the moves, in
order, that take `cube` to a solved state.
"""
function solve(cube::Cube)
    phases = (TopAndMiddleLayer(), BottomLayerEdges(), BottomLayerCorners())
    cube, solution = foldl(phases; init = (cube, Move[])) do (cube, solution), phase
        cube, ms = solve_phase(phase, cube)
        (cube, [solution; ms])
    end
    cube, endgame = solve_endgame(cube)
    is_solved(cube) || error("solver finished on an unsolved cube")
    return [solution; endgame]
end

# `julia --project=julia_pretty -m EigencubePretty [seed]` solves the cube
# scrambled from `seed` (default 42).
function (@main)(args)
    seed = isempty(args) ? 42 : parse(Int, only(args))
    println("Solved the cube scrambled from seed $seed in $(length(solve(scramble(100_000, seed)))) moves.")
    return 0
end

# Compile a whole solve into the package image, so the first one runs at full speed.
@compile_workload solve(scramble(1_000, 1))

end
