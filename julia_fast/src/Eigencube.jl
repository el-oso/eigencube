"""
A fast Rubik's cube solver over eigencube's linear-algebra model.

A cubelet is its home vector `c ∈ {-1,0,1}³` and a rotation matrix `R`: it sits
at `R c` and is solved when `R * Diagonal(c) == Diagonal(c)`. A quarter turn of
face `n` multiplies `R` by the turn's matrix for every cubelet with
`n ⋅ (R c) > 0`.

A [`Cube`](@ref) stores each rotation as one `Int8` (the axes of its first two
columns; the third is their cross product), so a state is 32 bytes. Every
operation decodes the 26 rotations, does the matrix arithmetic per cubelet, and
re-encodes, in loops the compiler vectorizes across cubelets.

The search reproduces eigencube's `rust_fast` step for step — same random
stream, priority encoding and heap tie-breaking — so both find the same
solution for the same seed.
"""
module Eigencube

using LinearAlgebra: Diagonal, I, tr, ⋅, ×
using StaticArrays: SVector, SMatrix, MVector, @SMatrix
using Base.Cartesian: @nif
using PrecompileTools: @compile_workload

export solve, shuffle, SOLVED, is_solved

const Vec3 = SVector{3, Int8}
const Mat3 = SMatrix{3, 3, Int8, 9}

norm1(v) = sum(abs, v)

# -- Rotations as one byte ----------------------------------------------------

# The per-cubelet functions below are `@inline` so that loops over the 32 lanes
# vectorize: an out-of-line call in the loop body forces one scalar call per lane.

# A signed unit vector ±eₖ has axis code ±k.
@inline axis_code(u) = u[1] + Int8(2) * u[2] + Int8(3) * u[3]
@inline unit_vector(a::Int8) = Vec3(ntuple(k -> ifelse(abs(a) == k, sign(a), zero(a)), 3))

"""
    rotation(q::Int8) -> Mat3

The rotation with code `q = 8(a + 3) + (b + 3)`, where `a` and `b` are the axis
codes of its first two columns `u` and `v`; the third column is `u × v`.
[`encode`](@ref) is the inverse.
"""
@inline function rotation(q::Int8)
    u, v = columns(q)
    return hcat(u, v, u × v)
end
# The first two columns `u`, `v` of the rotation with code `q`.
@inline columns(q::Int8) = unit_vector((q >> 3) - Int8(3)), unit_vector((q & Int8(7)) - Int8(3))
@inline encode(R) = (axis_code(R[:, 1]) + Int8(3)) << 3 | (axis_code(R[:, 2]) + Int8(3))

const IDENTITY = encode(Mat3(I))

function is_rotation_code(q::Int8)
    0 <= q <= 54 || return false
    u, v = columns(q)
    return norm1(u) == 1 && norm1(v) == 1 && iszero(u ⋅ v)
end

# -- Cubelets and lanes -------------------------------------------------------

# Cubelet i occupies lane i of every 32-lane state; lanes 27–32 are padding at
# the origin with the identity rotation, which no move turns.
const NLANES = 32
const Lanes{T} = NTuple{NLANES, T}
# Every point of {-1, 0, 1}³ except the core, z varying fastest and x slowest.
const CUBELETS = Tuple(filter(!iszero, [Vec3(x, y, z) for z in -1:1, y in -1:1, x in -1:1]))
const NCUBELETS = length(CUBELETS)
const HOME_X, HOME_Y, HOME_Z = ntuple(k -> ntuple(i -> i <= NCUBELETS ? CUBELETS[i][k] : zero(Int8), NLANES), 3)
@inline home(i) = Vec3(HOME_X[i], HOME_Y[i], HOME_Z[i])

"""
    Cube

Lane `i` holds the rotation code of cubelet `CUBELETS[i]`.
"""
struct Cube
    codes::Lanes{Int8}
    Cube(codes::Lanes{Int8}) =
        all(i -> is_rotation_code(codes[i]) && (i <= NCUBELETS || codes[i] == IDENTITY), 1:NLANES) ? new(codes) :
        throw(ArgumentError("lanes must hold rotation codes, with the identity in lanes 27–32: $codes"))
    # For codes produced by `encode`, which are valid by construction.
    global unchecked_cube(codes::Lanes{Int8}) = new(codes)
end

const SOLVED = Cube(ntuple(_ -> IDENTITY, NLANES))

@inline rotation(c::Cube, i::Int) = rotation(c.codes[i])

# -- Moves --------------------------------------------------------------------

"""
    QuarterTurn{K, S, D}

The quarter turn of the face with normal `S * eₖ`, clockwise as seen looking at
that face for `D = 1` and counterclockwise for `D = -1`. `turn * cube` applies it.
"""
struct QuarterTurn{K, S, D} end

normal(::QuarterTurn{K, S}) where {K, S} = Vec3(ntuple(i -> i == K ? S : 0, 3))
direction(::QuarterTurn{K, S, D}) where {K, S, D} = D

# Rodrigues' formula for a quarter turn about unit axis v: v vᵀ - d [v]ₓ.
cross_matrix((x, y, z)) = @SMatrix [0 -z y; z 0 -x; -y x 0]
rotation_matrix(t::QuarterTurn) = Mat3(normal(t) * normal(t)' - direction(t) * cross_matrix(normal(t)))

# Turn every cubelet in the slice `n ⋅ (R c) > 0` by `M`.
@inline function turn(c::Cube, M::Mat3, n::Vec3)
    out = MVector{NLANES, Int8}(undef)
    for i in 1:NLANES
        R = rotation(c, i)
        out[i] = ifelse(n ⋅ (R * home(i)) > 0, encode(M * R), c.codes[i])
    end
    return unchecked_cube(Tuple(out))
end

@inline Base.:*(t::QuarterTurn, c::Cube) = turn(c, rotation_matrix(t), normal(t))

# Reference order: faces -x, -y, -z, +z, +y, +x, each counterclockwise then
# clockwise, so move m's inverse is ((m - 1) ⊻ 1) + 1.
const MOVES = Tuple(QuarterTurn{K, S, D}() for (K, S) in ((1, -1), (2, -1), (3, -1), (3, 1), (2, 1), (1, 1)) for D in (-1, 1))
const ROTATION_MATRICES = map(rotation_matrix, MOVES)
inverse(m::Integer) = ((m - 1) ⊻ 1) + 1

# Dispatch a run-time move index to its `QuarterTurn`, so the turn is compiled
# with a constant matrix.
function apply_move(c::Cube, m::Int)
    m in eachindex(MOVES) || throw(ArgumentError("move index must be in 1:12, got $m"))
    return @nif 12 k -> (m == k) k -> (MOVES[k] * c)
end

# Move `t` is skipped after move `p` when it undoes `p`, or the two faces are
# opposite and out of canonical order. `PRUNE[last + 1][m]`, with `last = 0`
# for no previous move.
skips(p, t) = normal(p) == normal(t) && direction(p) == -direction(t) || normal(p) ⋅ normal(t) == -1 && normal(p) > normal(t)
const PRUNE = (map(_ -> false, MOVES), map(p -> map(t -> skips(p, t), MOVES), MOVES)...)

# -- Per-cubelet geometry -----------------------------------------------------

@inline is_cubelet_solved(c::Vec3, R::Mat3) = (D = Mat3(Diagonal(c)); R * D == D)
@inline is_cubelet_positioned(c::Vec3, R::Mat3) = R * c == c
@inline keeps_bottom_down(c::Vec3, R::Mat3) = R[3, 3] == 1

"""
    distance_to_solved(c, R) -> Int8

Fewest quarter turns of the whole group taking `R` to a rotation that solves
cubelet `c`. A center is solved once its axis is fixed, so the distance is
`1 - c ⋅ (R c)`. An edge or corner needs `R = I`; its distance follows from the
rotation angle (the trace) and, at 180°, from whether the axis is a coordinate
axis (a diagonal entry of 1) or a face diagonal.
"""
@inline function distance_to_solved(c::Vec3, R::Mat3)
    center = Int8(1) - c ⋅ (R * c)
    t = tr(R)
    half_turn_about_axis = (R[1, 1] == 1) | (R[2, 2] == 1) | (R[3, 3] == 1)
    other = ifelse(t == 3, Int8(0), ifelse(t == 1, Int8(1), ifelse(iszero(t) | half_turn_about_axis, Int8(2), Int8(3))))
    return ifelse(norm1(c) == 1, center, other)
end

# For a corner, each quarter turn changes one coordinate of its position.
@inline distance_to_position(c::Vec3, R::Mat3) = sum(R * c .!= c; init = zero(Int8))

# √d for the distances 0 … 3 that occur.
const SQRT = Float32.(sqrt.((0, 1, 2, 3)))
root(d::Integer) = SQRT[d + 1]

# -- Layers ---------------------------------------------------------------------

# Layers as tuples of the indices of the cubelets with home `c` where `f(c)`,
# in increasing order, so sums over them unroll at compile time.
layer(f) = Tuple(i for i in 1:NCUBELETS if f(CUBELETS[i]))
@inline is_bottom_corner(c) = c[3] == -1 && norm1(c) == 3
const ALL, TOP, MIDDLE, TOP_AND_MIDDLE = layer(_ -> true), layer(c -> c[3] == 1), layer(c -> iszero(c[3])), layer(c -> c[3] >= 0)
const TOP_EDGES, BOTTOM_EDGES = layer(c -> c[3] == 1 && norm1(c) == 2), layer(c -> c[3] == -1 && norm1(c) == 2)
const BOTTOM_CORNERS, NOT_BOTTOM_CORNERS = layer(is_bottom_corner), layer(!is_bottom_corner)

count_in(flags::Lanes{Bool}, layer::Tuple) = count(i -> flags[i], layer)

"""
    Survey(cube)

Every per-cubelet quantity the search reads, computed in one pass over the
lanes: the cubelets that are solved, in their home position, and keeping the
bottom face down, and the square root of each cubelet's quarter-turn distance —
to its home position for a bottom corner, to solved otherwise.
"""
struct Survey
    solved::Lanes{Bool}
    positioned::Lanes{Bool}
    down::Lanes{Bool}
    distance::Lanes{Int8}
end

function Survey(cube::Cube)
    solved, positioned, down, d = (MVector{NLANES, T}(undef) for T in (Bool, Bool, Bool, Int8))
    for i in 1:NLANES
        c, R = home(i), rotation(cube, i)
        solved[i] = is_cubelet_solved(c, R)
        positioned[i] = is_cubelet_positioned(c, R)
        down[i] = keeps_bottom_down(c, R)
        d[i] = ifelse(is_bottom_corner(c), distance_to_position(c, R), distance_to_solved(c, R))
    end
    return Survey(Tuple(solved), Tuple(positioned), Tuple(down), Tuple(d))
end

# Σ √dᵢ over a layer, as a sequential Float32 sum in increasing cubelet order.
sum_roots(s::Survey, layer::Tuple) = foldl((acc, i) -> acc + root(s.distance[i]), layer; init = 0.0f0)

is_solved(cube::Cube) = count_in(Survey(cube).solved, ALL) == NCUBELETS

# -- Random stream (xorshift64, as in the reference) --------------------------

mutable struct Rng
    x::UInt64
end
Rng(seed::Integer) = Rng(UInt64(max(seed, 1)))

function next_f64!(r::Rng)
    x = r.x ⊻ r.x << 13
    x ⊻= x >> 7
    r.x = x ⊻ x << 17
    return Float64(r.x >> 11) * 0x1p-53
end

# A normal variate approximated by the sum of 12 uniforms, accumulated left to right.
random_gauss!(r::Rng, mean::Float64, stdev::Float64) = mean + stdev * (foldl((s, _) -> s + next_f64!(r), 1:12; init = 0.0) - 6.0)

# -- Hashing and ordering -------------------------------------------------------

# Bytes k+1 … k+8 of `s` as a little-endian word.
word(s, k) = foldl((w, j) -> w | UInt64(s[k + j] % UInt8) << (8 * (j - 1)), ntuple(identity, Val(8)); init = zero(UInt64))

function Base.hash(c::Cube, h::UInt)
    for k in (0, 8, 16, 24)
        h = (bitrotate(h, 5) ⊻ word(c.codes, k)) * 0x517cc1b727220a95
    end
    h = (h ⊻ (h >> 29)) * 0xbf58476d1ce4e5b9
    return h ⊻ (h >> 32)
end

# The reference stores each rotation as its index in a breadth-first
# enumeration of the group from the identity, and breaks heap ties by
# comparing those bytes.
const BFS_INDEX = let rots = [Mat3(I)], index = zeros(Int8, 64)
    for R in rots, M in ROTATION_MATRICES  # `rots` grows while it is iterated
        M * R in rots || push!(rots, M * R)
    end
    foreach(((j, R),) -> index[encode(R) + 1] = j - 1, pairs(rots))
    Tuple(index)
end

# Lexicographic order of the reference's state bytes.
reference_bytes(c::Cube) = ntuple(i -> BFS_INDEX[c.codes[i] + 1], Val(NCUBELETS))
reference_less(a::Cube, b::Cube) = isless(reference_bytes(a), reference_bytes(b))

# -- Frontier ---------------------------------------------------------------------

"A search node: `state`, reached at `cost` by move `move` (0 at the start)."
struct Node
    cost::UInt16
    move::UInt8
    state::Cube
end

"A heap entry: a node's priority and its index in the frontier's node list."
struct Entry
    priority::UInt64
    node::UInt32
end

# Priority queue over preallocated buffers: the first `added` slots of `nodes`
# hold every node enqueued so far, which never move; a 4-ary heap of `length`
# 16-byte (priority, node index) entries orders them. Heap position i (0-based)
# is stored at index i + HEAP_OFFSET, behind three padding entries, so the four
# children of i, at 4i + 1 … 4i + 4, fill one 64-byte cache line.
const HEAP_OFFSET = 4
mutable struct Frontier
    const nodes::Vector{Node}
    const heap::Vector{Entry}
    added::Int
    length::Int
end
Frontier() = Frontier(Node[], fill(Entry(0, 0), HEAP_OFFSET - 1), 0, 0)

Base.isempty(f::Frontier) = iszero(f.length)
entry(f::Frontier, i::Int) = f.heap[i + HEAP_OFFSET]
set_entry!(f::Frontier, i::Int, e::Entry) = (f.heap[i + HEAP_OFFSET] = e)

# Empty `f`, with room for `capacity` nodes, and enqueue `start`.
function restart!(f::Frontier, start::Cube, capacity::Int)
    length(f.nodes) < capacity && (resize!(f.nodes, capacity); resize!(f.heap, capacity + HEAP_OFFSET - 1))
    f.added = f.length = 0
    return push!(f, Node(0, 0, start) => zero(UInt64))
end

# Pop order of the reference's max-heap of (Reverse(priority), cost, move, state):
# lowest priority first, then highest cost, move, and state bytes. The order is
# total up to identical nodes, so any valid heap pops the same sequence.
function before(f::Frontier, a::Entry, b::Entry)
    a.priority != b.priority && return a.priority < b.priority
    x, y = f.nodes[a.node], f.nodes[b.node]
    x.cost != y.cost && return x.cost > y.cost
    x.move != y.move && return x.move > y.move
    return reference_less(y.state, x.state)
end

# `push!` writes into the capacity reserved by `restart!` and never grows the
# buffers; exceeding it is a BoundsError. `push!` and `popfirst!` must inline
# into the search loop: as calls they cost about 5% of a solve.
@inline function Base.push!(f::Frontier, (node, priority)::Pair{Node, UInt64})
    f.added += 1
    f.nodes[f.added] = node
    x = Entry(priority, f.added)
    i = f.length
    f.length += 1
    while i > 0
        parent = entry(f, (i - 1) >> 2)
        before(f, x, parent) || break
        set_entry!(f, i, parent)
        i = (i - 1) >> 2
    end
    set_entry!(f, i, x)
    return f
end

@inline function Base.popfirst!(f::Frontier)
    top = entry(f, 0)
    n = f.length -= 1
    x = entry(f, n)
    i = 0
    while 4i + 1 < n
        first_child = 4i + 1
        best = first_child
        for c in (first_child + 1):min(first_child + 3, n - 1)
            before(f, entry(f, c), entry(f, best)) && (best = c)
        end
        before(f, entry(f, best), x) || break
        set_entry!(f, i, entry(f, best))
        i = best
    end
    n > 0 && set_entry!(f, i, x)
    return f.nodes[top.node]
end

# -- Visited table: open addressing over a compact metadata array ---------------

# `meta[i]` packs `(0x80 | tag) << 24 | cost << 16 | move`, or is zero for an
# empty slot; the key lives in `keys[i]` and is read only when the 7-bit hash
# tag matches. Only the first `mask + 1` slots are in use.
mutable struct Visited
    const meta::Vector{UInt32}
    const keys::Vector{Cube}
    mask::Int
end
Visited() = Visited(UInt32[], Cube[], 0)

# Empty `t` with room for `n` entries at load factor ≤ ½, and record `start` at cost 0.
function restart!(t::Visited, start::Cube, n::Int)
    slots = nextpow(2, 2n)
    length(t.meta) < slots && (resize!(t.meta, slots); resize!(t.keys, slots))
    t.mask = slots - 1
    fill!(view(t.meta, 1:slots), zero(UInt32))
    relax!(t, start, hash(start, zero(UInt)), 0x0000, 0x00)
    return t
end

tag(h::UInt) = UInt32(0x80 | (h >> 57) % UInt8) << 24
slot_cost(m::UInt32) = (m >> 16) % UInt8 % UInt16
slot_move(m::UInt32) = m % UInt8

# Linear probe for `key` with hash `h`: its slot index, or the empty slot where it belongs.
function probe(t::Visited, key::Cube, h::UInt)
    mask, tg = t.mask, tag(h)
    i = Int(h & mask)
    while true
        m = t.meta[i + 1]
        iszero(m) && return i + 1
        m & 0xff000000 == tg && t.keys[i + 1] == key && return i + 1
        i = (i + 1) & mask
    end
    return
end

# The metadata recorded for `key`.
lookup(t::Visited, key::Cube) = t.meta[probe(t, key, hash(key, zero(UInt)))]

# Record `key` reached at `cost` by `move`, unless it is already known at a cost
# no higher. Returns whether the entry was written.
function relax!(t::Visited, key::Cube, h::UInt, cost::UInt16, move::UInt8)
    cost <= 0x00ff || error("search depth $cost exceeds the visited table's 8-bit cost field")
    i = probe(t, key, h)
    m = t.meta[i]
    !iszero(m) && slot_cost(m) <= cost && return false
    t.meta[i], t.keys[i] = tag(h) | UInt32(cost) << 16 | move, key
    return true
end

# LLVM's prefetch hint (read, keep in all cache levels), the one intrinsic in
# this package: Base has no prefetch. Requesting all 12 successors' slots before
# probing any lets their cache misses overlap, worth about 15% of a solve.
prefetch(p::Ptr) = ccall("llvm.prefetch.p0", llvmcall, Cvoid, (Ptr{Cvoid}, Int32, Int32, Int32), p, 0, 3, 1)

# Start loading the slot where a key with hash `h` would land.
function prefetch(t::Visited, h::UInt)
    i = Int(h & t.mask) + 1
    return GC.@preserve t (prefetch(pointer(t.meta, i)); prefetch(pointer(t.keys, i)))
end

"Search buffers, reused across the A* calls of a solve."
struct Workspace
    visited::Visited
    frontier::Frontier
end
Workspace() = Workspace(Visited(), Frontier())

# -- Reduction phases -----------------------------------------------------------

abstract type Phase end
"Top layer, then the middle edges: 17 cubelets one at a time."
struct TopAndMiddle <: Phase end
"Bottom edges turned down, then solved."
struct BottomCross <: Phase end
"Bottom corners moved to their positions."
struct BottomCorners <: Phase end

schedule(::TopAndMiddle) = (steps = 17, random_weight = 0.25)
schedule(::BottomCross) = (steps = 8, random_weight = 0.25)
schedule(::BottomCorners) = (steps = 4, random_weight = 0.3)

is_goal(::TopAndMiddle, i::Int, s::Survey) =
    count_in(s.solved, TOP_EDGES) >= min(4, i + 1) &&
    count_in(s.solved, TOP) >= min(9, i + 1) &&
    count_in(s.solved, TOP_AND_MIDDLE) >= min(17, i + 1)

is_goal(::BottomCross, i::Int, s::Survey) =
    is_goal(TopAndMiddle(), 16, s) &&
    count_in(s.down, BOTTOM_EDGES) >= min(4, i + 1) &&
    count_in(s.solved, BOTTOM_EDGES) >= min(4, max(i - 3, 0))

is_goal(::BottomCorners, i::Int, s::Survey) =
    is_goal(BottomCross(), 7, s) && count_in(s.positioned, BOTTOM_CORNERS) >= min(4, i + 1)

# (Σ √dᵢ)² / n over a layer.
layer_norm(s::Survey, layer::Tuple, n::Float32) = Float64(sum_roots(s, layer)^2 / n)

heuristic(::TopAndMiddle, i::Int, s::Survey) = i < 9 ? layer_norm(s, TOP, 8.0f0) : layer_norm(s, TOP_AND_MIDDLE, 4.0f0)
heuristic(::BottomCross, ::Int, s::Survey) = layer_norm(s, NOT_BOTTOM_CORNERS, 3.0f0)
function heuristic(::BottomCorners, ::Int, s::Survey)
    top, middle = sum_roots(s, TOP), sum_roots(s, MIDDLE)
    bottom = sum_roots(s, BOTTOM_EDGES) + sum_roots(s, BOTTOM_CORNERS)
    return Float64(top^2 / 5.0f0 + middle^2 / 3.0f0 + bottom^2 / 8.0f0)
end

"""
    astar(ws, start, phase, step, max_moves)

Best-first search from `start` to `is_goal(phase, step, ·)`. Each pushed node's
heuristic is scaled by a fresh Gaussian weight; a search that exhausts its move
budget restarts with a budget 1.5× larger. Returns the move indices, or
`nothing` if the frontier empties.
"""
function astar(ws::Workspace, start::Cube, phase::Phase, step::Int, max_moves::Int)
    is_goal(phase, step, Survey(start)) && return Int[]
    rng, budget = Rng(42), max_moves
    while true
        result = bounded_search!(ws, start, phase, step, rng, budget)
        result === OUT_OF_BUDGET || return result
        schedule(phase).random_weight > 0 || return nothing
        budget = max(budget + 1, floor(Int, 1.5 * budget))
    end
    return
end

struct OutOfBudget end
const OUT_OF_BUDGET = OutOfBudget()

# One search that stops after simulating `budget` moves. Returns
# the move indices, `nothing` if the frontier empties, or `OUT_OF_BUDGET`.
function bounded_search!(ws::Workspace, start::Cube, phase::Phase, step::Int, rng::Rng, budget::Int)
    (; visited, frontier) = ws
    # Each simulated move adds at most one node.
    restart!(visited, start, budget + 1)
    restart!(frontier, start, budget + 1)
    simulated = 0
    while !isempty(frontier)
        node = popfirst!(frontier)
        is_stale(visited, node) && continue
        states, hashes = successors(visited, node.state)
        cost = node.cost + 0x0001
        for m in eachindex(MOVES)
            is_pruned(node.move, m) && continue
            simulated == budget && return OUT_OF_BUDGET
            simulated += 1
            state = states[m]
            relax!(visited, state, hashes[m], cost, UInt8(m)) || continue
            survey = Survey(state)
            is_goal(phase, step, survey) && return path_to(visited, state)
            h = weight(phase, rng) * heuristic(phase, step, survey)
            push!(frontier, Node(cost, UInt8(m), state) => priority(cost + h))
        end
        simulated == budget && return OUT_OF_BUDGET
    end
    return nothing
end

# A node is stale once a cheaper path to its state has been recorded.
is_stale(t::Visited, node::Node) = node.cost > slot_cost(lookup(t, node.state))

is_pruned(last, m) = PRUNE[last + 1][m]

# All 12 successors and their hashes, with their table slots prefetched before
# the in-order relaxation reads them.
function successors(t::Visited, state::Cube)
    # `@inline`: left to itself the compiler calls each turn out of line, and the
    # precompiled copies of those calls run about 20% slower per solve.
    states = map(@inline(turn -> turn * state), MOVES)
    hashes = map(c -> hash(c, zero(UInt)), states)
    foreach(h -> prefetch(t, h), hashes)
    return states, hashes
end

# The heuristic's random scale: a Gaussian with mean 1, floored at 0.01.
weight(phase::Phase, rng::Rng) = max(random_gauss!(rng, 1.0, schedule(phase).random_weight), 0.01)

# Nonnegative Float64s order like their bit patterns, so the frontier compares integers.
priority(f::Float64) = reinterpret(UInt64, f)

apply_moves(c::Cube, ms) = foldl(apply_move, ms; init = c)

# -- Assembling the solution: cold code that builds move lists ---------------------

move_index(n, d) = findfirst(t -> normal(t) == Vec3(n) && direction(t) == d, MOVES)
const LEFT, UP, DOWN = move_index((0, -1, 0), 1), move_index((0, 0, 1), 1), move_index((0, 0, -1), 1)
# (L' U' L U)² twists the front-left-bottom corner in place.
const TWIST = SVector(inverse(LEFT), inverse(UP), LEFT, UP, inverse(LEFT), inverse(UP), LEFT, UP)

# Whether turning the bottom face alone can orient the front-left-bottom corner.
function is_corner_oriented(c::Cube)
    i = findfirst(i -> rotation(c, i) * home(i) == Vec3(1, -1, -1), 1:NCUBELETS)
    isnothing(i) && error("no cubelet sits at the front-left-bottom corner")
    D = ROTATION_MATRICES[DOWN]
    return any(k -> is_cubelet_solved(home(i), D^k * rotation(c, i)), 0:3)
end

# The moves from the search's start to `state`, found by undoing each state's last move.
function path_to(visited::Visited, state::Cube)
    m = Int(slot_move(lookup(visited, state)))
    return iszero(m) ? Int[] : [path_to(visited, apply_move(state, inverse(m))); m]
end

solve_phase(ws, phase::Phase, c::Cube) = foldl(0:(schedule(phase).steps - 1); init = Int[]) do ms, step
    path = astar(ws, apply_moves(c, ms), phase, step, 100_000)
    isnothing(path) && error("$phase step $(step + 1) found no solution")
    [ms; path]
end

# The fewest `n` in `range` for which `moves_for(n)` takes `c` to `goal`.
function fewest(goal, moves_for, c::Cube, range)
    n = findfirst(n -> goal(apply_moves(c, moves_for(n))), range)
    isnothing(n) && error("no move count in $range reaches the goal")
    return range[n]
end

twists(n) = repeat(TWIST, n)
# Orient the front-left-bottom corner, then turn the bottom face to bring up the next corner.
corner_moves(c::Cube) = [twists(fewest(is_corner_oriented, twists, c, 0:2)); DOWN]

# Orient the four bottom corners in turn, then align the bottom face.
function solve_endgame(c::Cube)
    ms = foldl((ms, _) -> [ms; corner_moves(apply_moves(c, ms))], 1:4; init = Int[])
    return [ms; fill(DOWN, fewest(is_solved, n -> fill(DOWN, n), apply_moves(c, ms), 0:3))]
end

# The search buffers grow to tens of megabytes; keeping one set per task
# avoids reallocating and re-faulting them on every solve.
task_workspace() = get!(Workspace, task_local_storage(), :eigencube_workspace)::Workspace

"""
    shuffle(iters, seed) -> Cube

Apply `iters` random moves to the solved cube, drawn from the reference's
xorshift stream seeded with `seed`.
"""
function shuffle(iters::Integer, seed::Integer)
    rng = Rng(seed)
    return foldl((c, _) -> apply_move(c, floor(Int, next_f64!(rng) * 12) + 1), 1:iters; init = SOLVED)
end

"""
    solve(c::Cube, ws = task_workspace()) -> Vector{Int}

Indices into `MOVES` of the moves that solve `c`: the top and middle layers,
the bottom cross, the bottom corner positions, then the endgame twists. By
default each task reuses one set of search buffers across calls.
"""
function solve(c::Cube, ws::Workspace = task_workspace())
    phases = (TopAndMiddle(), BottomCross(), BottomCorners())
    ms = foldl((ms, phase) -> [ms; solve_phase(ws, phase, apply_moves(c, ms))], phases; init = Int[])
    ms = [ms; solve_endgame(apply_moves(c, ms))]
    is_solved(apply_moves(c, ms)) || error("solver finished on an unsolved cube")
    return ms
end

# `julia --project=julia_fast -m Eigencube [seed]` solves the cube scrambled
# from `seed` (default 42).
function (@main)(args)
    seed = isempty(args) ? 42 : parse(Int, only(args))
    println("Solved the cube scrambled from seed $seed in $(length(solve(shuffle(100_000, seed)))) moves.")
    return 0
end

# Compile a whole solve into the package image, so the first one runs at full speed.
@compile_workload solve(shuffle(1_000, 1))

end
