using TestItemRunner

@run_package_tests

@testsnippet Group begin
    using Eigencube: Mat3, ROTATION_MATRICES
    using LinearAlgebra: I
    # The 24 rotations of the cube, reached from the identity by quarter turns.
    function rotation_group()
        rots = [Mat3(I)]
        i = 1
        while i <= length(rots)
            for M in ROTATION_MATRICES
                R = M * rots[i]
                R in rots || push!(rots, R)
            end
            i += 1
        end
        return rots
    end
end

@testitem "quarter turns are proper rotations" setup = [Group] begin
    using LinearAlgebra: det
    for M in ROTATION_MATRICES
        @test M' * M == Mat3(I)
        @test det(M) == 1
        @test M^4 == Mat3(I)
    end
    @test length(rotation_group()) == 24
end

@testitem "one-byte rotation codes round-trip" setup = [Group] begin
    using Eigencube: encode, rotation, is_rotation_code
    for R in rotation_group()
        @test rotation(encode(R)) == R
        @test is_rotation_code(encode(R))
    end
    @test count(is_rotation_code, typemin(Int8):typemax(Int8)) == 24
end

@testitem "closed-form distances match breadth-first search" setup = [Group] begin
    using Eigencube: CUBELETS, is_cubelet_solved, is_cubelet_positioned, distance_to_solved, distance_to_position, norm1
    function bfs_distance(goal, c, R0)
        dist = Dict(R => 0 for R in rotation_group() if goal(c, R))
        queue = collect(keys(dist))
        for R in queue, M in ROTATION_MATRICES
            haskey(dist, M * R) || (dist[M * R] = dist[R] + 1; push!(queue, M * R))
        end
        return dist[R0]
    end
    for c in CUBELETS, R in rotation_group()
        @test distance_to_solved(c, R) == bfs_distance(is_cubelet_solved, c, R)
        norm1(c) == 3 && @test distance_to_position(c, R) == bfs_distance(is_cubelet_positioned, c, R)
    end
end

@testitem "every move is a 4-cycle and its inverse undoes it" begin
    using Eigencube: apply_move, inverse
    c = shuffle(1000, 7)
    for m in 1:12
        @test foldl((x, _) -> apply_move(x, m), 1:4; init = c) == c
        @test apply_move(c, m) != c
        @test apply_move(apply_move(c, m), inverse(m)) == c
    end
end

@testitem "solutions solve the cube" begin
    using Eigencube: apply_move
    for seed in (1, 42, 123)
        c = shuffle(100_000, seed)
        @test !is_solved(c)
        # Solved means R·diag(c) = diag(c); centers may still be turned about their own axis.
        @test is_solved(foldl(apply_move, solve(c); init = c))
    end
end

@testitem "solution lengths match the rust_fast reference" begin
    # Solution lengths printed by eigencube's rust_fast for seeds 1–5.
    @test [length(solve(shuffle(100_000, s))) for s in 1:5] == [128, 150, 166, 146, 142]
end

@testitem "invalid input is rejected" begin
    using Eigencube: Cube, IDENTITY, apply_move
    @test_throws "lanes must hold rotation codes" Cube(ntuple(i -> i == 3 ? Int8(0) : IDENTITY, 32))
    @test_throws "lanes must hold rotation codes" Cube(ntuple(i -> i == 30 ? Int8(13) : IDENTITY, 32))
    @test_throws "move index must be in 1:12" apply_move(SOLVED, 13)
    @test_throws "move index must be in 1:12" apply_move(SOLVED, 0)
end

@testitem "hot kernels do not allocate" begin
    using Eigencube: MOVES, Survey, TopAndMiddle, BottomCross, BottomCorners, heuristic, is_goal, apply_move
    allocated(f, args...) = (f(args...); @allocated f(args...))
    c = shuffle(1000, 3)
    s = Survey(c)
    @test iszero(allocated(*, MOVES[8], c))
    @test iszero(allocated(apply_move, c, 5))
    @test iszero(allocated(hash, c, zero(UInt)))
    @test iszero(allocated(Survey, c))
    for phase in (TopAndMiddle(), BottomCross(), BottomCorners())
        @test iszero(allocated(heuristic, phase, 3, s))
        @test iszero(allocated(is_goal, phase, 3, s))
    end
end
