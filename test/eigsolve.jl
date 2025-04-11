@testset "Lanczos - eigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2
            v = rand(T, (n,))
            n1 = div(n, 2)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=2)
            D1, V1, info = @test_logs (:info,) eigsolve(wrapop(A, Val(mode)),
                                                        wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs eigsolve(wrapop(A, Val(mode)),
                                wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n1 + 1, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) eigsolve(wrapop(A, Val(mode)),
                                         wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=2)
            @test_logs (:info,) eigsolve(wrapop(A, Val(mode)),
                                         wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                          verbosity=3)
            @test_logs((:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))
            alg = Lanczos(; orth=orth, krylovdim=4, maxiter=1, tol=tolerance(T),
                          verbosity=4)
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs((:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))

            @test KrylovKit.eigselector(wrapop(A, Val(mode)), scalartype(v); krylovdim=n,
                                        maxiter=1,
                                        tol=tolerance(T), ishermitian=true) isa Lanczos
            n2 = n - n1
            alg = Lanczos(; krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            D2, V2, info = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                   wrapvec(v, Val(mode)),
                                                   n2, :LR, alg)
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvals(A)

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)

            alg = Lanczos(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
                                                  wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end

@testset "Lanczos - eigsolve iteratively ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (N, N)) .- one(T) / 2
            A = (A + A') / 2
            v = rand(T, (N,))
            alg = Lanczos(; krylovdim=2 * n, maxiter=10,
                          tol=tolerance(T), eager=true, verbosity=0)
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n, :SR, alg)
            D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LR,
                                     alg)

            l1 = info1.converged
            l2 = info2.converged
            @test l1 > 0
            @test l2 > 0
            @test D1[1:l1] ≈ eigvals(A)[1:l1]
            @test D2[1:l2] ≈ eigvals(A)[N:-1:(N - l2 + 1)]

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
        end
    end
end

@testset "Arnoldi - eigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            v = rand(T, (n,))
            n1 = div(n, 2)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T))
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n1, :SR, alg)

            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=0)
            @test_logs eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n1 + 2, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                         :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=2)
            @test_logs (:info,) eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                         :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                          verbosity=3)
            @test_logs((:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))
            alg = Arnoldi(; orth=orth, krylovdim=4, maxiter=1, tol=tolerance(T),
                          verbosity=4)
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs((:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))

            @test KrylovKit.eigselector(wrapop(A, Val(mode)), eltype(v); orth=orth,
                                        krylovdim=n, maxiter=1,
                                        tol=tolerance(T)) isa Arnoldi
            n2 = n - n1
            alg = Arnoldi(; orth=orth, krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            D2, V2, info2 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n2, :LR, alg)
            D = sort(sort(eigvals(A); by=imag, rev=true); alg=MergeSort, by=real)
            D2′ = sort(sort(D2; by=imag, rev=true); alg=MergeSort, by=real)
            @test vcat(D1[1:n1], D2′[(end - n2 + 1):end]) ≈ D

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)

            if T <: Complex
                n1 = div(n, 2)
                D1, V1, info = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                        :SI,
                                        alg)
                n2 = n - n1
                D2, V2, info = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n2,
                                        :LI,
                                        alg)
                D = sort(eigvals(A); by=imag)

                @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ D

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                @test A * U1 ≈ U1 * Diagonal(D1)
                @test A * U2 ≈ U2 * Diagonal(D2)
            end

            alg = Arnoldi(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
                                                  wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end

@testset "Arnoldi - eigsolve iteratively ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (N, N)) .- one(T) / 2
            v = rand(T, (N,))
            alg = Arnoldi(; krylovdim=3 * n, maxiter=20,
                          tol=tolerance(T), eager=true, verbosity=0)
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n, :SR, alg)
            D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LR,
                                     alg)
            D3, V3, info3 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LM,
                                     alg)
            D = sort(eigvals(A); by=imag, rev=true)

            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test l1 > 0
            @test l2 > 0
            @test l3 > 0
            @test D1[1:l1] ≊ sort(D; alg=MergeSort, by=real)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg=MergeSort, by=real, rev=true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            U3 = stack(unwrapvec, V3)
            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            R3 = stack(unwrapvec, info3.residual)
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
            @test A * U3 ≈ U3 * Diagonal(D3) + R3

            if T <: Complex
                D1, V1, info1 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :SI, alg)
                D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :LI, alg)
                D = eigvals(A)

                l1 = info1.converged
                l2 = info2.converged
                @test l1 > 0
                @test l2 > 0
                @test D1[1:l1] ≈ sort(D; by=imag)[1:l1]
                @test D2[1:l2] ≈ sort(D; by=imag, rev=true)[1:l2]

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                R1 = stack(unwrapvec, info1.residual)
                R2 = stack(unwrapvec, info2.residual)
                @test A * U1 ≈ U1 * Diagonal(D1) + R1
                @test A * U2 ≈ U2 * Diagonal(D2) + R2
            end
        end
    end
end

@testset "Arnoldi - realeigsolve iteratively ($mode)" for mode in
                                                          (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64) : (Float64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            V = exp(randn(T, (N, N)) / 10)
            D = randn(T, N)
            A = V * Diagonal(D) / V
            v = rand(T, (N,))
            alg = Arnoldi(; krylovdim=3 * n, maxiter=20,
                          tol=tolerance(T), eager=true, verbosity=0)
            D1, V1, info1 = @constinferred realeigsolve(wrapop(A, Val(mode)),
                                                        wrapvec(v, Val(mode)), n, :SR, alg)
            D2, V2, info2 = realeigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :LR,
                                         alg)
            D3, V3, info3 = realeigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :LM,
                                         alg)
            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test l1 > 0
            @test l2 > 0
            @test l3 > 0
            @test D1[1:l1] ≊ sort(D; alg=MergeSort)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg=MergeSort, rev=true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

            @test eltype(D1) == T
            @test eltype(D2) == T
            @test eltype(D3) == T

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            U3 = stack(unwrapvec, V3)
            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            R3 = stack(unwrapvec, info3.residual)
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
            @test A * U3 ≈ U3 * Diagonal(D3) + R3

            if mode == :vector # solve eigenvalue problem as complex problem with real linear operator
                V = exp(randn(T, (2N, 2N)) / 10)
                D = randn(T, 2N)
                Ar = V * Diagonal(D) / V
                Z = zeros(T, N, N)
                J = [Z -I; I Z]
                Ar1 = (Ar - J * Ar * J) / 2
                Ar2 = (Ar + J * Ar * J) / 2
                A = complex.(Ar1[1:N, 1:N], -Ar1[1:N, (N + 1):end])
                B = complex.(Ar2[1:N, 1:N], +Ar2[1:N, (N + 1):end])
                f = buildrealmap(A, B)
                v = rand(complex(T), (N,))
                alg = Arnoldi(; krylovdim=3 * n, maxiter=20,
                              tol=tolerance(T), eager=true, verbosity=0)
                D1, V1, info1 = @constinferred realeigsolve(f, v, n, :SR, alg)
                D2, V2, info2 = realeigsolve(f, v, n, :LR, alg)
                D3, V3, info3 = realeigsolve(f, v, n, :LM, alg)

                l1 = info1.converged
                l2 = info2.converged
                l3 = info3.converged
                @test l1 > 0
                @test l2 > 0
                @test l3 > 0
                @test D1[1:l1] ≊ sort(D; alg=MergeSort)[1:l1]
                @test D2[1:l2] ≊ sort(D; alg=MergeSort, rev=true)[1:l2]
                # sorting by abs does not seem very reliable if two distinct eigenvalues are close
                # in absolute value, so we perform a second sort afterwards using the real part
                @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

                @test eltype(D1) == T
                @test eltype(D2) == T
                @test eltype(D3) == T

                U1 = stack(V1)
                U2 = stack(V2)
                U3 = stack(V3)
                R1 = stack(info1.residual)
                R2 = stack(info2.residual)
                R3 = stack(info3.residual)
                @test A * U1 + B * conj(U1) ≈ U1 * Diagonal(D1) + R1
                @test A * U2 + B * conj(U2) ≈ U2 * Diagonal(D2) + R2
                @test A * U3 + B * conj(U3) ≈ U3 * Diagonal(D3) + R3
            end
        end
    end
end

@testset "Arnoldi - realeigsolve imaginary eigenvalue warning" begin
    A = diagm(vcat(1, 1, exp.(-(0.1:0.02:2))))
    A[2, 1] = 1e-9
    A[1, 2] = -1e-9
    v = ones(Float64, size(A, 1))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-8, verbosity=0))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-8, verbosity=1))
    @test_logs (:info,) realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-8, verbosity=2))
    @test_logs (:warn,) realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-10, verbosity=1))
    @test_logs (:warn,) (:info,) realeigsolve(A, v, 1, :LM,
                                              Arnoldi(; tol=1e-10, verbosity=2))

    # this should not trigger a warning
    A[1, 2] = A[2, 1] = 0
    A[1, 1] = 1
    A[2, 2] = A[3, 3] = 0.99
    A[3, 2] = 1e-6
    A[2, 3] = -1e-6
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-12, verbosity=0))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-12, verbosity=1))
    @test_logs (:info,) realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-12, verbosity=2))
end


@testset "Block Lanczos - eigsolve for large sparse matrix and map input" begin

    function toric_code_strings(m::Int, n::Int)
        li = LinearIndices((m, n))
        bottom(i, j) = li[mod1(i, m), mod1(j, n)] + m * n
        right(i, j) = li[mod1(i, m), mod1(j, n)]
        xstrings = Vector{Int}[]
        zstrings = Vector{Int}[]
        for i in 1:m, j in 1:n
            # face center
            push!(xstrings, [bottom(i, j - 1), right(i, j), bottom(i, j), right(i - 1, j)])
            # cross
            push!(zstrings, [right(i, j), bottom(i, j), right(i, j + 1), bottom(i + 1, j)])
        end
        return xstrings, zstrings
    end

    function pauli_kron(n::Int, ops::Pair{Int, Char}...)
        mat = sparse(1.0I, 2^n, 2^n)
        for (pos, op) in ops
            if op == 'X'
                σ = sparse([0 1; 1 0])
            elseif op == 'Y'
                σ = sparse([0 -im; im 0])
            elseif op == 'Z'
                σ = sparse([1 0; 0 -1])
            elseif op == 'I'
                σ = sparse(1.0I, 2, 2)
            else
                error("Unknown Pauli operator $op")
            end

            left = sparse(1.0I, 2^(pos - 1), 2^(pos - 1))
            right = sparse(1.0I, 2^(n - pos), 2^(n - pos))
            mat = kron(left, kron(σ, right)) * mat
        end
        return mat
    end

    # define the function to construct the Hamiltonian matrix
    function toric_code_hamiltonian_matrix(m::Int, n::Int)
        xstrings, zstrings = toric_code_strings(m, n)
        N = 2 * m * n  # total number of qubits

        # initialize the Hamiltonian matrix as a zero matrix
        H = spzeros(2^N, 2^N)

        # add the X-type operator terms
        for xs in xstrings[1:(end-1)]
            ops = [i => 'X' for i in xs]
            H += pauli_kron(N, ops...)
        end

        for zs in zstrings[1:(end-1)]
            ops = [i => 'Z' for i in zs]
            H += pauli_kron(N, ops...)
        end

        return H
    end

    Random.seed!(4)
    sites_num = 3
    p = 5 # block size
    X1 = Matrix(qr(rand(2^(2 * sites_num^2), p)).Q)
    get_value_num = 10
    tol = 1e-6
    h_mat = toric_code_hamiltonian_matrix(sites_num, sites_num)

    # matrix input
    D, U, info = eigsolve(-h_mat, X1, get_value_num, :SR,
        Lanczos(; maxiter = 20, tol = tol, blockmode = true))
    @show D[1:get_value_num]
    @test count(x -> abs(x + 16.0) < 2.0 - tol, D[1:get_value_num]) == 4
    @test count(x -> abs(x + 16.0) < tol, D[1:get_value_num]) == 4

    # map input
    D, U, info = eigsolve(x -> -h_mat * x, X1, get_value_num, :SR,
        Lanczos(; maxiter = 20, tol = tol, blockmode = true))
    @show D[1:get_value_num]
    @test count(x -> abs(x + 16.0) < 1.9, D[1:get_value_num]) == 4
    @test count(x -> abs(x + 16.0) < 1e-8, D[1:get_value_num]) == 4

end

#= 
Why don’t I use the "vector", "inplace", or "outplace" modes here?
In the ordinary Lanczos algorithm, each iteration generates a single Lanczos vector, which is then added to the Lanczos basis. 
However, in the block version, I can’t pre-allocate arrays when using the WrapOp type. If I stick with the WrapOp approach, 
I would need to wrap each vector in a block individually as [v], which is significantly slower compared to pre-allocating.

Therefore, instead of using the "vector", "inplace", or "outplace" modes defined in testsetup.jl, 
I directly use matrices and define the map function manually for better performance.

Currently, I only test the out-of-place map because testing the in-place version requires solving the pre-allocation issue. 
In the future, I plan to add a non-Hermitian eigenvalue solver and implement more abstract types to improve efficiency. 
As a result, I’ve decided to postpone dealing with the in-place test issue for now.
=#

@testset "Block Lanczos - eigsolve full" begin
    @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
        Random.seed!(1234)
        A0 = rand(T, (n, n)) .- one(T) / 2
        A0 = (A0 + A0') / 2
        block_size = 2
        x₀m = Matrix(qr(rand(T, n, block_size)).Q)
        x₀ = [x₀m[:, i] for i in 1:block_size]
        n1 = div(n, 2)  # eigenvalues to solve
        eigvalsA = eigvals(A0)
        # Different from Lanczos, we don't set maxiter =1 here because the iteration times in Lanczos
        # in in fact in the control of deminsion of Krylov subspace. And we Don't use it.
        @testset for A in [A0, x -> A0 * x]
            alg = Lanczos(; krylovdim = n, maxiter = 4, tol = tolerance(T), verbosity = 2, blockmode = true)
            D1, V1, info = @test_logs (:info,) eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n, maxiter = 4, tol = tolerance(T), verbosity = 1, blockmode = true)
            @test_logs eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n1 + 1, maxiter = 4, tol = tolerance(T), verbosity = 1, blockmode = true)
            @test_logs (:warn,) eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n, maxiter = 4, tol = tolerance(T), verbosity = 2, blockmode = true)
            @test_logs (:info,) eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n1, maxiter = 4, tol = tolerance(T), verbosity = 3, blockmode = true)
            @test_logs((:info,), (:warn,), (:info,), eigsolve(A, x₀, 1, :SR, alg))
            # Because of the _residual! function, I can't make sure the stability of types temporarily. 
            # So I ignore the test of @constinferred
            n2 = n - n1
            alg = Lanczos(; krylovdim = 2 * n, maxiter = 4, tol = tolerance(T), blockmode = true)
            D2, V2, info = eigsolve(A, x₀, n2, :LR, alg)
            D2[1:n2]
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvalsA

            U1 = hcat(V1...)
            U2 = hcat(V2...)

            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            @test (x -> KrylovKit.apply(A, x)).(V1) ≈ D1 .* V1
            @test (x -> KrylovKit.apply(A, x)).(V2) ≈ D2 .* V2

            alg = Lanczos(; krylovdim = 2n, maxiter = 5, tol = tolerance(T), verbosity = 1, blockmode = true)
            @test_logs (:warn,) (:warn,) eigsolve(A, x₀, n + 1, :LM, alg)
        end
    end
end

# krylovdim is not used in block Lanczos so I don't add eager mode.
@testset "Block Lanczos - eigsolve iteratively" begin
    @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
        A0 = rand(T, (N, N)) .- one(T) / 2
        A0 = (A0 + A0') / 2
        block_size = 5
        x₀m = Matrix(qr(rand(T, N, block_size)).Q)
        x₀ = [x₀m[:, i] for i in 1:block_size]
        eigvalsA = eigvals(A0)
        @testset for A in [A0, x -> A0 * x]
            A = copy(A0)

            alg = Lanczos(; maxiter = 20, tol = tolerance(T), blockmode = true)
            D1, V1, info1 = eigsolve(A, x₀, n, :SR, alg)
            D2, V2, info2 = eigsolve(A, x₀, n, :LR, alg)

            l1 = info1.converged
            l2 = info2.converged

            @test l1 > 0
            @test l2 > 0
            @test D1[1:l1] ≈ eigvalsA[1:l1]
            @test D2[1:l2] ≈ eigvalsA[N:-1:(N-l2+1)]

            U1 = hcat(V1[1:l1]...);
            U2 = hcat(V2[1:l2]...);
            R1 = hcat(info1.residual[1:l1]...);
            R2 = hcat(info2.residual[1:l2]...);

            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I
            @test hcat([KrylovKit.apply(A, U1[:, i]) for i in 1:l1]...) ≈ U1 * Diagonal(D1) + R1
            @test hcat([KrylovKit.apply(A, U2[:, i]) for i in 1:l2]...) ≈ U2 * Diagonal(D2) + R2

        end
    end
end

# linear operator A must satisfies that A'H = HA. it means it's a self-adjoint operator.
@testset "Block Lanczos - eigsolve for abstract type" begin
    T = ComplexF64
    H = rand(T, (n, n))
    H = H' * H + I
    block_size = 2
    eig_num = 2
    Hip(x::Vector, y::Vector) = x' * H * y
    x₀ = [InnerProductVec(rand(T, n), Hip) for i in 1:block_size]
    Aip(x::InnerProductVec) = InnerProductVec(H * x.vec, Hip)
    D, V, info = eigsolve(Aip, x₀, eig_num, :SR, Lanczos(; krylovdim = n, maxiter = 10, tol = tolerance(T),
        verbosity = 0, blockmode = true))
    D_true = eigvals(H)
    @test D ≈ D_true[1:eig_num]
    @test KrylovKit.blockinner(V, V; S = T) ≈ I
    @test findmax([norm(Aip(V[i]) - D[i] * V[i]) for i in 1:eig_num])[1] < tolerance(T)
end

# For user interface, we should give one vector input block lanczos method.
@testset "Block Lanczos for init_generator - eigsolve full" begin
    @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
        T = ComplexF64
        Random.seed!(1234)
        A0 = rand(T, (n, n)) .- one(T) / 2
        A0 = (A0 + A0') / 2
        block_size = 2
        x₀ = rand(T, n)
        n1 = div(n, 2)  # eigenvalues to solve
        eigvalsA = eigvals(A0)
        # Different from Lanczos, we don't set maxiter =1 here because the iteration times in Lanczos
        # in in fact in the control of deminsion of Krylov subspace. And we Don't use it.
        @testset for A in [A0, x -> A0 * x]
            A = copy(A0)
            alg = Lanczos(; krylovdim = n, maxiter = 4, tol = tolerance(T), verbosity = 2, blockmode = true, 
                            init_generator = true, blocksize = block_size)
            D1, V1, info = @test_logs (:info,) eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n, maxiter = 4, tol = tolerance(T), verbosity = 1, blockmode = true, 
                            init_generator = true, blocksize = block_size)
            @test_logs eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n1 + 1, maxiter = 4, tol = tolerance(T), verbosity = 1, blockmode = true, 
                            init_generator = true, blocksize = block_size)
            @test_logs (:warn,) eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n, maxiter = 4, tol = tolerance(T), verbosity = 2, blockmode = true, 
                            init_generator = true, blocksize = block_size)
            @test_logs (:info,) eigsolve(A, x₀, n1, :SR, alg)
            alg = Lanczos(; krylovdim = n1, maxiter = 4, tol = tolerance(T), verbosity = 3, blockmode = true, 
                            init_generator = true, blocksize = block_size)
            @test_logs((:info,), (:warn,), (:info,), eigsolve(A, x₀, 1, :SR, alg))
            # Because of the _residual! function, I can't make sure the stability of types temporarily. 
            # So I ignore the test of @constinferred
            n2 = n - n1
            alg = Lanczos(; krylovdim = 2 * n, maxiter = 4, tol = tolerance(T), blockmode = true, 
                            init_generator = true, blocksize = block_size)
            D2, V2, info = eigsolve(A, x₀, n2, :LR, alg)
            D2[1:n2]
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvalsA

            U1 = hcat(V1...)
            U2 = hcat(V2...)

            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            @test (x -> KrylovKit.apply(A, x)).(V1) ≈ D1 .* V1
            @test (x -> KrylovKit.apply(A, x)).(V2) ≈ D2 .* V2

            alg = Lanczos(; krylovdim = 2n, maxiter = 5, tol = tolerance(T), verbosity = 1, blockmode = true, 
                            init_generator = true, blocksize = block_size)
            @test_logs (:warn,) (:warn,) eigsolve(A, x₀, n + 1, :LM, alg)
        end
    end
end

@testset "Block Lanczos for init_generator - eigsolve for abstract type" begin
    T = ComplexF64
    H = rand(T, (n, n))
    H = H' * H + I
    block_size = 2
    eig_num = 2
    Hip(x::Vector, y::Vector) = x' * H * y
    x₀ = InnerProductVec(rand(T, n), Hip)
    Aip(x::InnerProductVec) = InnerProductVec(H * x.vec, Hip)
    D, V, info = eigsolve(Aip, x₀, eig_num, :SR, Lanczos(; krylovdim = n, maxiter = 10, tol = tolerance(T),
        verbosity = 0, blockmode = true, init_generator = true, blocksize = block_size))
    D_true = eigvals(H)
    @test D ≈ D_true[1:eig_num]
    @test KrylovKit.blockinner(V, V; S = T) ≈ I
    @test findmax([norm(Aip(V[i]) - D[i] * V[i]) for i in 1:eig_num])[1] < tolerance(T)
end
