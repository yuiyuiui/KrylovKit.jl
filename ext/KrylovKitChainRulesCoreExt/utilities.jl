safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)

# ------------------------------------------------------------------
# Utilities for thunked cotangents
#
# Some AD pipelines (notably Zygote.jacobian) pass cotangents containing nested
# `Thunk`/`InplaceableThunk` objects. Provide a small recursive helper that
# materializes these when needed.
# ------------------------------------------------------------------

deep_unthunk(x) = x
deep_unthunk(x::AbstractThunk) = deep_unthunk(unthunk(x))
deep_unthunk(x::StructuralTangent) = map(deep_unthunk, x)
deep_unthunk(x::Tuple) = map(deep_unthunk, x)
deep_unthunk(x::NamedTuple) = map(deep_unthunk, x)

# ------------------------------------------------------------------
# ChainRules for LinearAlgebra wrappers
#
# Zygote's built-in adjoints for wrappers like `Hermitian` do not accept
# `ChainRulesCore.Thunk` / `InplaceableThunk` cotangents. Some tests (and users)
# intentionally thunk cotangents, so we provide ChainRules rrules here that
# explicitly `unthunk` the incoming cotangent.
#
# Note: `Hermitian(A, uplo)` only uses one triangle of `A`; the pullback returns
# a gradient that is zero on the unused triangle.
# ------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(LinearAlgebra.Hermitian), A::AbstractMatrix)
    H = LinearAlgebra.Hermitian(A)
    project_A = ProjectTo(A)
    function Hermitian_pullback(ΔH)
        ΔH = unthunk(ΔH)
        if ΔH isa AbstractZero
            return NoTangent(), ZeroTangent()
        end
        G = ΔH isa LinearAlgebra.Hermitian ? parent(ΔH) : ΔH
        ΔA = zero(A)
        @inbounds for j in axes(ΔA, 2), i in axes(ΔA, 1)
            if i <= j
                ΔA[i, j] = G[i, j]
            end
        end
        return NoTangent(), project_A(ΔA)
    end
    return H, Hermitian_pullback
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.Hermitian), A::AbstractMatrix,
                              uplo::Symbol)
    H = LinearAlgebra.Hermitian(A, uplo)
    project_A = ProjectTo(A)
    function Hermitian_pullback(ΔH)
        ΔH = unthunk(ΔH)
        if ΔH isa AbstractZero
            return NoTangent(), ZeroTangent(), NoTangent()
        end
        G = ΔH isa LinearAlgebra.Hermitian ? parent(ΔH) : ΔH
        ΔA = zero(A)
        if uplo === :U
            @inbounds for j in axes(ΔA, 2), i in axes(ΔA, 1)
                if i <= j
                    ΔA[i, j] = G[i, j]
                end
            end
        else
            @inbounds for j in axes(ΔA, 2), i in axes(ΔA, 1)
                if i >= j
                    ΔA[i, j] = G[i, j]
                end
            end
        end
        return NoTangent(), project_A(ΔA), NoTangent()
    end
    return H, Hermitian_pullback
end

# vecs are assumed orthonormal
function orthogonalprojector(vecs, n)
    function projector(w)
        w′ = zerovector(w)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], inner(vecs[i], w))
        end
        return w′
    end
    return projector
end
function orthogonalcomplementprojector(vecs, n)
    function projector(w)
        w′ = scale(w, 1)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], -inner(vecs[i], w))
        end
        return w′
    end
    return projector
end
# vecs are not assumed orthonormal, G is the Cholesky factorisation of the overlap matrix
function orthogonalprojector(vecs, n, G::Cholesky)
    overlaps = zeros(eltype(G), n)
    function projector(w)
        @inbounds for i in 1:n
            overlaps[i] = inner(vecs[i], w)
        end
        overlaps = ldiv!(G, overlaps)
        w′ = zerovector(w)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], +overlaps[i])
        end
        return w′
    end
    return projector
end
function orthogonalcomplementprojector(vecs, n, G::Cholesky)
    overlaps = zeros(eltype(G), n)
    function projector(w)
        @inbounds for i in 1:n
            overlaps[i] = inner(vecs[i], w)
        end
        overlaps = ldiv!(G, overlaps)
        w′ = scale(w, 1)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], -overlaps[i])
        end
        return w′
    end
    return projector
end

function _realview(v::AbstractVector{Complex{T}}) where {T}
    v_real = reinterpret(T, v)
    return view(v_real, axes(v_real, 1)[begin:2:end])
end

function _imagview(v::AbstractVector{Complex{T}}) where {T}
    v_real = reinterpret(T, v)
    return view(v_real, axes(v_real, 1)[(begin + 1):2:end])
end
