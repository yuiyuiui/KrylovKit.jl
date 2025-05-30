# Implementation details

## Orthogonalization
To denote a basis of vectors, e.g. to represent a given Krylov subspace, there is an
abstract type `Basis{T}`
```@docs
KrylovKit.Basis
```

Many Krylov based algorithms use an orthogonal basis to parameterize the Krylov subspace. In
that case, the specific implementation `OrthonormalBasis{T}` can be used:
```@docs
KrylovKit.OrthonormalBasis
```

We can orthogonalize or orthonormalize a given vector to another vector (assumed normalized)
or to a given [`KrylovKit.OrthonormalBasis`](@ref) using
```@docs
KrylovKit.orthogonalize
KrylovKit.orthonormalize
```
or using the possibly in-place versions
```@docs
KrylovKit.orthogonalize!!
KrylovKit.orthonormalize!!
```

The expansion coefficients of a general vector in terms of a given orthonormal basis can be obtained as
```@docs
KrylovKit.project!!
```
whereas the inverse calculation is obtained as
```@docs
KrylovKit.unproject!!
```

An orthonormal basis can be transformed using a rank-1 update using
```@docs
KrylovKit.rank1update!
```

Note that this changes the subspace. A mere rotation of the basis, which does not change
the subspace spanned by it, can be computed using

```@docs
KrylovKit.basistransform!
```

## Block Krylov method
The block version of the Krylov subspace algorithm is an approach to extending Krylov subspace techniques from
a starting block. It is mainly used for solving eigenvalue problems with repeated eigenvalues.

In our implementation, a block of vectors is stored in a new data structure `Block`,
which implements the [`KrylovKit.block_qr!`](@ref) and [`KrylovKit.block_reorthogonalize!`](@ref) interfaces.

```@docs
KrylovKit.Block
```

A block of vectors can be orthonormalized using
```@docs
KrylovKit.block_qr!
```

Additional procedure applied to the block is as follows:
```@docs
KrylovKit.block_reorthogonalize!
```



## Dense linear algebra

KrylovKit relies on Julia's `LinearAlgebra` module from the standard library for most of its
dense linear algebra dependencies.

## Factorization types
The central ingredient in a Krylov based algorithm is a Krylov factorization or
decomposition of a linear map. Such partial factorizations are represented as a
`KrylovFactorization`, of which `LanczosFactorization`, `BlockLanczosFactorization` and `ArnoldiFactorization` are three
concrete implementations:

```@docs
KrylovKit.KrylovFactorization
KrylovKit.LanczosFactorization
KrylovKit.BlockLanczosFactorization
KrylovKit.ArnoldiFactorization
KrylovKit.GKLFactorization
```

A `KrylovFactorization` or `GKLFactorization` can be destructured into its defining
components using iteration, but these can also be accessed using the following functions
```@docs
basis
rayleighquotient
residual
normres
rayleighextension
```

As the `rayleighextension` is typically a simple basis vector, we have created a dedicated
type to represent this without having to allocate an actual vector, i.e.
```@docs
KrylovKit.SimpleBasisVector
```

Furthermore, to store the Rayleigh quotient of the Arnoldi factorization in a manner that
can easily be expanded, we have constructed a custom matrix type to store the Hessenberg
matrix in a packed format (without zeros):

```@docs
KrylovKit.PackedHessenberg
```

## Factorization iterators
Given a linear map ``A`` and a starting vector ``x₀``, a Krylov factorization is obtained
by sequentially building a Krylov subspace ``{x₀, A x₀, A² x₀, ...}``. Rather then using
this set of vectors as a basis, an orthonormal basis is generated by a process known as
Lanczos, BlockLanczos or Arnoldi iteration (for symmetric/hermitian and for general matrices,
respectively). These processes are represented as iterators in Julia:

```@docs
KrylovKit.KrylovIterator
KrylovKit.LanczosIterator
KrylovKit.BlockLanczosIterator
KrylovKit.ArnoldiIterator
```

Similarly, there is also an iterator for the Golub-Kahan-Lanczos bidiagonalization proces:

```@docs
KrylovKit.GKLIterator
```

As an alternative to the standard iteration interface from Julia Base (using `iterate`),
these iterative processes and the factorizations they produce can also be manipulated
using the following functions:

```@docs
expand!
shrink!
initialize
initialize!
```
