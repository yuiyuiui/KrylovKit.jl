module KrylovKitZygoteExt

using KrylovKit
import ChainRulesCore: AbstractThunk, unthunk
using Zygote: Zygote

# Zygote has a number of internal adjoints for LinearAlgebra wrapper constructors.
# Some of these adjoints do not accept `ChainRulesCore.AbstractThunk` cotangents,
# but thunks are routinely produced by ChainRules-based rules (and also in our AD tests).
#
# In particular, `Hermitian(A)` can fail during Zygote jacobian construction when a thunked
# cotangent is propagated into Zygote's internal pullback. We patch this by unthunking
# the cotangent before delegating to Zygote's existing method.
#
# This is intentionally minimal and guarded: we only add the method if the corresponding
# internal callable type exists in the current Zygote version.
@static if isdefined(Zygote, Symbol("#back#933"))
    function (b::Zygote.var"#back#933")(Δ::AbstractThunk)
        return b(unthunk(Δ))
    end
end

end # module
