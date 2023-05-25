
"""
    EvalFunc(ctrl::Int64) 

This structure is used to define functions evaluating residuals, constraints and corresponding jacobians.

Both functions for residuals and constraints must be written as follows and must not return any value (components are modified in the body of the function):

# Example definition of an EvalFunc type function

```jldoctest
function (h::EvalFunc)(x::Vector{Float64}, hx::Vector{Float64}, Jh::Matrix{Float64})
    if h.ctrl == 1 
        hx[:] = [...]
    elseif h.ctrl == 2
        Jh[:] = [...] # if supplied anatically
    end
    # The elseif block above could also be, if jacobian not supplied anatically
    # elseif h.ctrl == 2 h.ctrl = 0 end
    return
end
```

The `ctrl` field indicates what is computed (i.e. evalutation or jacobian) when calling a function of type `EvalFunction`.

* `ctrl` = ``1`` means the function `h` is evaluated at point `x`, (modifies in place vector `hx`)

* `ctrl` = ``2`` means the jacobian of `h` is computed at point `x` if jacobian is supplied anatically (then modifies in place matrix `Jh`)

* if jacobian is not supplied anatically, `ctrl` is set to ``0`` on return and jacobian is computed numerically.
"""
abstract type EvalFunc end

"""
    ResidualsEval <: EvalFunc

Subtype of [`EvalFunc`](@ref)  dedicated to the evalutation of residuals values and jacobian matrix.
"""
mutable struct ResidualsEval <: EvalFunc
    ctrl::Int64
end

ResidualsEval() = ResidualsEval(0)
"""
    ConstraintsEval <: EvalFunc

Subtype of [`EvalFunc`](@ref) dedicated to the evalutation of constraints values and jacobian matrix.
"""
mutable struct ConstraintsEval <: EvalFunc
    ctrl::Int64
end

ConstraintsEval() = ConstraintsEval(0)

### Tests structures alternatives

#= Idée : faire une structure dont les attributs sont les focntions d'évaluation
Définir deux fonctions prenant en argument la structure et qui appellent la fonction attribut correspondante (évaluation de la fonction ou calcul de la jacobienne) =# 
mutable struct EvaluationFunction{T} <: EvalFunc where T <: Function
    eval::T
    jac_eval::T
    ctrl::Int64
    nb_eval::Int64
    nb_jac_eval::Int64
end

function r(residuals::EvaluationFunction, x::Vector)
    residuals.nb_eval += 1
    return residuals.eval(x)
end

function jac_r(residuals::EvaluationFunction, x::Vector)
    residuals.nb_jac_eval += 1
    return residuals.jac_eval(x)
end
# réflechir comment gérer la jacobienne mieux ?