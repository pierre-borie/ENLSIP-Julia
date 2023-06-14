
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

#= Idée : faire une structure dont les attributs sont les focntions d'évaluation =# 
mutable struct EvaluationFunction{T1,T2} <: EvalFunc where  {T1 <: Function, T2 <: Function}
    eval::T1
    jac_eval::T2
    ctrl::Int64
    nb_eval::Int64
    nb_jac_eval::Int64
end

function func_eval(h::EvaluationFunction, x::Vector, hx::Vector, Jhx::Matrix)
    
    #= ctrl = 1
    Evaluates the function h at point x 
    Result stored in place in vector hx =#
    if abs(h.ctrl) == 1
        hx[:] = h.eval(x)

    #= ctrl = 2
    Computes jacobian matrix of h at point x 
    Result stored in place in matrix Jhx =#
    elseif h.ctrl == 2
        Jhx[:] = h.jac_eval(x)
    end
    return
end