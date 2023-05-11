"""

    Iteration

Summarizes the useful informations about an iteration of the algorithm

* `x` : Departure point of the iteration 

* `p` : Descent direction

* `rx` : vector of size `m`, contains value of residuals at `x` 

* `cx` : vector of size `l`, contains value of constraints at `x`

* `t` : Number of constraints in current working set (ie constraints considered active)

* `α` : Value of steplength

* `λ` : Vector of size `t`, containts Lagrange multipliers estimates

* `rankA` : pseudo rank of matrix `A`, jacobian of active constraints

* `rankJ2` : pseudo rank of matrix `J2`, block extracted from `J`, jacobian of residuals

* `b_gn` : right handside of the linear system solved to compute first part of `p`

* `d_gn` :  right handside of the linear system solved to compute second part of `p`

* `predicted_reduction` : predicted linear progress

* `progress` :  reduction in the objective function

* `β` : scalar used to estimate convergence factor

* `restart` : indicate if current iteration is a restart step or no

* `first` : indicate if current iteration is the first one or no

* `add` : indicate if a constraint has been added to the working set 

* `del` : indicate if a constraint has been deleted from the working set

* `index_del` : index of the constraint that has been deleted from working set (`0` if no deletion)

* `code` : Its value caracterizes the method used to compute the search direction `p`

    - `1` represents Gauss-Newton method

    - `-1` represents Subspace minimization

    - `2`  represents Newton method

* `nb_newton_steps` : number of search direction computed using the method of Newton
"""
mutable struct Iteration
    x::Vector{Float64}
    p::Vector{Float64}
    rx::Vector{Float64}
    cx::Vector{Float64}
    t::Int64
    α::Float64
    index_α_upp::Int64
    λ::Vector{Float64}
    w::Vector{Float64}
    rankA::Int64
    rankJ2::Int64
    dimA::Int64
    dimJ2::Int64
    b_gn::Vector{Float64}
    d_gn::Vector{Float64}
    predicted_reduction::Float64
    progress::Float64
    grad_res::Float64
    speed::Float64
    β::Float64
    restart::Bool
    first::Bool
    add::Bool
    del::Bool
    index_del::Int64
    code::Int64
    nb_newton_steps::Int64
end


Base.copy(s::Iteration) = Iteration(s.x, s.p, s.rx, s.cx, s.t, s.α, s.index_α_upp, s.λ, s.w, s.rankA, s.rankJ2, s.dimA, s.dimJ2, s.b_gn, s.d_gn, 
s.predicted_reduction, s.progress, s.grad_res, s.speed, s.β, s.restart, s.first, s.add, s.del, s.index_del, s.code, s.nb_newton_steps)