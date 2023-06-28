include("structures.jl")
include("enlsip_functions.jl")

function instantiate_evalfunc(res_func, jacres_func, cons_func, jaccons_func)

    r = ResidualsFunction(res_func, jacres_func)
    c = ConstraintsFunction(cons_func, jaccons_func)
    return r,c
end


# Evaluation functions and jacobian matrices functions are given
function enlsip(
    x0::Vector,
    res_func,
    jacres_func,
    cons_func,
    jaccons_func,
    nb_res::Int64,
    nb_eq::Int64,
    nb_ineq::Int64;
    verbose::Bool=false,
    max_iter::Int64=100)

    @assert typeof(res_func) <: Function
    @assert typeof(jacres_func) <: Function
    @assert typeof(cons_func) <: Function
    @assert typeof(jaccons_func) <: Function

    @assert all(map(typeof,[res_func, jacres_func, cons_func, jaccons_func]), T -> T <: Function)
    
    residuals, constraints = instantiate_evalfunc(res_func, jacres_func, cons_func, jaccons_func)

    n = size(x0)
    total_constraints = nb_eq + nb_ineq

    sol = enlsip_solve(x0, residuals, constraints, n, nb_res, nb_eq, total_constraints, MAX_ITER=max_iter, verbose=verbose)
    return sol
end
