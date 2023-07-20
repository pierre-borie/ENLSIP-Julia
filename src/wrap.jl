using LinearAlgebra, Polynomials, Printf, Polynomials
using Formatting, DataFrames, CSV

include("structures.jl")
include("enlsip_functions.jl")

export solve


"""
    solve(model::EnlsipModel)

Function to solve an instance of EnlsipModel
"""
function solve(model::EnlsipModel; silent::Bool=false, max_iter::Int64 = 100, scaling::Bool=false)
    sol = enlsip(model.x0, model.residuals, model.constraints, model.nb_parameters, model.nb_residuals, model.nb_eqcons, model.nb_cons, verbose=!silent, scaling=scaling, MAX_ITER=max_iter)
    return sol
end