# Problem 42 from Hock-Schittkowski collection

using BenchmarkTools
include("../src/enlsip_functions.jl")

n = 4
m = 4
nb_eq = 2
nb_constraints = 2

res = ResidualsEval(0)

function (res::ResidualsEval)(x::Vector{Float64}, rx::Vector{Float64}, J::Matrix{Float64})

    # Evaluate the residuals
    if abs(res.ctrl) == 1
        rx[:] = [x[1] - 1.0;
        x[2] - 2.0;
        x[3] - 3.0;
        x[4] - 4.0]

    # The jacobian is computed analytically
    elseif res.ctrl == 2
        res.ctrl = 0
    end
    return
end

cons = ConstraintsEval(0)

function (cons::ConstraintsEval)(x::Vector{Float64}, cx::Vector{Float64}, A::Matrix{Float64})

    # Evaluate the constraints
    if abs(cons.ctrl) == 1
        cx[:] = [x[1] - 2.0;
                 x[3]^2 + x[4]^2 - 2.0]

    # The jacobian is computed numerically if ctrl is set to 0 on return
    elseif cons.ctrl == 2
        cons.ctrl = 0
    end
    return
end

x0 = [1.0;1.0;1.0;1.0]
@time enlsip(x0,res,cons,n,m,nb_eq,nb_constraints)
