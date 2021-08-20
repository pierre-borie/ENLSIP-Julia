include("../src/enlsip_functions.jl")
using BenchmarkTools

n = 3
m = 3
nb_eq = 0
nb_constraints = 7

res65 = ResidualsEval(0)

function (res65::ResidualsEval)(x::Vector{Float64}, rx::Vector{Float64}, J::Matrix{Float64})

    # Evaluate the residuals
    if abs(res65.ctrl) == 1
        rx[:] = [x[1] - x[2]; (x[1]+x[2]-10.0) / 3.0; x[3]-5.0]

    # The jacobian is computed analytically
    elseif res65.ctrl == 2
        J[:] = [1. -1. 0;
                1/3 1/3 0.;
                0. 0. 1.]
    end
    return
end

cons65 = ConstraintsEval(0)

function (cons65::ConstraintsEval)(x::Vector{Float64}, cx::Vector{Float64}, A::Matrix{Float64})

    # Evaluate the constraints
    if abs(cons65.ctrl) == 1
        cx[:] = [48.0 - x[1]^2-x[2]^2-x[3]^2;
                 x[1]+4.5;
                 x[2]+4.5;
                 x[3]+5.0;
                 -x[1]+4.5;
                 -x[2]+4.5;
                 -x[3]+5.0]
    # The jacobian is computed numerically if ctrl is set to 0 on return
    elseif cons65.ctrl == 2
        cons65.ctrl = 0
    end
    return
end

x0 = [-5.0;5.0;0.0]
enlsip(x0,res65,cons65,n,m,nb_eq,nb_constraints)