include("./enlsip-julia-0.2.0.jl")


# Dimensions and parameters of the problem
n = 2
m = 129
h = 5.0
σ = 0.001
nb_constraints = 2
nb_eq = 0

# Data 
t = [1.0 + (i-1)/h for i=1:m]
y = (t_i::Float64 -> 0.45 -1.15*exp(-0.0115*t_i) + 1.85*exp(-0.0225*t_i)).(t) + σ*randn(m) 

# Definition of residuals and constraints

r_i(x::Vector,t::Float64) = 0.45 -1.15*exp(x[1]*t) + 1.85*exp(x[2]*t)
res = ResidualsEval(0)

function (res::ResidualsEval)(x::Vector,rx::Vector,J::Matrix)
    # Evaluate the residuals
    if abs(res.ctrl) == 1
        rx[:] = y - (t_i::Float64 -> r_i(x,t_i)).(t)

    # The jacobian is computed numericaly using forward differences
    # ctrl is set to 0
    elseif res.ctrl == 2 res.ctrl = 0 end
    return
end

# Contraintes et matrice jacobienne associée

cons = ConstraintsEval(0)

function (cons::ConstraintsEval)(x::Vector,cx::Vector,A::Matrix)
    # Evaluate the constraints
    if abs(cons.ctrl) == 1
        cx[:] = [-x[1];-x[2]]
    # The jacobian is computed anaticaly
    elseif cons.ctrl == 2 cons.ctrl = 0 end
    return
end

x0 = zeros(n)
enlsip_020(x0,res,cons,n,m,nb_eq,nb_constraints)