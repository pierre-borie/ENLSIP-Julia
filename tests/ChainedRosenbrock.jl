using BenchmarkTools, ForwardDiff, Distributions

include("../src/enlsip_functions.jl")

# Data
n = 1000
m = 2(n-1)
nb_eq = n-2
nb_constraints = n-2

# Residuals

function r(x::Vector)
    n = length(x)
    m = 2(n-1)
    rx = Vector(undef,m)
    rx[1:n-1] = [10(x[i]^2 - x[i]) for i=1:n-1]
    rx[n:m] = [x[k-n+1] - 1 for k=n:m]
    return rx
end

resCR = ResidualsEval(0)

function (resCR::ResidualsEval)(x::Vector, rx::Vector, J::Matrix)

    # Evaluate the residuals
    if abs(resCR.ctrl) == 1
        rx[:] = r(x)

    # The jacobian is computed analytically
    elseif resCR.ctrl == 2
        J[:] = ForwardDiff.jacobian(r,x)
    end
    return
end

# Constraints
function c(x::Vector)
    n = length(x)
    cx = [3x[k+1]^3 + 2x[k+2] - 5 + sin(x[k+1]-x[k+2])*sin(x[k+1]+x[k+2]) + 4x[k+2] - 
        x[k]*exp(x[k]-x[k+1]) - 3 for k=1:n-2]
    return cx
end
    
consCR = ConstraintsEval(0)

function (consCR::ConstraintsEval)(x::Vector, cx::Vector, A::Matrix)

    # Evaluate the constraints
    if abs(consCR.ctrl) == 1
        cx[:] = c(x)
    # The jacobian is computed numerically if ctrl is set to 0 on return
    elseif consCR.ctrl == 2
        A[:] = ForwardDiff.jacobian(c,x)
    end
    return
end

x0 = [(mod(i,2) == 1 ? -1.2 : 1.0) for i=1:n]

enlsip(x0,resCR,consCR,n,m,nb_eq,nb_constraints)