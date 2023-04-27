using BenchmarkTools, ForwardDiff, Distributions

include("../src/enlsip_functions.jl")

# Parameters
n = 1000 # 1000 20, needs to be >= 8
m = 6 * (div(n,2)-1)
nb_eq = n-7
nb_constraints = nb_eq

# Residuals

function r(x::Vector)
    n = length(x)
    N = div(n,2) - 1
    s = √(10)
    
    rx1 = [10(x[2i-1]^2 - x[2i]) for i=1:N]
    rx2 = [x[2i-1] - 1 for i=1:N]
    rx3 = [3s*(x[2i+1]^2 - x[2i+2]) for i=1:N]
    rx4 = [x[2i+1]-1 for i=1:N]
    rx5 = [s*(x[2i] + x[2i+2] - 2) for i=1:N]
    rx6 = [(x[2i] - x[2i+2])*(1/s) for i=1:N]
    
    return [rx1;rx2;rx3;rx4;rx5;rx6]
end

resCW = ResidualsEval(0)


#Constraints

function (resCW::ResidualsEval)(x::Vector, rx::Vector, J::Matrix)

    # Evaluate the residuals
    if abs(resCW.ctrl) == 1
        rx[:] = r(x)

    # The jacobian is computed analytically
    elseif resCW.ctrl == 2
        J[:] = ForwardDiff.jacobian(r,x)
    end
    return
end

function c(x::Vector)
    n = length(x)
    cx = [(2+5x[k+5]^2)*x[k+5] + 1 + sum(x[i]*(1+x[i]) for i=max(k-5,1):k+1) for k=1:n-7]
    return cx
end
    
consCW = ConstraintsEval(0)

function (consCW::ConstraintsEval)(x::Vector, cx::Vector, A::Matrix)

    # Evaluate the constraints
    if abs(consCW.ctrl) == 1
        cx[:] = c(x)
    # The jacobian is computed numerically if ctrl is set to 0 on return
    elseif consCW.ctrl == 2
        A[:] = ForwardDiff.jacobian(c,x)
    end
    return
end

x0 = [(mod(i,2) == 1 ? -2. : 1.) for i=1:n]
enlsip(x0,resCW,consCW,n,m,nb_eq,nb_constraints)