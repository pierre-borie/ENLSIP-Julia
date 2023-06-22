using ForwardDiff

include("../src/enlsipStructures/evaluationFunctions.jl")

function r(x::Vector)
    return [x[1]-x[2], x[2]-x[3], (x[3]-x[4])^2 ,(x[4]-x[5])^2]
end

J = x -> ForwardDiff.jacobian(r,x)

function jac(x::Vector)
    return ForwardDiff.jacobian(r,x)
end

typeof(J) <: Function
typeof(jac)

test_res = EvaluationFunction{typeof(r),typeof(J)}(r,J,0,0)

ctrl = 1


n = 5
m = 4
nb_eq = 3
nb_constraints = 3

x0 = [1.,0.,1.,0.,0.]
rx = zeros(m)
Jx = zeros(m,n)

func_eval!(test_res,x0,rx,Jx,ctrl)

rx
Jx
ctrl = 2

func_eval!(test_res,x0,rx,Jx,ctrl)
Jx
