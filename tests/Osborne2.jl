using BenchmarkTools, ForwardDiff, Distributions

include("../src/enlsip_functions.jl")

# Parameters

n = 11
m = 65 
nb_eq = 0
nb_constraints = 22

# DataPoints

dataset = [1 0.0 1.366 ;2 0.1 1.191 ;3 0.2 1.112 ;4 0.3 1.013 ;5 0.4 0.991 ;6 0.5 0.885 ;7 0.6 0.831 ;8 0.7 0.847 ;9 0.8 0.786 ;10 0.9 0.725 ;11 1.0 0.746 ;
12 1.1 0.679 ;13 1.2 0.608 ;14 1.3 0.655 ;15 1.4 0.616 ;16 1.5 0.606 ;17 1.6 0.602 ;18 1.7 0.626 ;19 1.8 0.651 ;20 1.9 0.724 ;21 2.0 0.649 ;22 2.1 0.649 ;23 2.2 0.694 ;24 2.3 0.644 ;25 2.4 0.624 ;
26 2.5 0.661 ;27 2.6 0.612 ;28 2.7 0.558 ;29 2.8 0.533 ;30 2.9 0.495 ;31 3.0 0.500 ;32 3.1 0.423 ;33 3.2 0.395 ;34 3.3 0.375;35 3.4 0.538 ;36 3.5 0.522 ;37 3.6 0.506 ;38 3.7 0.490 ;
39 3.8 0.478 ;40 3.9 0.467 ;41 4.0 0.457 ;42 4.1 0.457 ;43 4.2 0.457 ;44 4.3 0.457 ;45 4.4 0.457 ;46 4.5 0.457 ;47 4.6 0.457 ;48 4.7 0.457 ;49 4.8 0.457 ;50 4.9 0.457 ;51 5.0 0.457;52 5.1 0.431 ;53 5.2 0.431 ;
54 5.3 0.424 ;55 5.4 0.420 ;56 5.5 0.414 ;57 5.6 0.411 ;58 5.7 0.406 ;59 5.8 0.406 ;60 5.9 0.406 ;61 6.0 0.406 ;62 6.1 0.406 ;63 6.2 0.406 ;64 6.3 0.406 ;65 6.4 0.406]

t = dataset[1:m,2]
y = dataset[1:m,3]


# Residuals 

function r_k(x::Vector, t::Float64, y::Float64)
    rx = x[1]*exp(-x[5]*t) + x[2]*exp(-x[6]*(t-x[9])^2) + x[3]*exp(-x[7]*(t-x[10])^2) + x[4]*exp(-x[8]*(t-x[11])^2)
    return y - rx
end

function r(x::Vector)
    return [r_k(x,t[k],y[k]) for k=1:m]
end

resOsborne2 = ResidualsEval(0)

function (resOsborne2::ResidualsEval)(x::Vector, rx::Vector, J::Matrix)

    # Evaluate the residuals
    if abs(resOsborne2.ctrl) == 1
        rx[:] = r(x)

    # The jacobian is computed analytically
    elseif resOsborne2.ctrl == 2
        J[:] = ForwardDiff.jacobian(r,x)
    end
    return
end

# Constraints

function c(x::Vector)
    res = [x[1] - 1.31; 1.4 - x[1];
            x[2] - 0.4314 ; 0.8 - x[2];
            x[3] - 0.6336; 1.0 - x[3];
            x[4] - 0.5; 1.0 - x[4];
            x[5] - 0.5; 1.0 - x[5];
            x[6] - 0.6; 3.0 - x[6];
            x[7] - 1.0; 5.0 - x[7];
            x[8] - 4.0; 7.0 - x[8];
            x[9] - 2.0; 2.5 - x[9];
            x[10] - 4.5689; 5.0 - x[10];
            x[11] - 5.0; 6.0 - x[11]]
    return res
end
    
consOsborne2 = ConstraintsEval(0)

function (consOsborne2::ConstraintsEval)(x::Vector, cx::Vector, A::Matrix)

    # Evaluate the constraints
    if abs(consOsborne2.ctrl) == 1
        cx[:] = c(x)
    # The jacobian is computed numerically if ctrl is set to 0 on return
    elseif consOsborne2.ctrl == 2
        A[:] = ForwardDiff.jacobian(c,x)
    end
    return
end

# Starting randomly generated
function generate_starting_point()
    x0 = [rand(Uniform(1.31,1.4));
        rand(Uniform(0.4314 , 0.8));
        rand(Uniform(0.6336, 1.0));
        rand(Uniform(0.5, 1.0));
        rand(Uniform(0.5, 1.0));
        rand(Uniform(0.6, 3.0));
        rand(Uniform(1.0, 5.0));
        rand(Uniform(4.0, 7.0));
        rand(Uniform(2.0, 2.5));
        rand(Uniform(4.5689, 5.0));
        rand(Uniform(5.0, 6.0))]
    return x0
end


# ENLSIP
x_saved =  [1.3344098963722457; 0.5572842161127423 ; 0.6757364753061974; 0.8291980513226953 ;0.9233565833014519 ;0.9588470511477797; 1.9610314699563896 ;4.055321823656234; 2.048625993866472 ;4.60296578920499; 5.95212572157736]
# x0 = generate_starting_point()
x0 = x_saved
enlsip(x0,resOsborne2,consOsborne2,n,m,nb_eq,nb_constraints,verbose=true)