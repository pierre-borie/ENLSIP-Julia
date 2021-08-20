include("../src/enlsip_functions.jl")
using BenchmarkTools

n = 2
m = 44
nb_eq = 0 # nombre de contraintes d'égalité
nb_constraints = 3 # nombre de contraintes d'égalité et d'inégalité

a = [8.,8.,10.,10.,10.,10.,12.,12.,12.,12.,14.,14.,14.,16.,16.,16.,18.,18.,20.,20.,20.,22.,22.,22., 
24.,24.,24.,26.,26.,26.,28.,28.,30.,30.,30.,32.,32.,34.,36.,36.,38.,38.,40.,42.]

b = [.49,.49,.48,.47,.48,.47,.46,.46,.45,.43,.45,.43,.43,.44,.43,.43,.46,.45,.42,.42,.43,.41,
.41,.40,.42,.40,.40,.41,.40,.41,.41,.40,.40,.40,.38,.41,.40,.40,.41,.38,.40,.40,.39,.39]


# Résidus et matrice jacobienne associée

r_i(x::Vector,t::Float64) = x[1] + (0.49 - x[1]) * exp(-x[2]*(t - 8))
res57 = ResidualsEval(0)

function (res57::ResidualsEval)(x::Vector{Float64},rx::Vector,J::Matrix{Float64})
    # Evaluate the residuals
    if abs(res57.ctrl) == 1
        rx[:] = b - (t::Float64 -> r_i(x,t)).(a)

    # The jacobian is computed numericaly using forward differences
    # ctrl is set to 0
    elseif res57.ctrl == 2 res57.ctrl = 0 end
    return
end

# Contraintes et matrice jacobienne associée

cons57 = ConstraintsEval(0)

function (cons57::ConstraintsEval)(x::Vector{Float64},cx::Vector{Float64},A::Matrix{Float64})
    # Evaluate the constraints
    if abs(cons57.ctrl) == 1
        cx[:] = [0.49 * x[2] - x[1] * x[2] - 0.09, x[1] - 0.4, x[2] + 4]
    
    # The jacobian is computed anaticaly
    elseif cons57.ctrl == 2
        A[:] = [-x[2] 0.49-x[1];
        1.0 0.0;
        0.0 1.0]
    end
end

x0 = [0.42, 5.0]

enlsip(x0,res57,cons57,n,m,nb_eq,nb_constraints)