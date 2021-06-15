# Summarizes the useful informations about an iteration of the algorithm

mutable struct Iteration
    x::Vector
    p::Vector
    rx::Vector
    cx::Vector
    t::Int64
    α::Float64
    λ::Vector
    w::Vector
    rankA::Int64
    rankJ2::Int64
    dimA::Int64
    dimJ2::Int64
    b_gn::Vector
    d_gn::Vector
    predicted_reduction::Float64
    progress::Float64
    β::Float64
    restart::Bool
    first::Bool
    add::Bool
    del::Bool
    code::Int64
end

Base.copy(s::Iteration) = Iteration(s.x, s.p, s.rx, s.cx, s.t, s.α, s.λ, s.w, s.rankA, s.rankJ2, s.dimA, s.dimJ2, s.b_gn, s.d_gn, s.predicted_reduction, s.progress, s.β, s.restart, s.first, s.add, s.del, s.code)

function show_iter(step::Iteration)
    if step.code == 2
        phase = "Newton"
    elseif step.code == -1
        phase = "Subspace Minimization"
    else
        phase = "Gauss-Newton"
    end

    println("\nMéthode : $phase")
    println("Departure point : $(step.x)")
    println("Search direction : $(step.p)")
    println("Lagrange multipliers : $(step.λ)")
    println("Penalty weights : $(step.w)")
    println("Steplength : $(step.α)")
    println("Next point : $(step.x + step.α * step.p)")
    println("dimA = $(step.dimA); dimJ2 = $(step.dimJ2)")
    println("rankA = $(step.rankA); rankJ2 = $(step.rankJ2)")
 #   println("b = $(step.b_gn); d = $(step.d_gn)")
    println("\n")
    
end
    

# Reprensents the useful informations about constraints at a point x, i.e. : 
# cx : constraint function evaluation 
# A : constraint jacobian evaluation 

# Used to distinguish active constraints

mutable struct Constraint
    cx::Vector
    A::Matrix
end

# In ENLSIP, the working-set is a prediction of the set of active constraints at the solution
# It is updated a every iteration thanks to a Lagrangian multipliers evaluation 

# This mutable struc ummarizes infos about the qualification of the constraints, i.e. :
# q : number of equality constraints
# t : number of constraints considered to be active (all equalities and some inequalities)
# l : total number of constraints (equality and inequality)
# active : indeces of the constraints considered as active (total length : l)
# inactive : indeces of the inequality constraints considered inactive (total length : l-t)


mutable struct WorkingSet
    q::Int64
    t::Int64
    l::Int64
    active::Vector{Int64}
    inactive::Vector{Int64}
end

function show_working_set(w::WorkingSet)
    s1 = "$(w.q) equalities, $(w.l-w.q) inequalities\n"
    s2 = "active : $(w.active[1:w.t])\n"
    s3 = "inactive : $(w.inactive[1:w.l-w.t])\n"
    s = string(s1,s2,s3)
    println(s)
end

# Computes and returns the rank of a triangular matrix T using its diagonal elements placed in decreasing order
# according to their absolute value

# diag_T is the diagonal of the Triangular matrix T whose rank is estimated
# τ is the relative tolerance to estimate the rank

function pseudo_rank(diag_T::Vector, τ::Float64 = sqrt(eps(Float64)))
    if isempty(diag_T) || abs(diag_T[1]) < τ
        r = 0
    else
        r = 1
        for j in eachindex(diag_T)
            if max(abs(diag_T[j] / diag_T[1]), abs(diag_T[j])) >= τ
                r = j
            end
        end
    end
    return r
end

# SUBDIR
# Computes a search direction with Gauss-Newton method


function sub_search_direction(
        J1::Matrix,
        J2::Matrix,
        rx::Vector,
        cx::Vector,
        Q1,
        P1::Matrix,
        L11::Matrix,
        Q2,
        P2::Matrix,
        R11::Matrix,
        Q3,
        R22::Matrix,
        P3::Matrix,
        n::Int64,
        t::Int64,
        dimA::Int64,
        dimJ2::Int64,
        code::Int64)

    # Résolution sans stabilisation
    if code == 1
        b = -transpose(P1) * cx
        δp1 = LowerTriangular(L11) \ b
        p1 = δp1

        d = - transpose(Q3) * (J1*p1 + rx)
        δp2 = UpperTriangular(R22[1:dimJ2,1:dimJ2]) \ d[1:dimJ2]
        p2 = P3 * [δp2; zeros(n-t-dimJ2)]

    # Résolution avec stabilisation
    elseif code == -1
        b = - transpose(Q2) * transpose(P1) * cx
        δp1 = UpperTriangular(R11[1:dimA,1:dimA]) \ b[1:dimA]
        p1 = P2 * [δp1; zeros(t-dimA)]

        d = - transpose(Q3) * (J1*p1 + rx)
        δp2 = UpperTriangular(R22[1:dimJ2, 1:dimJ2]) \ d[1:dimJ2]
        p2 = P3 * [δp2; zeros(n-t-dimJ2)]
    end

    p = Q1 * [p1;p2]
    return p, b, d
end

# GNSRCH
# Compute the search direction with the method of Gauss-Newton

function gn_search_direction(
    A::Matrix,
    J::Matrix,
    rx::Vector,
    cx::Vector,
    Q1,
    P1::Matrix,
    L11::Matrix,
    Q2,
    P2::Matrix,
    R11::Matrix,
    rankA::Int64,
    t::Int64,
    τ::Float64,
    current_iter::Iteration)
    code = (rankA == t ? 1 : -1)
    (m,n) = size(J)
    JQ1 = J*Q1
    J1, J2 = JQ1[:,1:t], JQ1[:,t+1:end]
    F_J2 = qr(J2, Val(true))
    Q3, P3, R22 = F_J2.Q, F_J2.P, F_J2.R
    rankJ2 = pseudo_rank(diag(R22), τ)
    p_gn, b_gn, d_gn = sub_search_direction(J1, J2,rx,cx,Q1,P1,L11,Q2,P2,R11,Q3,R22,P3,n,t,rankA,rankJ2,code)
    current_iter.rankA = rankA
    current_iter.rankJ2 = rankJ2
    current_iter.dimA = rankA
    current_iter.dimJ2 = rankJ2
    current_iter.b_gn = b_gn
    current_iter.d_gn = d_gn
    return p_gn

end

# NEWTON
# Computes the search direction by using the method of Newton

function newton_search_direction(
        x::Vector,
        c::Function,
        r::Function,
        active_cx::Vector,
        active::Vector,
        t::Int64,
        λ::Vector,
        rx::Vector,
        J::Matrix,
        m::Int64,
        n::Int64,
        Q1,
        P1::Matrix,
        L11::Matrix,
        Q2,
        R11::Matrix,
        P2::Matrix,
        rankA::Int64)

    if t == rankA
        b = -transpose(P1) * active_cx
        p1 = LowerTriangular(L11) \ b
     elseif t > rankA
        b = -transpose(Q2) * transpose(P1) * active_cx
        δp1 = UpperTriangular(R11) \ b
        p1 = P2 * δp1
    end

    if rankA == n return p1 end
    
    # Computation of J1, J2
    JQ1 = J*Q1
    J1, J2 = JQ1[:,1:t], JQ1[:,t+1:end]
   
    
    # Computation of lagrangian hessian

    res_mat, cons_mat = zeros(n,n), zeros(n,n)
    # Residual Hessian
    for i=1:m
        hess_ri(x) = ForwardDiff.hessian(x -> r(x)[i], x)
        res_mat += rx[i] * hess_ri(x)
    end
    
    # Active constraints hessians
    for i=1:t
        j = active[i]
        hess_cj(x) = ForwardDiff.hessian(x -> c(x)[j], x)
        cons_mat += λ[i] * hess_cj(x)
    end
    Γ_mat = res_mat - cons_mat

    if rankA == t
        E = transpose(Q1) * Γ_mat * Q1
    elseif t > rankA
        E = transpose(P2) * transpose(Q1) * Γ_mat * Q1 * P2
    end


    # Forms the system to compute p2
    E21 = E[t+1:n, 1:t]
    E22 = E[t+1:n, t+1:n]

    W22 = E22 + transpose(J2)*J2
    W21 = E21 + transpose(J2)*J1

    d = -W21 * p1 - transpose(J2) * rx

    if isposdef(W22)
        chol_W22 = cholesky(Symmetric(W22))
        y = chol_W22.L \ d
        p2 = chol_W22.U \ y
        p = Q1 * [p1;p2]
    else
        p = zeros(n)
    end
    return p
end

# MULEST
# Compute first order estimate of Lagrange multipliers

function first_lagrange_mult_estimate!(A::Matrix, λ::Vector, ∇fx::Vector, cx::Vector)
    # Solves the system A^T * λ_ls = ∇f(x) using qr factorisation of A^T
    # A^T*P1 = Q1 * (R)
    #              (0)
    # with R^T = L11
    # then computes estimates of lagrage multipliers by forming :
    #                  -1
    # λ = λ_ls - (A*A^T) *cx
    
    (t, n) = size(A)
    v = zeros(t)
    vnz = zeros(t)
    F = qr(transpose(A), Val(true))
    prankA = pseudo_rank(diag(F.R))
    b = transpose(F.Q) * ∇fx
    v[1:prankA] = UpperTriangular(F.R[1:prankA,1:prankA]) \ b[1:prankA]
    if prankA < t
        v[prankA+1:t] = zeros(t - prankA)
    end
    λ_ls = F.P * v
    
    # Compute the nonzero first order lagrange multiplier estimate by forming
    #                  -1
    # λ = λ_ls - (A*A^T) *cx

    b = -transpose(F.P) * cx
    y = zeros(t)
    #                -1
    # Compute y =(L11) * b
    y[1:prankA] = LowerTriangular(transpose(F.R)[1:prankA,1:prankA]) \ b[1:prankA]
    #              -1
    # Compute u = R  * y
    u = zeros(t)
    u[1:prankA] = UpperTriangular(F.R[1:prankA,1:prankA]) \ y[1:prankA]
    λ[:] = λ_ls + F.P * u
    return
end


# Compute second order least squares estimate of Lagrange multipliers
function second_lagrange_mult_estimate!(
    A::Matrix,
    J::Matrix,
    λ::Vector,
    rx::Vector,
    p_gn::Vector,
)

    # Solves the system A^T * λ = Jx^T(r(x) + Jx*p_gn))
    (t, n) = size(A)
    F = qr(transpose(A), Val(true))
    J1 = (J*F.Q)[:, 1:t]
    b = transpose(J1) * (rx + J * p_gn)
    v = UpperTriangular(F.R) \ b
    λ[:] = F.P * v    
    return

end

# Equivalent Fortran : DELETE in dblreduns.f

function delete_constraint!(W::WorkingSet,s::Int64)

    l,t = W.l, W.t

    # Ajout de la contrainte à l'ensemble inactif
    W.inactive[l-t+1] = W.active[s]
    sort!(@view W.inactive[1:l-t+1])

    # Réorganisation de l'ensemble actif
    for i = s:t-1
        W.active[i] = W.active[i+1]
    end
    W.active[t] = 0
    W.t -= 1
    return
end

# Equivalent Fortran : ADDIT in dblreduns.f

function add_constraint!(W::WorkingSet, s::Int64)

    l,t = W.l, W.t
    # Ajout de la contrainte à l'ensemble actif
    W.active[t+1] = W.inactive[s]
    sort!(@view W.active[1:t+1])

    # Réorganisation de l'ensemble inactif

    for i = s:l-t-1
        W.inactive[i] = W.inactive[i+1]
    end
    W.inactive[l-t] = 0
    W.t += 1
    return
end

# Equivalent Fortran : SIGNCH in dblreduns.f
# Returns the index of the constraint that has to be deleted from the working set
# Obtainted with the lagrange mulitpliers estimates

function check_constraint_deletion(
    q::Int64,
    A::Matrix,
    λ::Vector,
    ∇fx::Vector)

    t = length(λ)
    δ = 10
    τ = 0.5
    sq_rel = sqrt(eps(Float64))
    s = 0
    if t > q
        e = 0
        for i = q+1:t
            row_i = norm(A[i,:])
            if row_i * λ[i] <= -sq_rel && row_i * λ[i] <= e

                e = row_i * λ[i]
                s = i
            end
        end
        grad_res = norm(transpose(A) * λ - ∇fx)
        if grad_res > -e * δ
            s = 0
        end
    end
    return s
end

# EVADD
# Move violated constraints to the working set

function evaluate_violated_constraints(
        cx::Vector,
        W::WorkingSet)
    
    # Data
    ε = sqrt(eps(Float64))
    added = false
    if W.l > W.t
        i = 1
        while i <= W.l - W.t
            k = W.inactive[i]
            if cx[k] < ε
                add_constraint!(W, i)
                added = true
            else
                i += 1
            end
        end
    end
    return added
end 

# WRKSET
# Estimate the lagrange multipliers and eventually delete a constraint from the working set
# Compute the search direction using Gauss-Newton method

function update_working_set!(
    W::WorkingSet,
    rx::Vector,
    C::Constraint,
    ∇fx::Vector,
    J::Matrix,
    p_gn::Vector,
    iter_k::Iteration)

    λ = Vector{Float64}(undef, W.t)
    ε_rank = sqrt(eps(Float64))
    first_lagrange_mult_estimate!(C.A, λ, ∇fx,C.cx)
    s = check_constraint_deletion(W.q, C.A, λ, ∇fx)

    # Constraint number s is deleted from the current working set
    if s != 0
        deleteat!(λ,s)
        deleteat!(C.cx,s)
        delete_constraint!(W, s)
        C.A = C.A[setdiff(1:end,s),:]
        F_A = qr(transpose(C.A), Val(true))
        L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q, F_A.P
        rankA = pseudo_rank(diag(L11), ε_rank)
        F_L11 = qr(L11, Val(true))
        R11, Q2, P2 = F_L11.R, F_L11.Q, F_L11.P
        p_gn[:] = gn_search_direction(C.A,J,rx,C.cx,Q1,P1,L11,Q2,P2,R11,rankA,W.t,ε_rank,iter_k)

    # No first order estimate implies deletion of a constraint
    elseif s == 0
        F_A = qr(transpose(C.A), Val(true))
        L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q, F_A.P
        rankA = pseudo_rank(diag(L11), ε_rank)
        F_L11 = qr(L11, Val(true))
        R11, Q2, P2 = F_L11.R, F_L11.Q, F_L11.P
        p_gn[:] = gn_search_direction(C.A,J,rx,C.cx,Q1,P1,L11,Q2,P2,R11,rankA,W.t,ε_rank,iter_k)
        second_lagrange_mult_estimate!(C.A,J,λ,rx,p_gn)
        if (W.t == rankA)
            s = check_constraint_deletion(W.q, C.A, λ, ∇fx)
            if s != 0
                deleteat!(λ,s)
                C.cx = C.cx[setdiff(1:end,s)]
                delete_constraint!(W, s)
                C.A = A[setdiff(1:end,s),:]
                F_A = qr(transpose(A), Val(true))
                L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q, F_A
                rankA = pseudo_rank(diag(L11), ε_rank)
                F_L11 = qr(L11, Val(true))
                R11, Q2, P2 = F_L11.R, F_L11.Q, F_L11.P
                p_gn[:] = gn_search_direction(C.A,J,rx,C.cx,Q1,P1,L11,Q2,P2,R11,rankA,W.t,ε_rank,iter_k)
            end
        end
    end
    iter_k.λ = λ
    return
end

# INIALC
# Compute the first working set and penalty constants

function init_working_set(cx::Vector, K::Array{Array{Float64,1},1}, step::Iteration, q::Int64,l::Int64)
    δ, ϵ, ε_rel = 0.1, 0.01, sqrt(eps(Float64))

    # Initialisation des pénalités
    K[:] = [δ * ones(l) for i=1:length(K)]
    for i=1:l
        pos = min(abs(cx[i]) + ϵ, δ)
        step.w[i] = pos
    end

    # Determination du premier ensemble actif
    active = zeros(Int64, l); inactive = zeros(Int64, l - q)
    t = q; lmt = 0

    # Les contraintes d'égalité sont toujours actives
    active[1:q] = [i for i=1:q]

    for i = q+1:l
        if cx[i] <= ε_rel
            t += 1; active[t] = i
        else
            lmt += 1; inactive[lmt] = i
        end
    end
    step.t = t
    first_working_set = WorkingSet(q, t, l, active, inactive)
    return first_working_set
end

# PRESUB
# Returns dimension when previous descent direction was computed with subspace minimization

function subspace_min_previous_step(
    τ::Vector,
    ρ::Vector,
    ρ_prk::Float64,
    c1::Float64,
    pseudo_rank::Int64,
    previous_dimR::Int64,
    progress::Float64,
    predicted_linear_progress::Float64,
    prelin_previous_dim::Float64,
    previous_α::Float64)

    # Data

    stepb, pgb1, pgb2, predb, rlenb, c2 = 2e-1, 3e-1, 1e-1, 7e-1, 2.0, 1e2

    if ((previous_α < step_τ) && 
        (progress <= pgb1 * predicted_linear_progress^2) &&
        (progress <= pgb2 * prelin_previous_dim^2))
        
        # Bad step
        dim = max(1, previous_dimR-1) 
        if ((previous_dimR > 1) && (ρ[dim] > c1 * ρ_prk))
            suggested_dim = dim
        end

    else
        dim = previous_dimR
        if (((ρ[dim] > predb * ρ_prk) && (rlenb * τ[dim] < τ[dim+1])) ||
            (c2 * τ[dim] < τ[dim+1]))
            suggested_dim = dim
        else
            i1 = previous_dimR-1
            buff = [i for i = i1:previous_dimR if ρ[i] > predb * ρ_prk]
            suggested_dim = (isempty(buff) ? pseudo_rank : min(buff))
        end
    end
    return suggested_dim
end

# PREGN
# Returns dimension to use when previous descent direction was computed with Gauss-Newton method

function gn_previous_step(
    τ::Vector,
    τ_prk::Float64,
    mindim::Int64,
    ρ::Vector,
    ρ_prk::Float64,
    pseudo_rank::Int64)

    # Data
    τ_max, ρ_min = 2e-1, 5e-1
    pm1 = pseudo_rank - 1
    if mindim > pm1
        suggested_dim = mindim
    else
        k = pm1
        while (τ[k] >= τ_max*τ_prk || ρ[k] <= ρ_min*ρ_prk) && k > mindim
            k -= 1
        end

        suggested_dim = (k > mindim ? k : max(mindim, pm1))
    end

    return suggested_dim
end

# GNDCHK 

# Decides what method should be used to compute the search direction
# This information is told by the value returned by method_code : 
# 1 if Gauss-Newton search direction is accepted
# -1 if subspace inimization is suggested
# 2 if the method of Newton is suggested

# β_k = sqrt(||b1||^2 + ||d1||^2) is an information used to compute the convergence rate

function check_gn_direction(
    b1nrm::Float64,
    d1nrm::Float64,
    d1nrm_as_km1::Float64,
    dnrm::Float64,
    active_c_sum::Float64,
    iter_number::Int64,
    rankA::Int64,
    n::Int64,
    m::Int64,
    restart::Bool,
    constraint_added::Bool,
    constraint_deleted::Bool,
    W::WorkingSet,
    cx::Vector,
    λ::Vector,
    iter_km1::Iteration)
    
    δ = 1e-1
    c1, c2, c3, c4, c5 = 5e-1, 1e-1, 4e0, 1e1, 5e-2
    ε_rel = eps(Float64)
    β_k = sqrt(d1nrm + b1nrm)
    
    method_code = 1
    cond1 = (iter_number == 1 || constraint_added || constraint_deleted)
    cond2 = (β_k < c1 * iter_km1.β)
    cond3 = ((iter_km1.progress > c2 * iter_km1.predicted_reduction) && ((dnrm <= c3 * β_k)))
    if !(cond1 || cond2 || cond3)
        method_code = -1
        non_linearity_k = sqrt(d1nrm*d1nrm + active_c_sum)
             non_linearity_km1 = sqrt(d1nrm_as_km1 + active_c_sum)
        to_reduce = false
        if W.q < W.t
            to_reduce = (to_reduce || any(<(0), λ[W.q+1:W.t]))
        end
        if (W.l-W.t > 0)
            inact_c = [cx[W.inactive[j]] for j = 1:((W.l-W.t))]
            to_reduce = (to_reduce || any(<(δ), inact_c))
             end
        cond4 = active_c_sum > c2
        cond5 = (constraint_deleted || constraint_added || to_reduce || (W.t == n && W.t == rankA))
        ϵ = max(1e-2, 10.0 * ε_rel)
        cond6 = (W.l == W.q) && !((β_k < ϵ*dnrm) || (b1nrm < ϵ && m == n-W.t))
        
        if !(cond4 || cond5 || cond6)
            cond7 = (iter_km1.α < c5 && non_linearity_km1 < c2*non_linearity_k) || m == n-W.t
            cond8 = !(dnrm <= c4*β_k)
            if cond7 || cond8
                method_code = 2
            end
        end
    end
    return method_code, β_k
end

# DIMUPP
# Determine suitable dimension for solving the system Rx = y
# (i.e how many columns of R should be used)
# where R is rankR*rankR Upper Triangular
# Returns the dimension and a real scalar containing 1.0 when restart is false
# or L(previous_dimR-1)/L(previous_dimR) 
# where L(i) is the length of an estimated search direction computed by using dimension i


function determine_solving_dim(
    previous_dimR::Int64,
    rankR::Int64,
    predicted_linear_progress::Float64,
    obj_progress::Float64,
    prelin_previous_dim::Float64,
    R::UpperTriangular{Float64,Array{Float64,2}},
    y::Vector,
    previous_α::Float64,
    restart::Bool)

    newdim = rankR
    η = 1.0
    mindim = 1
    if rankR > 0
        l_estim_sd, l_estim_righthand = zeros(rankR), zeros(rankR)
        l_estim_sd[1] = abs(y[1])
        l_estim_righthand[1] = abs(y[1] / R[1,1])

        if rankR > 1
            for i = 2:rankR
                l_estim_sd[i] = y[i]
                l_estim_righthand[i] = y[i] / R[i,i]
                l_estim_righthand[i] = norm(l_estim_righthand[i-1:i])
                l_estim_sd[i] = norm(l_estim_sd[i-1:i])
            end
        end

        nrm_estim_sd = l_estim_sd[rankR]
        nrm_estim_righthand = l_estim_righthand[rankR]

        # Determine lowest possible dimension

        dsum = 0.0
        psimax = 0.0
        for i = 1:rankR
            dsum += l_estim_sd[i]^2
            psi = sqrt(dsum) * abs(R[i,i])
            if psi > psimax
                psimax = psi
                mindim = i
            end
        end

        k = mindim
        if !restart
            if previous_dimR == rankR || previous_dimR <= 0
                # Gauss-Newton at previous step
                suggested_dim = gn_previous_step(l_estim_sd, nrm_estim_sd, mindim, l_estim_righthand, nrm_estim_righthand, rankR)
                
            elseif previous_dimR != rankR && rankR > 0
                # Subbspace-Minimization at previous step
                suggested_dim = subspace_min_previous_step(l_estim_sd,l_estim_righthand,nrm_estim_righthand,
                    c1,rankR,previous_dimR,obj_progress,predicted_linear_progress,
                    prelin_previous_dim,previous_α)
            end
            newdim = max(mindim,suggested_dim)
        end

        newdim = max(0, min(rankR, previous_dimR))
        if newdim != 0
            k = max(previous_dimR-1, 1)
            if l_estim_sd[newdim] != 0
                η = l_estim_sd[k] / l_estim_sd[newdim]
            end
        end
    end

    return newdim, η
end

# SUBSPC
# Computes the dimensions of the subspaces where minimization should be done

function choose_subspace_dimensions(
        rx_sum::Float64,
        rx::Vector,
        active_cx_sum::Float64,
        J1::Matrix,
        m::Int64,
        n::Int64,
        t::Int64,
        rankJ2::Int64,
        rankA::Int64,
        b::Vector,
        Q1,
        R11::Matrix,
        P2::Matrix,
        Q3,
        P3::Matrix,
        R22::Matrix,
        previous_iter::Iteration,
        restart::Bool = false)
    
    # Data
    β1, β2, α_low = 0.1, 0.1, 0.2
    previous_α = previous_iter.α
    
    if rankA <= 0
        dimA = 0
        η_A = 1.0
        d = -rx
        
    elseif rankA > 0
        previous_dimA = abs(previous_iter.dimA) + t - previous_iter.t
        nrm_b_asprev = norm(b[1:previous_dimA])
        nrm_b = norm(b)
        constraint_progress = dot(previous_iter.cx,previous_iter.cx) - active_cx_sum
        
        # Determine Dimension for matrix R11 to be used
        dimA, η_A = determine_solving_dim(previous_dimA,rankA,nrm_b,constraint_progress,nrm_b_asprev,UpperTriangular(R11),b,previous_α,restart)
        
        # Solve for p1 the system R11*P2*p1 = b
        # Using dimA columns of R11
        # Forms right hand side d = r(x)+J1*p1
        
        δp1 = UpperTriangular(R11[1:dimA,1:dimA]) \ b[1:dimA]
        p1 = P2 * [δp1;zeros(t-dimA)]
        d = -(rx + J1*p1)
    end
    
    if rankJ2 > 0 d = transpose(Q3)*d end
    previous_dimJ2 = abs(previous_iter.dimJ2) + t - previous_iter.t
    nrm_d_asprev = norm(d[1:previous_dimJ2])
    nrm_d = norm(d)
    residual_progress = dot(previous_iter.rx, previous_iter.rx) - rx_sum
    dimJ2, η_J2 = determine_solving_dim(previous_dimJ2,rankJ2,nrm_d,residual_progress,nrm_d_asprev,UpperTriangular(R22),d,previous_α,restart)
    
    if !restart && previous_α >= α_low
        dimA = max(dimA, previous_dimA)
        dimJ2 = max(dimJ2, previous_dimJ2)
    end
    return dimA, dimJ2
end

# ANALYS
# Check if the latest step was sufficientlt good and eventually 
# recompute the search direction by using either subspace minimization
# or the method of Newton

function search_direction_analys!(
        previous_iter::Iteration,
        current_iter::Iteration,
        iter_number::Int64,
        x::Vector,
        c::Function,
        r::Function,
        rx::Vector,
        cx::Vector,
        active_cx::Vector,
        λ::Vector,
        rx_sum::Float64,
        active_cx_sum::Float64,
        p_gn::Vector,
        d_gn::Vector,
        b_gn::Vector,
        nrm_b1_gn::Float64,
        nrm_d1_gn::Float64,
        nrm_d_gn::Float64,
        J::Matrix,
        m::Int64,
        n::Int64,
        working_set::WorkingSet,
        rankA::Int64,
        rankJ2::Int64,
        P1::Matrix,
        Q1,
        L11::Matrix,
        P2::Matrix,
        Q2,
        R11::Matrix,
        P3::Matrix,
        Q3,
        R22::Matrix,
        constraint_added::Bool,
        constraint_deleted::Bool,
        restart::Bool = false)
    
    
    prev_dimJ2m1 = previous_iter.dimJ2 + previous_iter.t - working_set.t - 1
    nrm_d1_asprev = norm(d_gn[1:prev_dimJ2m1])
    

       method_code, β = check_gn_direction(nrm_b1_gn, nrm_d1_gn, nrm_d1_asprev, nrm_d_gn, active_cx_sum, iter_number, rankA, n, m, restart, constraint_added, constraint_deleted, working_set, cx, λ, previous_iter)

    
    # Gauss-Newton accepted
    if method_code == 1
        dimA = rankA
        dimJ2 = rankJ2
        p, b, d = p_gn, b_gn, d_gn
        
    # Subspace minimization to recompute the search direction
    # using dimA columns of matrix R11 and dimJ2 columns of matrix R22
    elseif method_code == -1
        JQ1 = J*Q1
        J1, J2 = JQ1[:,1:working_set.t], JQ1[:,working_set.t+1:end]
        b = -transpose(Q2)*transpose(P1)*active_cx
        dimA, dimJ2 = choose_subspace_dimensions(rx_sum,rx, active_cx_sum, J1, m, n, working_set.t, rankJ2, rankA, b, Q1, R11, P2, Q3, P3, R22, previous_iter, restart)
        p, b, d = sub_search_direction(J1, J2, rx, active_cx, Q1, P1, L11, Q2, P2, R11, Q3, R22, P3, n, working_set.t, dimA, dimJ2, method_code)
        
    
    
    # Search direction computed with the method of Newton
    elseif method_code == 2
        p = newton_search_direction(x, c, r, active_cx, working_set.active, working_set.t, λ, rx, J, m, n, Q1, P1, L11, Q2, R11, P2, rankA)
        b, d = b_gn, d_gn
        dimA = -working_set.t
        dimJ2 = working_set.t - n
    end
    current_iter.b_gn = b
    current_iter.d_gn = d
    current_iter.dimA = dimA
    current_iter.dimJ2 = dimJ2
    current_iter.code = method_code
    current_iter.β = β
    current_iter.p = p
    return  
end

# PSI in dblreduns.f
# Merit function

function psi(
        r_val::Vector, 
        c_val::Vector, 
        w::Vector, 
        l::Int64, 
        t::Int64,
        active::Vector,
        inactive::Vector)
    
    penalty_constraint_sum = 0.
    
    
    # First part of sum with active constraints 
    for i = 1:t
        j = active[i]
        penalty_constraint_sum += w[j] * c_val[j]^2
    end
    
    # Second part of sum wit inactive constraints
    for i = 1:l-t
        j = inactive[i]
        if c_val[j] < 0.0
            penalty_constraint_sum  += w[j] * c_val[j]^2
        end
    end
    return 0.5 * (dot(r_val,r_val) + penalty_constraint_sum)
end

# MAXNRM
# Update the penalty weights corresponding to the
# constraints in the current working setb

function max_norm_weight_update!(
        nrm2_Ap::Float64,
        rmy::Float64,
        α_w::Float64,
        δ::Float64,
        w::Vector,
        active::Vector,
        t::Int64,
        K::Array{Array{Float64,1},1}
)
    μ = (abs(α_w-1.0) <= δ ? 0.0 : rmy / nrm2_Ap)
    i1 = (active[1] != 0 ? active[1] : 1)
    
    previous_w = w[i1]
    ν = max(μ, K[4][1])
    for i = 1:t
        k = active[i]
        w[k] = ν
    end
    
    if μ > previous_w
        mu_not_placed = true
        i = 1
        while i <= 4 && mu_not_placed
            if μ > K[i][1]
                for j = 4:-1:i+1
                    K[j][1] = K[j-1][1]
                end
                K[i][1] = μ
                mu_not_placed = false
            end
            i += 1
        end
    end
    return
end

function penalty_weight_update(
        w_old::Vector,
        Jp::Vector,
        Ap::Vector,
        K::Array{Array{Float64,1},1},
        rx::Vector,
        rx_sum::Float64,
        cx::Vector,
        active::Vector,
        t::Int64,
        dimA::Int64,
        norm_code::Int64 = 0)
    # Data
    δ = 0.25
    w = w_old[:]
    
    nrm2_Ap = dot(Ap,Ap)
    nrm2_Jp = dot(Jp,Jp)
    Jp_rx = dot(Jp,rx)
    
    AtwA = 0.
    BtwA = 0.
    if dimA > 0
        for i = 1:dimA
            k = active[i]
            AtwA += w[k] * Ap[i]^2
            BtwA += w[k] * Ap[i] * cx[k]
        end
    end
    
    α_w = 1.0
    if abs(AtwA + nrm2_Jp) > eps(Float64)
        α_w = (-BtwA - Jp_rx) / (AtwA + nrm2_Jp)
    end
    
    rmy = (abs(Jp_rx + nrm2_Jp) / δ) - nrm2_Jp
    if norm_code == 0
        max_norm_weight_update!(nrm2_Ap, rmy, α_w, δ, w, active, t, K)
    end
    # TODO: add weight update using euclidean norm method
    
    return w
end

# Equivalent Fortran : QUAMIN in dblreduns.f

function minimize_quadratic(x1::Float64, y1::Float64, x2::Float64, y2::Float64,
    x3::Float64, y3::Float64)

    d1, d2 = y2 - y1, y3 - y1
    s = (x3 - x1)^2 * d1 - (x2 - x1)^2 * d2
    q = 2 * ((x2 - x1) * d2 - (x3 - x1) * d1)
    return x1 - s / q
end


# Equivalent Fortran : MINRN in dblreduns.f


function minrn(x1::Float64, y1::Float64, x2::Float64, y2::Float64,
    x3::Float64, y3::Float64, α_min::Float64, α_max::Float64)

    εrank = sqrt(eps(Float64))

    # α not computable
    # Add an error in this case
    if abs(x1 - x2) < εrank || abs(x3 - x1) < εrank || abs(x3 - x2) < εrank
        α, pα = 0., 0.

    else
    # Compute minimum of quadradic passing through fx, fv and fw
    # at points x, v and w
        u = minimize_quadratic(x1, y1, x2, y2, x3, y3)
        α = clamp(u, α_min, α_max)
        t1 = (α - x1) * (α - x2) * y3 / ((x3 - x1) * (x3 - x2))
        t2 = (α - x3) * (α - x2) * y1 / ((x1 - x3) * (x1 - x2))
        t3 = (α - x3) * (α - x2) * y2 / ((x2 - x1) * (x2 - x3))
        
        # Value of the estimation of ψ(α)
        pα = t1 + t2 + t3
    end
    
    return α, pα
end



function parameters_rm(v0::Vector, v1::Vector, v2::Vector, α_best::Float64,
        ds::Polynomial{Float64}, dds::Polynomial{Float64})
    dds_best = dds(α_best)
    η, d = 0.1, 1.
    normv2 = dot(v2, v2)
    h0 = abs(ds(α_best) / dds_best)
    Dm = abs(6 * dot(v1,v2) + 12 * α_best*normv2) + 24 * h0 * normv2
    hm = max(h0, 1)

    # s'(α) = 0 is solved analytically
    if dds_best * η < 2 * Dm * hm
        
        # If t = α+a1 solves t^3 + b*t + c = O then α solves s'(α) = 0
        (a3, a2, a1) = coeffs(ds) / (2 * normv2)
        
        b = a2 - (a1^2) / 3
        c = a3 - a1 * a2/3 + 2*(a1/3)^3
        d = (c/2)^2 + (b/3)^3
        # Two interisting roots
        if d < 0
            α_hat, β_hat = two_roots(b, c, d, a1, α_best)
        
        # Only one root is computed     
        else
            α_hat = one_root(c, d, a1)
        end
    
    # s'(α) = 0 is solved using Newton-Raphson's method
    else
        α_hat = newton_raphson(α_best, Dm, ds, dds)
    end
    
    # If only one root computed
    if d >= 0 
        β_hat = α_hat 
    end
    return α_hat, β_hat
    
end

function bounds(α_l::Float64, α_u::Float64, α::Float64, s::Polynomial{Float64})
    α = min(α, α_u)
    α = max(α, α_l)
    return α, s(α)
end

function newton_raphson(α_best::Float64, Dm::Float64, ds::Polynomial{Float64}, dds::Polynomial{Float64})
    α, newtonstep = α_best, 0
    ε, error = 1e-4, 1.
    while error > ε || newtonstep < 3
        c = dds(α)
        h = -ds(α) / c
        α += h
        error = (2 * Dm * h^2) / abs(c)
        newtonstep += 1
    end
    return α
end


# Equivalent Fortran : ONER in dblreduns.f
function one_root(c::Float64, d::Float64, a::Float64)
    arg1, arg2 = -c/2 + sqrt(d), -c/2 - sqrt(d)
    return cbrt(arg1) + cbrt(arg2) - a/3
end

# Equivalent Fortran : TWOR in dblreduns.f
function two_roots(b::Float64, c::Float64, d::Float64, a::Float64, α_best::Float64)
    φ = acos(abs(c/2) / (-b/3)^(3/2))
    t = (c <= 0 ? 2*sqrt(-b/3) : -2*sqrt(-b/3))
    
    # β1 is the global minimizerof s(α). 
    # If d is close to zero the root β1 is stable while β2 and β3 become unstable
    β1 = t * cos(φ/3) - a/3
    β2 = t * cos((φ + 2 * π) / 3) - a/3
    β3 = t * cos((φ + 4 * π) / 3) - a/3
    
    # Sort β1, β2 and β3 so that β1 <= β2 <= β3
    β1, β2, β3 = sort([β1, β2, β3])
    
    #β1 or β3 are now the roots of interest
    α, β = (α_best <= β2 ? (β1, β3) : (β3, β1))
    return α, β
end  


# Equivalent Fortran : MINRM in dblreduns.f
function minrm(v0::Vector, v1::Vector, v2::Vector, α_best::Float64, α_l::Float64, α_u::Float64)
    
    s = Polynomial([0.5 * norm(v0)^2, dot(v0,v1), dot(v0,v2) + 0.5 * dot(v1,v1), dot(v1,v2), 0.5 * dot(v2,v2)^2])
    ds = derivative(s)
    dds = derivative(ds)
    α_hat, β_hat = parameters_rm(v0, v1, v2, α_best, ds, dds)
    sα, sβ = s(α_hat), s(β_hat)
    α_old = α_hat
    α_hat, sα = bounds(α_l, α_u, α_hat, s)
    if α_old == β_hat
        β_hat, sβ = α_hat, s(α_hat)
    else
        β_hat, sβ = bounds(α_l, α_u, β_hat, s)
    end
    return α_hat, sα, β_hat, sβ
end

# Equivalent Fortran : GAC in dblreduns.f

function goldstein_armijo_condition(p::Vector, ϕ::Function, ϕ0::Float64, 
        dϕ0::Float64, α0::Float64, α_min::Float64)
    
    τ = 0.25
    ε_rel = sqrt(eps(Float64))
    α, ϕα = α0, ϕ(α0)
    p_max = norm(p)
    exit = p_max * α >= ε_rel || α >= α_min
    while ϕα >= ϕ0 + τ * α * dϕ0 && exit
        α *= 0.5
        ϕα = ϕ(α)
        exit = p_max * α >= ε_rel || α >= α_min
    end
    return α
end

# Useful functions for the line search

function best_alpha(list_alpha::Vector, s::Function)
        minimum(Dict(s(t) => t for t in list_alpha)).second
    end
    
function progress_check(s_star::Float64, ϕ::Function, αk::Float64, αk_1::Float64,
        η::Float64 = 0.2, δ::Float64 = 0.2)
    satisfied = true
    ϕk = ϕ(αk)
    if s_star <= η * ϕk
        if ϕk <= δ * ϕ(αk_1)
            satisfied = false
        end
    end
    return satisfied
end

# Equivalent Fortran : LINEC in dblreduns.f
function linesearch_constrained(
        r::Function,
        rx::Vector, 
        J::Matrix,
        c::Function,
        cx::Vector, 
        A::Matrix,
        active_constraint::Constraint,
        x::Vector, 
        p::Vector, 
        w::Vector, 
        active::Vector, 
        inactive::Vector, 
        t::Int64,
        l::Int64,
        n::Int64,
        m::Int64,
        α0::Float64, 
        αl::Float64, 
        αu::Float64)
    
    # Data
    τ, γ = 0.25, 0.4
    
    # Arrays and matrix corresponding to active constraints
   
    
    active_index = active[1:t]
    w_active = w[active_index]
    active_cx = active_constraint.cx
    active_A = active_constraint.A

   # ϕ = α::Float64 -> psi(x + α * p, r, c, w, active, inactive, t)
    ϕ = α::Float64 -> psi(r(x+α*p), c(x+α*p), w, l, t, active, inactive)
    ϕ0 = ϕ(0.)
    dϕ0 = dot(active_A * p, w_active .* active_cx) + dot(J * p, rx)

    is_acceptable = α::Float64 -> (ϕ(α) <= ϕ0 + τ * dϕ0 * α) || (ϕ(α) <= γ * ϕ0)

    # Computation of v0 and v1
        
    F = α::Float64 -> vcat(r(x + α * p), (w_active.^0.5) .* c(x + α * p)[active_index])
    v0 = F(0.)
    v1 = vcat(J * p, (w_active.^0.5) .* (active_A * p))
    
    k, α_1 = 0, 0.
    
    # Find minimum of parabola interpolating f at 0 and α0
    v2 = [(((F(α0)[i] - v0[i]) / α0) - v1[i]) / α0 for i = 1:m+t]
    αbest = best_alpha([0, α0], t::Float64 -> 0.5*norm(v0 + v1 * t + v2 * t^2)^2)
    α_star, s_star = minrm(v0, v1, v2, αbest, αl, αu)
    
    if is_acceptable(α0)
        satisfied = progress_check(s_star, ϕ, α0, α_1)
        
        if !satisfied α1 = α_star end
        while !satisfied
            if k > 0 α_1, α0, α1 = α0, α1, α_star end
                
            α_best = best_alpha([α_1, α0, α1], t::Float64 -> 0.5*norm(v0 + v1 * t + v2 * t^2)^2)
             
            satisfied = progress_check(s_star, ϕ, α1, α0)
            if k == 0 k += 1 end
        end
        if k == 0 return α0 end
        
    else
        α1 = α_star
        if is_acceptable(α1)
            if ϕ0 <= ϕ(α0)
                v2 = [(((F(α0)[i] - v0[i]) / α1) - v1[i]) / α1 for i = 1:m+t]
                α_best = best_alpha([0, α1], t::Float64 -> 0.5*norm(v0 + v1 * t + v2 * t^2)^2)
                α_star, s_star = minrm(v0, v1, v2, α_best, αl, αu)
                α0 = 0.
            else

                α_best = best_alpha([α_1, α0, α1], t::Float64 -> 0.5*norm(v0 + v1 * t + v2 * t^2)^2)
                α_star, s_star = minrn(0., ϕ0, α0, ϕ(α0), α1, ϕ(α1), αl, αu)
                satisfied = progress_check(s_star, ϕ, α1, α0)
                if !satisfied α1 = α_star end
                
                while !satisfied
                    if k > 0 α_1, α0, α1 = α0, α1, α_star end
                
                    α_best = best_alpha([α_1, α0, α1], t::Float64 -> 0.5*norm(v0 + v1 * t + v2 * t^2)^2)
                    α_star, s_star = minrn(α_1, ϕ(α_1), α0, ϕ(α0), α1, ϕ(α1), αl, αu)
                    satisfied = progress_check(s_star, ϕ, α1, α0)
                    if k == 0 k += 1 end
                end
            end
        else
            α1 = goldstein_armijo_condition(p, ϕ, ϕ0, dϕ0, α1, αl)
            # TODO: add some test conditions on α1
        end
            
        
    end
    α = best_alpha([α0, α1], ϕ)
    return α
end

# UPBND
# Determine the upper bound of the steplength

function upper_bound_steplength(
        A::Matrix,
        cx::Vector,
        p::Vector,
        inactive::Vector,
        t::Int64,
        l::Int64
        
    )
    
    α_upper = Inf
    if norm(inactive, Inf) > 0
        for i = 1:l-t
            j = inactive[i]
            ∇cj_p = dot(A[j,:],p)
            if cx[j] > 0 && ∇cj_p < 0 && -dot(A[j,:],p) < α_upper
                α_upper = -cx[j] / ∇cj_p
            end
        end
    end
    α_upper = min(3., α_upper)
    return α_upper
end

# STPLNG
# Update the penalty weights and compute the steplength using the merit function psi
# If search direction computed with method of Newton, an undamped step is taken (i.e. α=1)

function compute_steplength(
        x::Vector,
        r::Function,
        rx::Vector,
        J::Matrix,
        p::Vector,
        c::Function,
        cx::Vector,
        A::Matrix,
        active_constraint::Constraint,
        w_old::Vector,
        work_set::WorkingSet,
        K::Array{Array{Float64,1},1},
        dimA::Int64,
        n::Int64,
        m::Int64,
        previous_α::Float64,
        method_code::Int64
    )
    
    # Data
    c1 = 1e-3
    rx_sum = dot(rx,rx)
    Jp = J*p
    active_Ap = active_constraint.A * p
    
    if method_code != 2
        # Compute penalty weights
        w = penalty_weight_update(w_old, Jp, active_Ap, K, rx, rx_sum, cx, work_set.active, work_set.t, dimA)

        # Determine upper bound of the steplength
        α_upper = upper_bound_steplength(A, cx, p, work_set.inactive, work_set.t, work_set.l)
        α_low = α_upper / 3000.0

        # Determine a first guess of the steplength
        α0 = 2.0 * min(1.0, 3.0*previous_α,α_upper)

        # Compute the steplength
        α = linesearch_constrained(r, rx, J, c, cx, A, active_constraint, x, p, w, work_set.active, work_set.inactive, work_set.t, work_set.l, n, m, α0, α_low, α_upper)
    else
        w = w_old
        α = 1.0
    end
    # TODO: Computation of predicted linear progress as done in the code
    return α, w
end

# Equivalent Fortran : TERCRI in dblmod2nls.f

function check_termination_criterias(
        method_code::Int64,
        previous_x::Vector,
        x::Vector, 
        p_gn::Vector,
        d::Vector,
        rx::Vector,
        cx::Vector,
        active_C::Constraint,
        λ::Vector,
        ∇fx::Vector, 
        W::WorkingSet, 
        dimJ2::Int64,
        restart::Bool = false)
    
    ε_rel = sqrt(eps(Float64))
    
    if !restart && method_code != -1
              
        inactive_index = W.inactive[1:W.l-W.t]
        inactive_cx = cx[inactive_index]
        
        # Necessary conditions 
        
        constraint_cond = norm(active_C.cx) < ε_rel && all(>(ε_rel), inactive_cx)
        if constraint_cond
            lagrange_mult_pos = [λ[i] for i=W.q+1:W.t if λ[i] > 0]
            sigmin = (isempty(lagrange_mult_pos) ? 0 : minimum(lagrange_mult_pos))
            
            if W.q+1 == W.t || isempty(λ)
                u = 1 + dot(rx,rx)
            else
                u =  maximum(map(abs,λ))
            end
            
            if sigmin >= ε_rel * u && norm(transpose(active_C.A) * λ - ∇fx) < sqrt(ε_rel) * (1 + norm(∇fx))
                
                # Sufficient conditions
                norm_rx2 = dot(rx,rx)
                return dot(d[1:dimJ2],d[1:dimJ2]) <= ε_rel^2 * norm_rx2 || norm_rx2 <= ε_rel^2 || norm(x_previous - x) < ε_rel * norm(x) || sqrt(ε_rel) / norm(p_gn) > 0.25
            end
        end
    end
    return false
end

mutable struct ENLSIP
    sol::Vector
    obj_value::Float64
end



enlsip_010 = ENLSIP([0.0],0.0)

function (enlsip_010::ENLSIP)(
        x0::Vector, 
        r::Function, 
        c::Function, 
        jac_r::Function, 
        jac_c::Function, 
        q::Int64,
        l::Int64)

    MAX_ITER = 100
    # Objective function
    
    f(x::Vector) = 0.5 * dot(r(x),r(x))
    ∇f = x::Vector -> transpose(jac_r(x))*r(x)

    # Vector of penalty constants
    K = [zeros(l) for i=1:4]

    # First Iteration
    rx = r(x0)
    cx = c(x0)
    J = jac_r(x0)
    ∇fx = ∇f(x0)
    A = jac_c(x0)
    m,n = size(J)
    nb_iteration = 1

    first_iter = Iteration(x0, zeros(n), rx, cx, l, 3.0,zeros(l), zeros(l), 0, 0, 0, 0, zeros(n), zeros(n), 0., 0., 0., false, true, false, false,1)

    # Initialization of the working set
    working_set = init_working_set(cx, K, first_iter, q, l)
    println("Premier ensemble actif :")
    show_working_set(working_set)
    first_iter.t = working_set.t
    active_C = Constraint(cx[working_set.active[1:working_set.t]], A[working_set.active[1:working_set.t],:])
   
    rx_sum = dot(rx,rx)
    p_gn = zeros(n)
    
    # Estimation of la Lagrange multipliers
    # Computation of the Gauss-Newton search direction
    update_working_set!(working_set, rx, active_C, ∇fx, J, p_gn, first_iter)
    active_cx_sum = dot(active_C.cx,active_C.cx)
    first_iter.t = working_set.t
    previous_iter = copy(first_iter)
    F_A = qr(transpose(active_C.A), Val(true))
    L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q, F_A.P
    F_L = qr(L11, Val(true))
    R11, Q2, P2 = F_L.R, F_L.Q, F_L.P
    J2 = (J*Q1)[:,working_set.t+1:end]
    F_J2 = qr(J2, Val(true))
    Q3, P3, R22 = F_J2.Q, F_J2.P, F_J2.R
    nrm_b1 = norm(first_iter.b_gn[1:first_iter.dimA])
    nrm_d1 = norm(first_iter.d_gn[1:first_iter.dimJ2])
    nrm_d = norm(first_iter.d_gn)
    
    # Analys of the lastly computed search direction
    search_direction_analys!(previous_iter,first_iter,nb_iteration,x0,c,r,rx,cx,active_C.cx,first_iter.λ,rx_sum,active_cx_sum,p_gn,first_iter.d_gn,first_iter.b_gn,nrm_b1,nrm_d1,nrm_d,J,m,n,working_set,first_iter.rankA,first_iter.rankJ2,P1,Q1,L11,P2,Q2,R11,P3,Q3,R22,first_iter.add,first_iter.del)
    

    # Computation of penalty constants and steplentgh
    α, w = compute_steplength(x0,r,rx,J,first_iter.p,c,cx,A,active_C,previous_iter.w,working_set,K,first_iter.dimA,n,m,previous_iter.α,first_iter.code)
    first_iter.α = α
    first_iter.w = w
    x = x0 + α*first_iter.p
   
    println("------Iteration $nb_iteration ------")
    show_iter(first_iter)
    show_working_set(working_set)
    # Check for termination criterias at new point
    
    rx = r(x)
    rx_sum = dot(rx,rx)
    cx = c(x)
    J = jac_r(x)
    ∇fx = ∇f(x)
    A = jac_c(x)
    
    
    terminated = check_termination_criterias(first_iter.code,previous_iter.x,x,first_iter.p,first_iter.d_gn,rx,cx,active_C,first_iter.λ,∇fx,working_set,first_iter.dimJ2)

    
    # Check for violated constraints and add it to the working set
    first_iter.add = evaluate_violated_constraints(c(x),working_set)
    
    active_C = Constraint(cx[working_set.active[1:working_set.t]], A[working_set.active[1:working_set.t],:])
    
    nb_iteration += 1
    previous_iter = first_iter
    iter = copy(first_iter)
    iter.x = x
    

    # Next iterations
    while !terminated && nb_iteration < MAX_ITER
        p_gn = zeros(n)
    
        # Estimation of Lagrange multipliers
        # Computation of the Gauss-Newton search direction
        update_working_set!(working_set, rx, active_C, ∇fx, J, p_gn,iter)
        active_cx_sum = dot(active_C.cx,active_C.cx)
        iter.t = working_set.t
        F_A = qr(transpose(active_C.A), Val(true))
        L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q, F_A.P
        F_L = qr(L11, Val(true))
        R11, Q2, P2 = F_L.R, F_L.Q, F_L.P
        J2 = (J*Q1)[:,working_set.t+1:end]
        F_J2 = qr(J2, Val(true))
        Q3, P3, R22 = F_J2.Q, F_J2.P, F_J2.R
        nrm_b1 = norm(iter.b_gn[1:iter.dimA])
        nrm_d1 = norm(iter.d_gn[1:iter.dimJ2])
        nrm_d = norm(iter.d_gn)
        
        # Analys of the lastly computed search direction
        search_direction_analys!(previous_iter,iter,nb_iteration,x,c,r,rx,cx,active_C.cx,iter.λ,rx_sum,active_cx_sum,p_gn,iter.d_gn,iter.b_gn,nrm_b1,nrm_d1,nrm_d,J,m,n,working_set,iter.rankA,iter.rankJ2,P1,Q1,L11,P2,Q2,R11,P3,Q3,R22,iter.add,first_iter.del)
    

        # Computation of penalty constants and steplentgh
        α, w = compute_steplength(x,r,rx,J,iter.p,c,cx,A,active_C,previous_iter.w,working_set,K,iter.dimA,n,m,previous_iter.α,iter.code)
        iter.α = α
        iter.w = w
        x = x + α*iter.p
        println("------ Iteration $nb_iteration ------")
        show_iter(iter)
        
        # Check for termination criterias at new point
        
        rx = r(x)
        rx_sum = dot(rx,rx)
        cx = c(x)
        J = jac_r(x)
        ∇fx = ∇f(x)
        A = jac_c(x)
    
        
        terminated = check_termination_criterias(iter.code,previous_iter.x,x,iter.p,iter.d_gn,rx,cx,active_C,iter.λ,∇fx,working_set,iter.dimJ2)
        
    
        # Check for violated constraints and add it to the working set
        iter.add = evaluate_violated_constraints(cx,working_set)
        if iter.add
            active_C = Constraint(cx[working_set.active[1:working_set.t]], A[working_set.active[1:working_set.t],:])
        end
        println("Working Set :")
        show_working_set(working_set)
        
        nb_iteration += 1
        previous_iter = iter
        iter = copy(iter)
        iter.x = x
    end    
    enlsip_010.sol = iter.x
    enlsip_010.obj_value = f(iter.x)
    return
end
