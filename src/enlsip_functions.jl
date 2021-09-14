using LinearAlgebra, Polynomials, Printf


"""

    Iteration

Summarizes the useful informations about an iteration of the algorithm

* `x` : Departure point of the iteration 

* `p` : Descent direction

* `rx` : vector of size `m`, contains value of residuals at `x` 

* `cx` : vector of size `l`, constains value of constraints at `x`

* `t` : Number of constraints in current working set (ie constraints considered active)

* `α` : Value of steplength

* `λ` : Vector of size `t`, containts Lagrange multipliers estimates

* `rankA` : pseudo rank of matrix `A`, jacobian of active constraints

* `rankJ2` : pseudo rank of matrix `J2`, block extracted from `J`, jacobian of residuals

* `b_gn` : right handside of the linear system solved to compute first part of `p`

* `d_gn` :  right handside of the linear system solved to compute second part of `p`

* `predicted_reduction` : predicted linear progress

* `progress` :  reduction in the objective function

* `β` : scalar used to estimate convergence factor

* `restart` : indicate if current iteration is a restart step or no

* `first` : indicate if current iteration is the first one or no

* `add` : indicate if a constraint has been added to the working set 

* `del` : indicate if a constraint has been deleted from the working set

* `index_del` : index of the constraint that has been deleted from working set (`0` if no deletion)

* `code` : Its value caracterizes the method used to compute the search direction `p`

    - ``1`` represents Gauss-Newton method

    - ``-1`` represents Subspace minimization

    - ``2``  represents Newton method
"""

mutable struct Iteration
    x::Vector{Float64}
    p::Vector{Float64}
    rx::Vector{Float64}
    cx::Vector{Float64}
    t::Int64
    α::Float64
    λ::Vector{Float64}
    w::Vector{Float64}
    rankA::Int64
    rankJ2::Int64
    dimA::Int64
    dimJ2::Int64
    b_gn::Vector{Float64}
    d_gn::Vector{Float64}
    predicted_reduction::Float64
    progress::Float64
    β::Float64
    restart::Bool
    first::Bool
    add::Bool
    del::Bool
    index_del::Int64
    code::Int64
end


Base.copy(s::Iteration) = Iteration(s.x, s.p, s.rx, s.cx, s.t, s.α, s.λ, s.w, s.rankA, s.rankJ2, s.dimA, s.dimJ2, s.b_gn, s.d_gn, s.predicted_reduction, s.progress, s.β, s.restart, s.first, s.add, s.del, s.index_del, s.code)

# Used to distinguish active constraints
# Reprensents the useful informations about active constraints at a point x, i.e. :
# cx : Vector of size t, contains values of constraints in current working set
# A : Matrix of size t x t, constaints jacobian of constraints in current working set
# scaling : Boolean indicating if internal scaling of cx and A is done 
# diag_scale : Vector of size t, contains the diagonal elements of the scaling matrix if scaling is done 
# i.e  the i-th element equals [1 / ||∇c_i(x)||] for i = 1,...,t, which is the inverse of the length of A i-th row 
# Otherwise, it contains the length of each row in the matrix A

mutable struct Constraint
    cx::Vector{Float64}
    A::Matrix{Float64}
    scaling::Bool
    diag_scale::Vector{Float64}
end

# EVSCAL 
# Scale jacobian matrix of active constraints A and active constraints evaluation vector cx if so indicated (ie if scale different from 0) by forming vectors :
# diag*A and diag*cx
# where diag is an array of dimension whose i-th element equals either ||∇c_i(x)|| or  (1/||∇c_i(x)|) depending on wether scaling is done or not. 
# The vectors are formed by modifying in place matrix A and vector cx 

function evaluate_scaling!(C::Constraint)

    t = size(C.A,1)
    ε_float = eps(Float64)
    C.diag_scale = zeros(t)
    for i = 1:t
        row_i = norm(C.A[i,:])
        C.diag_scale[i] = row_i
        if C.scaling
            if abs(row_i) < ε_float row_i = 1.0 end
            C.A[i,:] /= row_i
            C.cx[i] /= row_i
            C.diag_scale[i] = 1.0 / row_i
        end
    end
    return
end

# In ENLSIP, the working-set is a prediction of the set of active constraints at the solution
# It is updated at every iteration thanks to a Lagrangian multipliers estimation

# Fields of the struc below summarize infos about the qualification of the constraints, i.e. :
# q : number of equality constraints
# t : number of constraints in current working set (all equalities and some inequalities considered to be active at the solution)
# l : total number of constraints (equalities and inequalities)
# active : first t elements are indeces of constraints in working set sorted in increasing order, last ones equal 0 (total length : l)
# inactive : first l-t elements are indeces of constraints not in working set sorted in increasing order, last ones equal 0 (total length : l-q)


mutable struct WorkingSet
    q::Int64
    t::Int64
    l::Int64
    active::Vector{Int64}
    inactive::Vector{Int64}
end

# Computes and returns the rank of a triangular matrix T using its diagonal elements placed in decreasing order
# according to their absolute value

# diag_T is the diagonal of the Triangular matrix T whose rank is estimated
# τ is the relative tolerance to estimate the rank

function pseudo_rank(diag_T::Vector{Float64}, τ::Float64=sqrt(eps(Float64)))
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


"""
    EvalFunc(ctrl::Int64) 

This structure is used to define functions evaluating residuals, constraints and corresponding jacobians.

Both functions for residuals and constraints must be written as follows and must not return any value (components are modified in the body of the function):

# Example definition of an EvalFunc type function

```jldoctest
function (h::EvalFunc)(x::Vector{Float64}, hx::Vector{Float64}, Jh::Matrix{Float64})
    if h.ctrl == 1 
        hx[:] = [...]
    elseif h.ctrl == 2
        Jh[:] = [...] # if supplied anatically
    end
    # The elseif block above could also be, if jacobian not supplied anatically
    # elseif h.ctrl == 2 h.ctrl = 0 end
    return
end
```

The `ctrl` field indicates what is computed (i.e. evalutation or jacobian) when calling a function of type `EvalFunction`.

* `ctrl` = ``1`` means the function `h` is evaluated at point `x`, (modifies in place vector `hx`)

* `ctrl` = ``2`` means the jacobian of `h` is computed at point `x` if jacobian is supplied anatically (then modifies in place matrix `Jh`)

* if jacobian is not supplied anatically, `ctrl` is set to ``0`` on return and jacobian is computed numerically.
"""
abstract type EvalFunc end

"""
    ResidualsEval <: EvalFunc

Subtype of [`EvalFunc`](@ref)  dedicated to the evalutation of residuals values and jacobian matrix.
"""
mutable struct ResidualsEval <: EvalFunc
    ctrl::Int64
end

"""
    ConstraintsEval <: EvalFunc

Subtype of [`EvalFunc`](@ref) dedicated to the evalutation of constraints values and jacobian matrix.
"""
mutable struct ConstraintsEval <: EvalFunc
    ctrl::Int64
end

# JACDIF
# Compute the (m x n) jacobian of h(x) at the current point by using forward differences
# Result is stored in place in the matrix Jh
function jac_forward_diff!(
    x::Vector{Float64},
    h::EvalFunc,
    hx::Vector{Float64},
    n::Int64,
    m::Int64,
    Jh::Matrix{Float64})

    δ = sqrt(eps(Float64))

    for j = 1:n
        δ_j = max(abs(x[j]), 1.0) * δ
        e_j = [(i == j ? 1.0 : 0.0) for i = 1:n]
        hx_forward = zeros(m)
        h.ctrl = 1
        x_forward = x + δ_j * e_j
        h(x_forward, hx_forward, Jh)

        if h.ctrl >= -10
            Jh[:,j] = (hx_forward - hx) / δ_j
        end
    end
    return
end

# NEWPNT
# Compute in place the jacobians J and A corresponding to the ResidualsEvals and the constraints respectively at current point x

function new_point!(x::Vector{Float64},
                    r::ResidualsEval,
                    rx::Vector{Float64},
                    c::ConstraintsEval,
                    cx::Vector{Float64},
                    J::Matrix{Float64},
                    A::Matrix{Float64},
                    n::Int64,
                    m::Int64,
                    l::Int64)
    r.ctrl = 2
    r(x, rx, J)
    if r.ctrl == 0
        # Compute the jacobian numerically
        jac_forward_diff!(x, r, rx, n, m, J)
    end

    c.ctrl = 2
    if l != 0
        c(x, cx, A)
        if c.ctrl == 0
            # Compute the jacobian numerically
            jac_forward_diff!(x, c, cx, n, l, A)
        end
    end
    return
end

# SUBDIR
# Computes a search direction with Gauss-Newton method using dimA and dimJ2 as subspaces dimensions

function sub_search_direction(
        J1::Matrix{Float64},
        rx::Vector{Float64},
        cx::Vector{Float64},
        Q1::Matrix{Float64},
        P1::Matrix{Float64},
        L11::Matrix{Float64},
        Q2::Matrix{Float64},
        P2::Matrix{Float64},
        R11::Matrix{Float64},
        Q3::Matrix{Float64},
        R22::Matrix{Float64},
        P3::Matrix{Float64},
        n::Int64,
        t::Int64,
        rankA::Int64,
        dimA::Int64,
        dimJ2::Int64,
        code::Int64)

    # Solving without stabilization
    
    if code == 1
        b = -transpose(P1) * cx
        δp1 = LowerTriangular(L11) \ b
        p1 = δp1
        d = - transpose(Q3) * (J1 * p1 + rx)
        δp2 = UpperTriangular(R22[1:dimJ2,1:dimJ2]) \ d[1:dimJ2]
        p2 = P3 * [δp2; zeros(n - t - dimJ2)]

    # Solving with stabilization
    elseif code == -1
        b = - transpose(Q2) * transpose(P1) * cx
        δp1 = UpperTriangular(R11[1:dimA,1:dimA]) \ b[1:dimA]
        p1 = P2[1:rankA,1:rankA] * [δp1; zeros(rankA - dimA)]
        d = - transpose(Q3) * (J1 * p1 + rx)
        δp2 = UpperTriangular(R22[1:dimJ2, 1:dimJ2]) \ d[1:dimJ2]
        p2 = P3 * [δp2; zeros(n - rankA - dimJ2)]
    end

    p = Q1 * [p1;p2]
    return p, b, d
end




# GNSRCH
# Compute the search direction with the method of Gauss-Newton
# dimA and dimJ2 are equal to the rank of the corresponding matrices

function gn_search_direction(
    J::Matrix{Float64},
    rx::Vector{Float64},
    cx::Vector{Float64},
    Q1::Matrix{Float64},
    P1::Matrix{Float64},
    L11::Matrix{Float64},
    Q2::Matrix{Float64},
    P2::Matrix{Float64},
    R11::Matrix{Float64},
    rankA::Int64,
    t::Int64,
    τ::Float64,
    current_iter::Iteration)

    code = (rankA == t ? 1 : -1)

    (m, n) = size(J)
    JQ1 = J * Q1
    J1, J2 = JQ1[:,1:rankA], JQ1[:,rankA + 1:end]
    F_J2 = qr(J2, Val(true))
    Q3, P3, R22 = F_J2.Q*Matrix(I,m,m), F_J2.P, F_J2.R
    rankJ2 = pseudo_rank(diag(R22), τ)
    p_gn, b_gn, d_gn = sub_search_direction(J1, rx, cx, Q1, P1, L11, Q2, P2, R11, Q3, R22, P3, n, t, rankA, rankA, rankJ2, code)
    current_iter.rankA = rankA
    current_iter.rankJ2 = rankJ2
    current_iter.dimA = rankA
    current_iter.dimJ2 = rankJ2
    current_iter.b_gn = b_gn
    current_iter.d_gn = d_gn
    return p_gn, F_J2

end

# HESSF
#                                     m
# Compute in place the (n x n) matrix B = Σ  [r_k(x) * G_k]
#                                    k=1
# where G_k is the hessian of residual r_k(x)

function hessian_res!(
    r::ResidualsEval,
    x::Vector{Float64},
    rx::Vector{Float64},
    n::Int64,
    m::Int64,
    B::Matrix{Float64})

    # Only residuals evaluation
    r.ctrl = 1
    dummy = zeros(1, 1)
    # Data
    ε1 = eps(Float64)^(1 / 3)

    for k in 1:n, j in 1:k
        ε_k = max(abs(x[k]), 1.0) * ε1
        ε_j = max(abs(x[j]), 1.0) * ε1
        e_k = [i == k for i = 1:n]
        e_j = [i == j for i = 1:n]

        f1, f2, f3, f4 = zeros(m), zeros(m), zeros(m), zeros(m)
        r(x + ε_j * e_j + ε_k * e_k, f1, dummy)
        r(x - ε_j * e_j + ε_k * e_k, f2, dummy)
        r(x + ε_j * e_j - ε_k * e_k, f3, dummy)
        r(x - ε_j * e_j - ε_k * e_k, f4, dummy)

        # Compute line j of g_k
        g_kj = (f1 - f2 - f3 + f4) / (4 * ε_j * ε_k)

        s = dot(g_kj, rx)
        B[k,j] = s
        if j != k B[j,k] = s end
    end
end

# HESSH
#                                         t
# Compute in place the (n x n) matrix B = Σ  [λ_i * G_k]
#                                        k=1
# where G_k is the hessian of residual c_k(x), k in current working set
# λ = (λ_1,...,λ_t) are the lagrange multipliers estimates

function hessian_cons!(
    c::ConstraintsEval,
    x::Vector{Float64},
    λ::Vector{Float64},
    active::Vector{Int64},
    n::Int64,
    l::Int64,
    t::Int64,
    B::Matrix{Float64})

    # Only constraints evaluation
    c.ctrl = 1
    dummy = zeros(1, 1)
    # Data
    ε1 = eps(Float64)^(1 / 3)
    active_indeces = @view active[1:t]

    for k in 1:n, j in 1:k
        ε_k = max(abs(x[k]), 1.0) * ε1
        ε_j = max(abs(x[j]), 1.0) * ε1
        e_k = [i == k for i = 1:n]
        e_j = [i == j for i = 1:n]

        f1, f2, f3, f4 = zeros(l), zeros(l), zeros(l), zeros(l)
        c(x + ε_j * e_j + ε_k * e_k, f1, dummy)
        c(x - ε_j * e_j + ε_k * e_k, f2, dummy)
        c(x + ε_j * e_j - ε_k * e_k, f3, dummy)
        c(x - ε_j * e_j - ε_k * e_k, f4, dummy)
        act_f1 = @view f1[active_indeces]
        act_f2 = @view f2[active_indeces]
        act_f3 = @view f3[active_indeces]
        act_f4 = @view f4[active_indeces]

        # Compute line j of G_k
        g_kj = (act_f1 - act_f2 - act_f3 + act_f4) / (4.0 * ε_k * ε_j)
        s = dot(g_kj, λ)
        B[k,j] = s
        if k != j B[j,k] = s end
    end
end

# NEWTON
# Computes the search direction p by minimizing :
#      T    T                             T       T
# 0.5*p * (J * J - c_mat + r_mat) * p + (J * r(x)) * p
# s.t.
#     A*p + c(x) = 0
#
#
#         t
# c_mat = Σ  [λ_i * K_i]
#        i=1
# where K_i is the hessian of constraint c_i(x), i in current working set
#         m
# r_mat = Σ  [r_i(x) * G_i]
#        i=1
# where G_i is the hessian of residual r_i(x)

function newton_search_direction(
    x::Vector{Float64},
    c::ConstraintsEval,
    r::ResidualsEval,
    active_cx::Vector{Float64},
    active::Vector{Int64},
    n::Int64,
    m::Int64,
    l::Int64,
    t::Int64,
    λ::Vector{Float64},
    rx::Vector{Float64},
    J::Matrix{Float64},
    Q1::Matrix{Float64},
    P1::Matrix{Float64},
    L11::Matrix{Float64},
    Q2::Matrix{Float64},
    R11::Matrix{Float64},
    P2::Matrix{Float64},
    rankA::Int64)

    if t == rankA
        b = -transpose(P1) * active_cx
        p1 = LowerTriangular(L11) \ b
     elseif t > rankA
        b = -transpose(Q2) * transpose(P1) * active_cx
        δp1 = UpperTriangular(R11[1:rankA,1:rankA]) \ b[1:rankA]
        p1 = P2[1:rankA,1:rankA] * δp1
    end

    if rankA == n return p1 end
    
    # Computation of J1, J2
    JQ1 = J * Q1
    J1, J2 = JQ1[:,1:rankA], JQ1[:,rankA + 1:end]

    # Computation of hessian matrices
    r_mat, c_mat = zeros(n, n), zeros(n, n)

    hessian_res!(r, x, rx, n, m, r_mat)
    hessian_cons!(c, x, λ, active, n, l, t, c_mat)

    Γ_mat = r_mat - c_mat

    if rankA == t
        E = transpose(Q1) * Γ_mat * Q1
    elseif t > rankA
        Q1P2 = Q1*P2
        E = transpose(Q1P2) * Γ_mat * Q1P2
    end

    # Forms the system to compute p2
    E21 = E[rankA + 1:n, 1:rankA]
    E22 = E[rankA + 1:n, rankA + 1:n]

    W22 = E22 + transpose(J2) * J2
    W21 = E21 + transpose(J2) * J1

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
# Solves the system A^T * λ_ls = ∇f(x) using qr factorisation of A^T
# A^T*P1 = Q1 * (R)
#              (0)
# with R^T = L11
# then computes estimates of lagrage multipliers by forming :
#                  -1
# λ = λ_ls - (A*A^T) *cx

function first_lagrange_mult_estimate!(A::Matrix{Float64}, 
    λ::Vector{Float64}, 
    ∇fx::Vector{Float64}, 
    cx::Vector{Float64},
    scaling_done::Bool,
    diag_scale::Vector{Float64},
    F::Factorization{Float64})


    (t, n) = size(A)
    v = zeros(t)
    vnz = zeros(t)
    prankA = pseudo_rank(diag(F.R))
    b = transpose(F.Q) * ∇fx
    v[1:prankA] = UpperTriangular(F.R[1:prankA,1:prankA]) \ b[1:prankA]
    if prankA < t
        v[prankA + 1:t] = zeros(t - prankA)
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

    # Back transform due to row scaling of matrix A
    if scaling_done λ[:] = λ .* diag_scale end
    return
end

# LEAEST
# Compute second order least squares estimate of Lagrange multipliers
#                     T          T            T
# Solves the system  A * λ = J(x) (r(x) + J(x) * p_gn))
function second_lagrange_mult_estimate!(
    J::Matrix{Float64},
    Q1::Matrix{Float64},
    R::Matrix{Float64},
    P1::Matrix{Float64},
    λ::Vector{Float64},
    rx::Vector{Float64},
    p_gn::Vector{Float64},
    t::Int64)

    J1 = (J * Q1)[:, 1:t]
    b = transpose(J1) * (rx + J * p_gn)
    v = UpperTriangular(R) \ b
    λ[:] = P1 * v
    return

end



# Equivalent Fortran : DELETE in dblreduns.f
# Moves the active constraint number s to the inactive set

function delete_constraint!(W::WorkingSet, s::Int64)

    l, t = W.l, W.t

    # Ajout de la contrainte à l'ensemble inactif
    W.inactive[l - t + 1] = W.active[s]
    sort!(@view W.inactive[1:l - t + 1])

    # Réorganisation de l'ensemble actif
    for i = s:t - 1
        W.active[i] = W.active[i + 1]
    end
    W.active[t] = 0
    W.t -= 1
    return
end

# Equivalent Fortran : ADDIT in dblreduns.f
# Add the inactive constraint nulber s to the active s

function add_constraint!(W::WorkingSet, s::Int64)

    l, t = W.l, W.t
    # s-th inactive constraint moved from inactive to active set
    W.active[t + 1] = W.inactive[s]
    sort!(@view W.active[1:t + 1])
    # Inactive set reorganized
    for i = s:l - t - 1
        W.inactive[i] = W.inactive[i + 1]
    end
    W.inactive[l - t] = 0
    W.t += 1
    return
end

# SIGNCH 
# Returns the index of the constraint that shall be deleted from the working set
# Returns 0 if no constraint shall be deleted
# Obtainted with the lagrange mulitpliers estimates

function check_constraint_deletion(
    q::Int64,
    A::Matrix{Float64},
    λ::Vector{Float64},
    ∇fx::Vector{Float64},
    scaling::Bool,
    diag_scale::Vector{Float64})

    t = length(λ)
    δ = 10
    τ = 0.5
    sq_rel = sqrt(eps(Float64))
    s = 0
    if t > q
        e = 0
        for i = q + 1:t
            row_i = (scaling ? 1.0 / diag_scale[i] : diag_scale[i])
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
        cx::Vector{Float64},
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

# Updates QR factorisation of A^T by appyling Givens rotations
# Returns 
function update_QR_A(
    Q::Matrix{Float64},
    R::Matrix{Float64},
    p::Vector{Int64},
    j::Int64,
    t::Int64)

    # Update permutation vector, form permutation matrix, delete j-th column of matrix R 
    setdiff!(p,j)
    p[:] = [(e > j ? e-1 : e) for e in p]
    P = (1.0*Matrix(I,t,t))[:,p]
    R_temp = R[:,setdiff(1:end,j)]

    # Apply Givens rotations to transform R to Upper triangular form

    for i = j:t
        G, r = givens(R_temp[i,i],R_temp[i+1,i],i,i+1)
        lmul!(G,R_temp)
        rmul!(Q,G')
    end
    L = transpose(R)
    # println(diag(L[1:t,1:t]))
    # println(maximum(map(abs,A-P*L*transpose(Q))))
    # println("||AQ-PL|| = $(norm(A-P*L*transpose(Q)))")
    return P, Matrix(transpose(R_temp[1:t,1:t])), Q
end

# WRKSET
# Estimate the lagrange multipliers and eventually delete a constraint from the working set
# Compute the search direction using Gauss-Newton method

function update_working_set(
    W::WorkingSet,
    rx::Vector{Float64},
    A::Matrix{Float64},
    C::Constraint,
    ∇fx::Vector{Float64},
    J::Matrix{Float64},
    p_gn::Vector{Float64},
    iter_k::Iteration)


    λ = Vector{Float64}(undef, W.t)
    ε_rank = sqrt(eps(Float64))
    F_A = qr(transpose(C.A), Val(true))
    first_lagrange_mult_estimate!(C.A, λ, ∇fx, C.cx, C.scaling, C.diag_scale,F_A)
    s = check_constraint_deletion(W.q,C.A,λ,∇fx,C.scaling,C.diag_scale)
    (m,n) = size(J)
    # Constraint number s is deleted from the current working set
    if s != 0
        # Save s-th element of cx,λ and row s of A to test for feasible direction
        cx_s = C.cx[s]
        A_s = C.A[s,:]
        λ_s = λ[s]
        diag_scale_s = C.diag_scale[s]
        index_s = W.active[s]
        deleteat!(λ, s)
        deleteat!(C.cx, s)
        deleteat!(C.diag_scale, s)
        delete_constraint!(W, s)
        iter_k.del = true
        iter_k.index_del = index_s
        C.A = C.A[setdiff(1:end, s),:]
        Q1 = F_A.Q*Matrix(I,n,n)
        vect_P1 = F_A.p[:]
        P1, L11, Q1 = update_QR_A(Q1,F_A.R,vect_P1,s,W.t)
        rankA = pseudo_rank(diag(L11), ε_rank)
        F_L11 = qr(L11, Val(true))
        R11, Q2, P2 = F_L11.R, F_L11.Q*Matrix(I,W.t,W.t), F_L11.P

        p_gn[:], F_J2 = gn_search_direction(J, rx, C.cx, Q1, P1, L11, Q2, P2, R11, rankA, W.t, ε_rank, iter_k)

        # Test for feasible direction
        As_p = (rankA <= W.t ? 0.0 : dot(A_s, p_gn))
        feasible = (As_p >= -cx_s && As_p > 0)

        if !feasible
            insert!(C.cx, s, cx_s)
            insert!(λ, s, λ_s)
            insert!(C.diag_scale, s, diag_scale_s)
            s_inact = findfirst(isequal(index_s), W.inactive)
            add_constraint!(W, s_inact)
            iter_k.index_del = 0
            iter_k.del = false
            C.A = (C.scaling ? A[W.active[1:W.t],:] .* C.diag_scale : A[W.active[1:W.t],:])
            L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q*Matrix(I,n,n), F_A.P
            rankA = pseudo_rank(diag(L11), ε_rank)
            F_L11 = qr(L11, Val(true))
            R11, Q2, P2 = F_L11.R, F_L11.Q*Matrix(I,W.t,W.t), F_L11.P
            p_gn[:], F_J2 = gn_search_direction(J, rx, C.cx, Q1, P1, L11, Q2, P2, R11, rankA, W.t, ε_rank, iter_k)
            if !(W.t != rankA || iter_k.rankJ2 != min(m, n - rankA))
                second_lagrange_mult_estimate!(J, Q1, Matrix(L11'), P1, λ, rx, p_gn, W.t)
                s2 = check_constraint_deletion(W.q,C.A,λ,∇fx,C.scaling,C.diag_scale)
                if s2 != 0
                    index_s2 = W.active[s2]
                    deleteat!(λ, s2)
                    deleteat!(C.diag_scale, s)
                    C.cx = C.cx[setdiff(1:end, s2)]
                    delete_constraint!(W, s2)
                    iter_k.del = true
                    iter_k.index_del = index_s2
                    C.A = C.A[setdiff(1:end, s2),:]
                    Q1 = F_A.Q*Matrix(I,n,n)
                    vect_P1 = F_A.p[:]
                    P1, L11, Q1 = update_QR_A(Q1,F_A.R,vect_P1,s2,W.t)
                    rankA = pseudo_rank(diag(L11), ε_rank)
                    F_L11 = qr(L11, Val(true))
                    R11, Q2, P2 = F_L11.R, F_L11.Q*Matrix(I,W.t,W.t), F_L11.P
                    p_gn[:], F_J2 = gn_search_direction(J, rx, C.cx, Q1, P1, L11, Q2, P2, R11, rankA, W.t, ε_rank, iter_k)
                end
            end
        end
    # No first order estimate implies deletion of a constraint
    elseif s == 0
        L11, Q1, P1 = Matrix(transpose(F_A.R)), F_A.Q*Matrix(I,n,n), F_A.P
        rankA = pseudo_rank(diag(L11), ε_rank)
        F_L11 = qr(L11, Val(true))
        R11, Q2, P2 = F_L11.R, F_L11.Q*Matrix(I,W.t,W.t), F_L11.P
        p_gn[:], F_J2 = gn_search_direction(J, rx, C.cx, Q1, P1, L11, Q2, P2, R11, rankA, W.t, ε_rank, iter_k)
        if !(W.t != rankA || iter_k.rankJ2 != min(m, n - rankA))
            second_lagrange_mult_estimate!(J, Q1, Matrix(L11'), P1, λ, rx, p_gn, W.t)
            s2 = check_constraint_deletion(W.q,C.A,λ,∇fx,C.scaling,C.diag_scale)
            if s2 != 0
                index_s2 = W.active[s2]
                deleteat!(λ, s2)
                deleteat!(C.diag_scale, s)
                C.cx = C.cx[setdiff(1:end, s2)]
                delete_constraint!(W, s2)
                iter_k.del = true
                iter_k.index_del = index_s2
                C.A = C.A[setdiff(1:end, s2),:]
                Q1 = F_A.Q*Matrix(I,n,n)
                vect_P1 = F_A.p[:]
                P1, L11, Q1 = update_QR_A(Q1,F_A.R,vect_P1,s2,W.t)
                rankA = pseudo_rank(diag(L11), ε_rank)
                F_L11 = qr(L11, Val(true))
                R11, Q2, P2 = F_L11.R, F_L11.Q*Matrix(I,W.t,W.t), F_L11.P
                p_gn[:], F_J2 = gn_search_direction(J, rx, C.cx, Q1, P1, L11, Q2, P2, R11, rankA, W.t, ε_rank, iter_k)
            end
        end
    end
    iter_k.λ = λ
    return P1, L11, Q1, F_L11, F_J2
end

# INIALC
# Compute the first working set and penalty constants

function init_working_set(cx::Vector{Float64}, K::Array{Array{Float64,1},1}, step::Iteration, q::Int64, l::Int64)
    δ, ϵ, ε_rel = 0.1, 0.01, sqrt(eps(Float64))

    # Initialisation des pénalités
    K[:] = [δ * ones(l) for i = 1:length(K)]
    for i = 1:l
        pos = min(abs(cx[i]) + ϵ, δ)
        step.w[i] = pos
    end

    # Determination du premier ensemble actif
    active = zeros(Int64, l); inactive = zeros(Int64, l - q)
    t = q; lmt = 0

    # Les contraintes d'égalité sont toujours actives
    active[1:q] = [i for i = 1:q]
            
    for i = q + 1:l
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
    τ::Vector{Float64},
    ρ::Vector{Float64},
    ρ_prk::Float64,
    c1::Float64,
    pseudo_rk::Int64,
    previous_dimR::Int64,
    progress::Float64,
    predicted_linear_progress::Float64,
    prelin_previous_dim::Float64,
    previous_α::Float64)

    # Data

    stepb, pgb1, pgb2, predb, rlenb, c2 = 2e-1, 3e-1, 1e-1, 7e-1, 2.0, 1e2
    if ((previous_α < stepb) &&
        (progress <= pgb1 * predicted_linear_progress^2) &&
        (progress <= pgb2 * prelin_previous_dim^2))
        
        # Bad step
        dim = max(1, previous_dimR - 1)
        if ((previous_dimR > 1) && (ρ[dim] > c1 * ρ_prk))
            return dim
        end
    end

    dim = previous_dimR
    if (((ρ[dim] > predb * ρ_prk) && (rlenb * τ[dim] < τ[dim + 1])) ||
        (c2 * τ[dim] < τ[dim + 1]))
        suggested_dim = dim
    else
        i1 = previous_dimR - 1
        buff = [i for i = i1:previous_dimR if ρ[i] > predb * ρ_prk]
        suggested_dim = (isempty(buff) ? pseudo_rk : min(buff))
    end
    return suggested_dim
end

# PREGN
# Returns dimension to use when previous descent direction was computed with Gauss-Newton method

function gn_previous_step(
    τ::Vector{Float64},
    τ_prk::Float64,
    mindim::Int64,
    ρ::Vector{Float64},
    ρ_prk::Float64,
    pseudo_rank::Int64)

    # Data
    τ_max, ρ_min = 2e-1, 5e-1
    pm1 = pseudo_rank - 1
    if mindim > pm1
        suggested_dim = mindim
    else
        k = pm1
        while (τ[k] >= τ_max * τ_prk || ρ[k] <= ρ_min * ρ_prk) && k > mindim
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
    cx::Vector{Float64},
    λ::Vector{Float64},
    iter_km1::Iteration,
    scaling::Bool,
    diag_scale::Vector{Float64})

    # Data
    δ = 1e-1
    c1, c2, c3, c4, c5 = 5e-1, 1e-1, 4e0, 1e1, 5e-2
    ε_rel = eps(Float64)
    β_k = sqrt(d1nrm^2 + b1nrm^2)

    # If any of the following conditions is satisfied the Gauss-Newton direction is accepted
    # 1) The first iteration step
    # 2) estimated convergence factor < c1
    # 3) the real progress > c2 * predicted linear progress (provided we are not close to the solution)
    method_code = 1
    first_step = (iter_number == 0 && !restart)
    add_or_del = (constraint_added || constraint_deleted)
    conv_factor_lower_c1 = (β_k < c1 * iter_km1.β)
    progress_not_close = ((iter_km1.progress > c2 * iter_km1.predicted_reduction) && ((dnrm <= c3 * β_k)))
    if !(first_step || add_or_del || conv_factor_lower_c1 || progress_not_close)
        # Subspace minimization is suggested if one of the following holds true
        # 4) There is something left to reduce in subspaces 
        # 5) Addition and/or deletion to/from current working set in the latest step
        # 6) The nonlinearity is too sever

        method_code = -1
        non_linearity_k = sqrt(d1nrm * d1nrm + active_c_sum)
        non_linearity_km1 = sqrt(d1nrm_as_km1 + active_c_sum)
        to_reduce = false
        if W.q < W.t
            sqr_ε = sqrt(eps(Float64))
            rows = zeros(W.t-W.q)
            for i = W.q+1:W.t
                rows[i] = (scaling ? 1.0 / diag_scale[i] : diag_scale[i])
            end
            lagrange_mult_cond = any(>=(-sqr_ε), λ[W.q + 1:W.t] .* rows) && any(<(0), λ[W.q + 1:W.t])
            to_reduce = (to_reduce || lagrange_mult_cond)
        end
        if (W.l - W.t > 0)
            inact_c = [cx[W.inactive[j]] for j = 1:((W.l - W.t))]
            to_reduce = (to_reduce || any(<(δ), inact_c))
             end
        cond4 = active_c_sum > c2
        cond5 = (constraint_deleted || constraint_added || to_reduce || (W.t == n && W.t == rankA))
        ϵ = max(1e-2, 10.0 * ε_rel)
        cond6 = (W.l == W.q) && !((β_k < ϵ * dnrm) || (b1nrm < ϵ && m == n - W.t))

        if !(cond4 || cond5 || cond6)
            cond7 = (iter_km1.α < c5 && non_linearity_km1 < c2 * non_linearity_k) || m == n - W.t
            cond8 = !(dnrm <= c4 * β_k)
            if cond7 || cond8
                # Method of Newton is the only alternative
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
    y::Vector{Float64},
    previous_α::Float64,
    restart::Bool)

    # Data
    c1 = 0.1
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
                l_estim_righthand[i] = norm(l_estim_righthand[i - 1:i])
                l_estim_sd[i] = norm(l_estim_sd[i - 1:i])
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
            newdim = max(mindim, suggested_dim)
        end

        newdim = max(0, min(rankR, previous_dimR))
        if newdim != 0
            k = max(previous_dimR - 1, 1)
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
        rx::Vector{Float64},
        active_cx_sum::Float64,
        J1::Matrix{Float64},
        t::Int64,
        rankJ2::Int64,
        rankA::Int64,
        b::Vector{Float64},
        R11::Matrix{Float64},
        P2::Matrix{Float64},
        Q3::Matrix{Float64},
        R22::Matrix{Float64},
        previous_iter::Iteration,
        restart::Bool)

    # Data
    c1, c2, α_low = 0.1, 0.01, 0.2
    previous_α = previous_iter.α

    if rankA <= 0
        dimA = 0
        previous_dimA = 0
        η_A = 1.0
        d = -rx

    elseif rankA > 0
        previous_dimA = abs(previous_iter.dimA) + t - previous_iter.t
        nrm_b_asprev = norm(b[1:previous_dimA])
        nrm_b = norm(b)
        constraint_progress = dot(previous_iter.cx, previous_iter.cx) - active_cx_sum

        # Determine Dimension for matrix R11 to be used
        dimA, η_A = determine_solving_dim(previous_dimA, rankA, nrm_b, constraint_progress, nrm_b_asprev, UpperTriangular(R11), b, previous_α, restart)

        # Solve for p1 the system R11*P2*p1 = b
        # Using dimA columns of R11
        # Forms right hand side d = r(x)+J1*p1

        δp1 = UpperTriangular(R11[1:dimA,1:dimA]) \ b[1:dimA]
        p1 = P2[1:rankA,1:rankA] * [δp1;zeros(rankA - dimA)]
        d = -(rx + J1 * p1)
    end

    if rankJ2 > 0 d = transpose(Q3) * d end

    previous_dimJ2 = abs(previous_iter.dimJ2) + t - previous_iter.t
    nrm_d_asprev = norm(d[1:previous_dimJ2])
    nrm_d = norm(d)
    residual_progress = dot(previous_iter.rx, previous_iter.rx) - rx_sum
    dimJ2, η_J2 = determine_solving_dim(previous_dimJ2, rankJ2, nrm_d, residual_progress, nrm_d_asprev, UpperTriangular(R22), d, previous_α, restart)

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
        x::Vector{Float64},
        c::ConstraintsEval,
        r::ResidualsEval,
        rx::Vector{Float64},
        cx::Vector{Float64},
        active_cx::Vector{Float64},
        λ::Vector{Float64},
        rx_sum::Float64,
        active_cx_sum::Float64,
        p_gn::Vector{Float64},
        d_gn::Vector{Float64},
        b_gn::Vector{Float64},
        nrm_b1_gn::Float64,
        nrm_d1_gn::Float64,
        nrm_d_gn::Float64,
        J::Matrix{Float64},
        m::Int64,
        n::Int64,
        working_set::WorkingSet,
        rankA::Int64,
        rankJ2::Int64,
        P1::Matrix{Float64},
        Q1::Matrix{Float64},
        L11::Matrix{Float64},
        P2::Matrix{Float64},
        Q2::Matrix{Float64},
        R11::Matrix{Float64},
        P3::Matrix{Float64},
        Q3::Matrix{Float64},
        R22::Matrix{Float64},
        constraint_added::Bool,
        constraint_deleted::Bool,
        scaling::Bool,
        diag_scale::Vector{Float64},
        restart::Bool=false)


    prev_dimJ2m1 = previous_iter.dimJ2 + previous_iter.t - working_set.t - 1
    nrm_d1_asprev = norm(d_gn[1:prev_dimJ2m1])


    method_code, β = check_gn_direction(nrm_b1_gn,nrm_d1_gn,nrm_d1_asprev,nrm_d_gn,active_cx_sum,iter_number,rankA,n,m,restart,constraint_added,constraint_deleted,working_set,cx,λ,previous_iter,scaling,diag_scale)

    # Gauss-Newton accepted
    if method_code == 1
        dimA = rankA
        dimJ2 = rankJ2
        p, b, d = p_gn, b_gn, d_gn

    # Subspace minimization to recompute the search direction
    # using dimA columns of matrix R11 and dimJ2 columns of matrix R22
    elseif method_code == -1
        JQ1 = J * Q1
        J1, J2 = JQ1[:,1:rankA], JQ1[:,rankA + 1:end]
        b = -transpose(Q2) * transpose(P1) * active_cx
        dimA, dimJ2 = choose_subspace_dimensions(rx_sum,rx,active_cx_sum,J1,working_set.t,rankJ2,rankA,b,R11,P2,Q3,R22,previous_iter,restart)
        p, b, d = sub_search_direction(J1, rx, active_cx, Q1, P1, L11, Q2, P2, R11, Q3, R22, P3, n, working_set.t, rankA, dimA, dimJ2, method_code)



    # Search direction computed with the method of Newton
    elseif method_code == 2
        p = newton_search_direction(x, c, r, active_cx, working_set.active, n, m, working_set.l, working_set.t, λ, rx, J, Q1, P1, L11, Q2, R11, P2, rankA)
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

function psi(
    x::Vector{Float64},
    α::Float64,
    p::Vector{Float64},
    r::ResidualsEval,
    c::ConstraintsEval,
    w::Vector{Float64},
    m::Int64,
    l::Int64,
    t::Int64,
    active::Vector{Int64},
    inactive::Vector{Int64})


    r.ctrl, c.ctrl = 1, 1

    r_new, c_new = zeros(m), zeros(l)
    dummy = zeros((1, 1))

    penalty_constraint_sum = 0.0
    x_new = x + α * p
    r(x_new, r_new, dummy)
    c(x_new, c_new, dummy)
    # First part of sum with active constraints
    for i = 1:t
        j = active[i]
        penalty_constraint_sum += w[j] * c_new[j]^2
    end

    # Second part of sum with inactive constraints
    for i = 1:l - t
        j = inactive[i]
        if c_new[j] < 0.0
            penalty_constraint_sum  += w[j] * c_new[j]^2
        end
    end
    return 0.5 * (dot(r_new, r_new) + penalty_constraint_sum)
end

# ASSORT

function assort!(
    K::Array{Array{Float64,1},1},
    w::Vector{Float64},
    t::Int64,
    active::Vector{Int64})

    for i in 1:t, ii in 1:4
        k = active[i]
            if w[k] > K[ii][k]
            for j = 4:-1:ii + 1
                K[j][k] = K[j - 1][k]
            end
            K[ii][k] = w[k]
        end
    end
    return
end


# EUCMOD
# Solve the problem :
#
#     min ||w||      (euclidean norm)
# s.t.
#     w_i ≧ w_old_i
#
#     <y,w> ≧ τ  (if ctrl = 2)
#
#     <y,w> = τ  (if ctrl = 1)

function min_norm_w!(
    ctrl::Int64,
    w::Vector{Float64},
    w_old::Vector{Float64},
    y::Vector{Float64},
    τ::Float64,
    pos_index::Vector{Int64},
    nb_pos::Int64)

    w[:] = w_old
    if nb_pos > 0
        y_sum = dot(y, y)
        y_norm = norm(y)
        # Scale the vector y
        if y_norm != 0.0 y /= y_norm end
        τ_new = τ
        s = 0.0
        n_runch = nb_pos
        terminated = false
        while !terminated
            τ_new -= s
            c = (norm(y, Inf) <= eps(Float64) ? 1.0 : τ_new / y_sum)
            y_sum, s = 0.0, 0.0
            i_stop = n_runch
            k = 1
            while k <= n_runch
                i = pos_index[k]
                buff = c * y[k] * y_norm
                if buff >= w_old[i]
                    w[i] = buff
                    y_sum += y[k]^2
                    k += 1
                else
                    s += w_old[i] * y[k] * y_norm
                    n_runch -= 1
                    for j = k:n_runch
                        pos_index[j] = pos_index[j + 1]
                        y[j] = y[j + 1]
                    end
                end
            end
            y_sum *= y_norm * y_norm
            terminated = (n_runch <= 0 ) || (ctrl == 2) || (i_stop == n_runch)
        end

    end
    return
end


# EUCNRM
# Update the penalty constants using the euclidean norm

function euclidean_norm_weight_update!(
    vA::Vector{Float64},
    cx::Vector{Float64},
    active::Vector{Int64},
    t::Int64,
    μ::Float64,
    dimA::Int64,
    previous_w::Vector{Float64},
    w::Vector{Float64},
    K::Array{Array{Float64,1},1})

    if t == 0
        # if no active constraints, previous penalty weights are used
        w[:] = previous_w
    else
        # Compute z = [<∇c_i(x),p>^2]_i for i ∈ active
        z = vA.^2
        # Compute ztw = z(TR)w_old where w_old holds the 4th lowest weights used so far
        # for constraints in active set
        w_old = K[4]
        ztw = dot(z, w_old[active[1:t]])
        pos_index = zeros(Int64, t)

        if (ztw >= μ) && (dimA < t)

            # if ztw ≧ μ, no need to change w_old unless t = dimA
            y = zeros(t)
            # Form vector y and scalar γ (\gamma)
            # pos_index holds indeces for the y_i > 0
            ctrl, nb_pos, γ = 2, 0, 0.0
            for i = 1:t
                k = active[i]
                y_elem = vA[i] * (vA[i] + cx[k])
                if y_elem > 0
                    nb_pos += 1
                    pos_index[nb_pos] = k
                    y[nb_pos] = y_elem
                else
                    γ -= y_elem * w_old[k]
                end
            end
            min_norm_w!(ctrl, w, w_old, y, γ, pos_index, nb_pos)
        elseif (ztw < μ) && (dimA < t)

            # Form vector e and scalar τ (\tau)
            e = zeros(t)
            ctrl, nb_pos, τ = 2, 0, μ
            for i = 1:t
                k = active[i]
                e_elem = -vA[i] * cx[k]
                if e_elem > 0
                    nb_pos += 1
                    pos_index[nb_pos] = k
                    e[nb_pos] = e_elem
                else
                    τ -= e_elem * w_old[k]
                end
            end
            min_norm_w!(ctrl, w, w_old, e, τ, pos_index, nb_pos)
        elseif (ztw < μ) && (dimA == t)

            # Use vector z already formed (z = [<∇c_i(x),p>^2]_i for i ∈ active)
            # pos_index holds the indeces in active since z elements are > 0
            ctrl = 1
            pos_index[:] = active[1:t]
            min_norm_w!(ctrl, w, w_old, z, μ, pos_index, t)
        end
        assort!(K, w, t, active)
    end
    return
end


# MAXNRM
# Update the penalty weights corresponding to the
# constraints in the current working setb

function max_norm_weight_update!(
        nrm2_Ap::Float64,
        rmy::Float64,
        α_w::Float64,
        δ::Float64,
        w::Vector{Float64},
        active::Vector{Int64},
        t::Int64,
        K::Array{Array{Float64,1},1})
    μ = (abs(α_w - 1.0) <= δ ? 0.0 : rmy / nrm2_Ap)
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
                for j = 4:-1:i + 1
                    K[j][1] = K[j - 1][1]
                end
                K[i][1] = μ
                mu_not_placed = false
            end
            i += 1
        end
    end
    return
end

# WEIGHT
# Determine the penalty constants that should be used in the current linesearch
# where ψ(α) is approximalety minimized

function penalty_weight_update(
        w_old::Vector{Float64},
        Jp::Vector{Float64},
        Ap::Vector{Float64},
        K::Array{Array{Float64,1},1},
        rx::Vector{Float64},
        cx::Vector{Float64},
        active::Vector{Int64},
        t::Int64,
        dimA::Int64,
        norm_code::Int64)
    # Data
    δ = 0.25
    w = w_old[:]

    nrm2_Ap = dot(Ap, Ap)
    nrm2_Jp = dot(Jp, Jp)
    Jp_rx = dot(Jp, rx)
            
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
    elseif norm_code == 2
        euclidean_norm_weight_update!(Ap, cx, active, t, rmy, dimA, w_old, w, K)
    end
    #                               T                       T
    # Computation of ψ'(0) = [J(x)p] r(x)+   Σ      w_i*[∇c_i(x) p]c_i(x)
    #                                     i ∈ active
    BtwA = 0.0
    for i = 1:t
        k = active[i]
        BtwA += w[k] * Ap[i] * cx[k]
    end
    dψ0 = BtwA + Jp_rx
    return w, dψ0
end


# CONCAT
# Compute in place the components of vector v used for polynomial minimization

function concatenate!(v::Vector{Float64},
                      rx::Vector{Float64},
                      cx::Vector{Float64},
                      w::Vector{Float64},
                      m::Int64,
                      t::Int64,
                      l::Int64,
                      active::Vector{Int64},
                      inactive::Vector{Int64})

    v[1:m] = rx[:]
    if t != 0
        for i = 1:t
            k = active[i]
            v[m + k] = sqrt(w[k]) * cx[k]
        end
end
    if l != 0
        for j = 1:l - t
            k = inactive[j]
            v[m + k] = (cx[k] > 0 ? 0.0 : sqrt(w[k]) * cx[k])
        end
    end
    return
end

# LINC2
# Compute in place vectors v0 and v2 so that one dimensional minimization in R^m can be done
# Also modifies components of v1 related to constraints

function coefficients_linesearch!(v0::Vector{Float64},
                                 v1::Vector{Float64},
                                 v2::Vector{Float64},
                                 α_k::Float64,
                                 rx::Vector{Float64},
                                 cx::Vector{Float64},
                                 rx_new::Vector{Float64},
                                 cx_new::Vector{Float64},
                                 w::Vector{Float64},
                                 m::Int64,
                                 t::Int64,
                                 l::Int64,
                                 active::Vector{Int64},
                                 inactive::Vector{Int64})

    # Compute v0
    concatenate!(v0, rx, cx, w, m, t, l, active, inactive)

    v_buff = zeros(m + l)
    concatenate!(v_buff, rx_new, cx_new, w, m, t, l, active, inactive)

    # Computation of v2 components
    v2[:] = ((v_buff - v0) / α_k - v1) / α_k
    return
end


# Equivalent Fortran : QUAMIN in dblreduns.f

function minimize_quadratic(x1::Float64, y1::Float64,
                            x2::Float64, y2::Float64,
                            x3::Float64, y3::Float64)

    d1, d2 = y2 - y1, y3 - y1
    s = (x3 - x1)^2 * d1 - (x2 - x1)^2 * d2
    q = 2 * ((x2 - x1) * d2 - (x3 - x1) * d1)
    return x1 - s / q
end


# Equivalent Fortran : MINRN in dblreduns.f


function minrn(x1::Float64, y1::Float64,
               x2::Float64, y2::Float64,
               x3::Float64, y3::Float64,
               α_min::Float64,
               α_max::Float64,
               p_max::Float64)

    ε = sqrt(eps(Float64)) / p_max

    # α not computable
    # Add an error in this case
    if abs(x1 - x2) < ε || abs(x3 - x1) < ε || abs(x3 - x2) < ε
        α, pα = 0., 0.

    else
    # Compute minimum of quadradic passing through y1, y2 and y3
    # respectively at points x1, x2 and x3
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



function parameters_rm(
    v0::Vector{Float64},
    v1::Vector{Float64},
    v2::Vector{Float64},
    x_min::Float64,
    ds::Polynomial{Float64},
    dds::Polynomial{Float64})

    dds_best = dds(x_min)
    η, d = 0.1, 1.
    normv2 = dot(v2, v2)
    h0 = abs(ds(x_min) / dds_best)
    Dm = abs(6 * dot(v1, v2) + 12 * x_min * normv2) + 24 * h0 * normv2
    hm = max(h0, 1)

    # s'(α) = 0 is solved analytically
    if dds_best * η < 2 * Dm * hm

        # If t = α+a1 solves t^3 + b*t + c = O then α solves s'(α) = 0
        (a3, a2, a1) = coeffs(ds) / (2 * normv2)

        b = a2 - (a1^2) / 3
        c = a3 - a1 * a2 / 3 + 2 * (a1 / 3)^3
        d = (c / 2)^2 + (b / 3)^3
        # Two interisting roots
        if d < 0
            α_hat, β_hat = two_roots(b, c, d, a1, x_min)

        # Only one root is computed
        else
    α_hat = one_root(c, d, a1)
        end

    # s'(α) = 0 is solved using Newton-Raphson's method
    else
        α_hat = newton_raphson(x_min, Dm, ds, dds)
    end

    # If only one root computed
    if d >= 0
        β_hat = α_hat
    end
    return α_hat, β_hat

end

function bounds(α_min::Float64, α_max::Float64, α::Float64, s::Polynomial{Float64})
    α = min(α, α_max)
    α = max(α, α_min)
    return α, s(α)
end

function newton_raphson(
    x_min::Float64,
    Dm::Float64,
    ds::Polynomial{Float64},
    dds::Polynomial{Float64})

    α, newton_iter = x_min, 0
    ε, error = 1e-4, 1.
    while error > ε || newton_iter < 3
        c = dds(α)
        h = -ds(α) / c
        α += h
        error = (2 * Dm * h^2) / abs(c)
        newton_iter += 1
    end
    return α
end


# Equivalent Fortran : ONER in dblreduns.f
function one_root(c::Float64, d::Float64, a::Float64)
    arg1, arg2 = -c / 2 + sqrt(d), -c / 2 - sqrt(d)
    return cbrt(arg1) + cbrt(arg2) - a / 3
end

# Equivalent Fortran : TWOR in dblreduns.f
function two_roots(b::Float64, c::Float64, d::Float64, a::Float64, x_min::Float64)
    φ = acos(abs(c / 2) / (-b / 3)^(3 / 2))
    t = (c <= 0 ? 2 * sqrt(-b / 3) : -2 * sqrt(-b / 3))

    # β1 is the global minimizer of s(α).
    # If d is close to zero the root β1 is stable while β2 and β3 become unstable
    β1 = t * cos(φ / 3) - a / 3
    β2 = t * cos((φ + 2 * π) / 3) - a / 3
    β3 = t * cos((φ + 4 * π) / 3) - a / 3

    # Sort β1, β2 and β3 so that β1 <= β2 <= β3
    β1, β2, β3 = sort([β1, β2, β3])

    # β1 or β3 are now the roots of interest
    α, β = (x_min <= β2 ? (β1, β3) : (β3, β1))
    return α, β
end


# Equivalent Fortran : MINRM in dblreduns.f
function minrm(
    v0::Vector{Float64},
    v1::Vector{Float64},
    v2::Vector{Float64},
    x_min::Float64,
    α_min::Float64,
    α_max::Float64)

    s = Polynomial([0.5 * dot(v0, v0), dot(v0, v1), dot(v0, v2) + 0.5 * dot(v1, v1), dot(v1, v2), 0.5 * dot(v2, v2)])
    ds = derivative(s)
    dds = derivative(ds)
    α_hat, β_hat = parameters_rm(v0, v1, v2, x_min, ds, dds)
    sα, sβ = s(α_hat), s(β_hat)
    α_old = α_hat
    α_hat, sα = bounds(α_min, α_max, α_hat, s)
    if α_old == β_hat
        β_hat, sβ = α_hat, s(α_hat)
    else
        β_hat, sβ = bounds(α_min, α_max, β_hat, s)
    end
    return α_hat, sα, β_hat, sβ
end


# REDC
# Returns true if essential reduction in the objective function is likely
# Otherwise returns false

# TODO: rajouter la sélection du meilleur entre α et α_k dans la fonction principale
# if ψ_k < ψ_α
#     α = α_k
#     ψ_α = ψ_k
# end

function check_reduction(
    ψ_α::Float64,
    ψ_k::Float64,
    approx_k::Float64,
    η::Float64,
    diff_psi::Float64)

    # Data
    δ = 0.2

    if ψ_α - approx_k >= η * diff_psi
        reduction_likely = !((ψ_α - ψ_k < η * diff_psi) && (ψ_k > δ * ψ_α))
    else
        reduction_likely = false
    end
    return reduction_likely
end


# GAC
# Halfs the value of u until a Goldstein-Armijo condition is satisfied
# or until steplength times search direction is below square root of relative_prevision

function goldstein_armijo_step(
    ψ0::Float64,
    dψ0::Float64,
    α_min::Float64,
    τ::Float64,
    p_max::Float64,
    x::Vector{Float64},
    α0::Float64,
    p::Vector{Float64},
    r::ResidualsEval,
    c::ConstraintsEval,
    w::Vector{Float64},
    m::Int64,
    l::Int64,
    t::Int64,
    active::Vector{Int64},
    inactive::Vector{Int64})

    u = α0
    sqr_ε = sqrt(eps(Float64))
    exit = (p_max * u < sqr_ε) || (u <= α_min)
    ψu = psi(x, u, p, r, c, w, m, l, t, active, inactive)
    while !exit && (ψu > ψ0 + τ * u * dψ0)
        u *= 0.5
        ψu = psi(x, u, p, r, c, w, m, l, t, active, inactive)
        exit = (p_max * u < sqr_ε) || (u <= α_min)
    end
    return u
end


# LINEC
# Linesearch routine for constrained least squares problems
# Compute the steplength α (\alpha) for the iteration x_new = x + αp
# x current point, p search direction
#
# α is close to the solution of the problem
# min ψ(α)
# with α_low <= α <= α_upp
#
# ψ(α) = 0.5 * (||r(x+αp)||^2 + Σ (w_i * c_i(x+αp)^2) +  Σ min(0,w_j * c_j(x+αp))^2)
#                               i                        j
# i correspond to constraints in current working set, j to inactive constraints

function linesearch_constrained(
    x::Vector{Float64},
    α0::Float64,
    p::Vector{Float64},
    r::ResidualsEval,
    c::ConstraintsEval,
    rx::Vector{Float64},
    cx::Vector{Float64},
    JpAp::Vector{Float64},
    w::Vector{Float64},
    m::Int64,
    l::Int64,
    t::Int64,
    active::Vector{Int64},
    inactive::Vector{Int64},
    ψ0::Float64,
    dψ0::Float64,
    α_low::Float64,
    α_upp::Float64)

    # Only evalutations for residuals and constraints
    r.ctrl = 1
    c.ctrl = 1
    dummy = zeros((1, 1))

    # LINC1
    # Set values of constants and compute α_min, α_max and α_k

    η = 0.3 # \eta
    τ = 0.25 # \tau
    γ = 0.4 # \gamma

    α_min, α_max = α_low, α_upp
    α_k = min(α0, α_max)
    α_km1 = 0.0
    ψ_km1 = ψ0
    p_max = norm(p, Inf)

    # LINC2
    # Computation of v1
    v1 = JpAp
    if t != 0
        for i = 1:t
            k = active[i]
            v1[m + k] = sqrt(w[k]) * v1[m + k]
        end
    end
    if l - t != 0
        for j = 1:l - t
            k = inactive[j]
            v1[m + k] = (cx[k] > 0 ? 0.0 : sqrt(w[k]) * v1[m + k])
        end
    end

    ψ_k = psi(x, α_k, p, r, c, w, m, l, t, active, inactive)

    diff_psi = ψ0 - ψ_k

    rx_new, cx_new = zeros(m), zeros(l)
    r(x + α_k * p, rx_new, dummy)
    c(x + α_k * p, cx_new, dummy)

    v0, v2 = zeros(m + l), zeros(m + l)
    coefficients_linesearch!(v0, v1, v2, α_k, rx, cx, rx_new, cx_new, w, m, t, l, active, inactive)

    # Set x_min = the best of the points 0 and α0

    x_min = (diff_psi >= 0 ? α_k : 0.0)

    # Minimize in R^m. Use two points 0 and α0
    # New suggestion of steplength is α_kp1 (stands for "k+1")
    # pk is the value of the approximating function at α_kp1

    α_kp1, pk, β, pβ = minrm(v0, v1, v2, x_min, α_min, α_max)


    if α_kp1 != β && pβ < pk && β <= α_k
        α_kp1 = β
        pk = pβ
    end

    # UPDATE

    α_km2 = α_km1
    ψ_km2 = ψ_km1
    α_km1 = α_k
    ψ_km1 = ψ_k
    α_k = α_kp1
    ψ_k = psi(x, α_k, p, r, c, w, m, l, t, active, inactive)

    # Test termination condition at α0

    if (-diff_psi <= τ * dψ0 * α_km1) || (ψ_km1 < γ * ψ0)
        # Termination condition satisfied at α0

        diff_psi = ψ0 - ψ_k

        # REDUCE
        # Check if essential reduction is likely
        reduction_likely = check_reduction(ψ_km1, ψ_k, pk, η, diff_psi)

        while reduction_likely
            # Value of the objective function can most likely be reduced
            # Minimize in R^n using 3 points : α_km2, α_km1 and α_k
            # New suggestion of the steplength is α_kp1, pk is its approximated value
            α_kp1, pk = minrn(α_k, ψ_k, α_km1, ψ_km1, α_km2, ψ_km2, α_min, α_max, p_max)

            # UPDATE
            α_km2 = α_km1
            ψ_km2 = ψ_km1
            α_km1 = α_k
            ψ_km1 = ψ_k
            α_k = α_kp1
            ψ_k = psi(x, α_k, p, r, c, w, m, l, t, active, inactive)
            diff_psi = ψ0 - ψ_k
            reduction_likely = check_reduction(ψ_km1, ψ_k, pk, η, diff_psi)
        end

        # Terminate but choose the best point out of α_km1 and α_k
        if (ψ_km1 - pk >= η * diff_psi) && (ψ_k < ψ_km1)
            α_km1 = α_k
            ψ_km1 = ψ_k
        end
    # Termination condition not satisfied at α0
    else
        diff_psi = ψ0 - ψ_k
        # Test termination condition at α1, i.e. α_k
        if (-diff_psi <= τ * dψ0 * α_k) || (ψ_k < γ * ψ0)
            # Termination condition satisfied at α1
            # Check if α0 is somewhat good
            if ψ0 <= ψ_km1
                x_min = α_k
                r(x + α_k * p, rx_new, dummy)
                c(x + α_k * p, cx_new, dummy)
                v0, v2 = zeros(m + l), zeros(m + l)
                coefficients_linesearch!(v0, v1, v2, α_k, rx, cx, rx_new, cx_new, w, m, t, l, active, inactive)
                α_kp1, pk, β, pβ = minrm(v0, v1, v2, x_min, α_min, α_max)
                if α_kp1 != β && pβ < pk && β <= α_k
                    α_kp1 = β
                    pk = pβ
                end
                α_km1 = 0.0
                ψ_km1 = ψ0

            else
                # Minimize in R^n. use 3 points : 0, α0 and α1
                # New suggestion of the steplength is α_kp1
                # pk is the value of the approximating function at α_kp1
                α_kp1, pk = minrn(α_k, ψ_k, α_km1, ψ_km1, α_km2, ψ_km2, α_min, α_max, p_max)
            end
            diff = ψ0 - ψ_k

            # UPDATE
            α_km2 = α_km1
            ψ_km2 = ψ_km1
            α_km1 = α_k
            ψ_km1 = ψ_k
            α_k = α_kp1
            ψ_k = psi(x, α_k, p, r, c, w, m, l, t, active, inactive)

            # Check if essential reduction is likely
            reduction_likely = check_reduction(ψ_km1, ψ_k, pk, η, diff_psi)

            while reduction_likely
                # Value of the objective function can most likely be reduced
                # Minimize in R^n using 3 points : α_km2, α_km1 and α_k
                # New suggestion of the steplength is α_kp1, pk its approximated value
                α_kp1, pk = minrn(α_k, ψ_k, α_km1, ψ_km1, α_km2, ψ_km2, α_min, α_max, p_max)

                # UPDATE
                α_km2 = α_km1
                ψ_km2 = ψ_km1
                α_km1 = α_k
                ψ_km1 = ψ_k
                α_k = α_kp1
                ψ_k = psi(x, α_k, p, r, c, w, m, l, t, active, inactive)

                reduction_likely = check_reduction(ψ_km1, ψ_k, pk, η, diff_psi)
            end
            # Terminate but choose the best point out of α_km1 and α_k
            if (ψ_km1 - pk >= η * diff_psi) && (ψ_k < ψ_km1)
                α_km1 = α_k
                ψ_km1 = ψ_k
            end

        else
            # Take a pure Goldstein-Armijo step
            α_km1 = goldstein_armijo_step(ψ0, dψ0, α_min, τ, p_max, x, α_k, p, r, c, w, m, l, t, active, inactive)
        end
    end
    α = α_km1
    return α
end

# UPBND
# Determine the upper bound of the steplength

function upper_bound_steplength(
        A::Matrix{Float64},
        cx::Vector{Float64},
        p::Vector{Float64},
        inactive::Vector{Int64},
        t::Int64,
        l::Int64,
        index_del::Int64
    )

    α_upper = Inf
    if norm(inactive, Inf) > 0
        for i = 1:l - t
            j = inactive[i]
            if j != index_del
                ∇cjTp = dot(A[j,:], p)
                α_j = -cx[j] / ∇cjTp
                if cx[j] > 0 && ∇cjTp < 0 && α_j < α_upper
                    α_upper = α_j
                end
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
    iter::Iteration,
    x::Vector{Float64},
    r::ResidualsEval,
    rx::Vector{Float64},
    J::Matrix{Float64},
    p::Vector{Float64},
    c::ConstraintsEval,
    cx::Vector{Float64},
    A::Matrix{Float64},
    active_constraint::Constraint,
    w_old::Vector{Float64},
    work_set::WorkingSet,
    K::Array{Array{Float64,1},1},
    dimA::Int64,
    m::Int64,
    ind_constraint_del::Int64,
    previous_α::Float64,
    prev_rankJ2::Int64,
    rankJ2::Int64,
    method_code::Int64,
    weight_code::Int64)

    # Data
    error = 0
    c1 = 1e-3
    rx_sum = dot(rx, rx)
    Jp = J * p
    Ap = A * p
    JpAp = vcat(Jp, Ap)
    active_Ap = (active_constraint.A) * p
    if active_constraint.scaling
        active_Ap = active_Ap ./ active_constraint.diag_scale
    end
    active_index = work_set.active[1:work_set.t]
    if method_code != 2
        # Compute penalty weights and derivative of ψ at α = 0
        w, dψ0 = penalty_weight_update(w_old,Jp,active_Ap,K,rx,cx, work_set.active, work_set.t, dimA, weight_code)

        #
        # Compute ψ(0) = 0.5 * [||r(x)||^2 +    Σ     (w_i*c_i(x)^2)]
        #                                   i ∈ active
        ψ0 = 0.5 * (dot(rx, rx) + dot(w[active_index], cx[active_index].^2))
        # check is p is a descent direction
        if dψ0 >= 0 error = -1 end
        # TODO : handle error due to ψ'(0) > 0

        # Determine upper bound of the steplength
        α_upp = upper_bound_steplength(A, cx, p, work_set.inactive, work_set.t, work_set.l, ind_constraint_del)
        α_low = α_upp / 3000.0

        # Determine a first guess of the steplength
        magfy = (rankJ2 < prev_rankJ2 ? 6.0 : 3.0)
        α0 = min(1.0, magfy * previous_α, α_upp)
       # Compute the steplength
        α = linesearch_constrained(x, α0, p, r, c, rx, cx, JpAp, w, m, work_set.l, work_set.t, work_set.active, work_set.inactive, ψ0, dψ0, α_low, α_upp)


        # Compiutation of predicted linear progress
        uppbound = min(1.0,α_upp)
        atwa = dot(w[active_index],active_Ap.^2)
        iter.predicted_reduction = uppbound * (-2.0 * dot(Jp,rx) - uppbound * dot(Jp,Jp) + (2.0-uppbound^2) * atwa)

        # Computation of new point and actual progress
        r.ctrl = 1
        c.ctrl = 1
        rx_new = zeros(m)
        cx_new = zeros(work_set.l)
        r(x+α*p,rx_new,J)
        c(x+α*p,cx_new,A)
        whsum = dot(w[active_index], cx_new[active_index].^2)
        iter.progress = 2*ψ0 - dot(rx_new,rx_new) - whsum
        
    else
        w = w_old
        α = 1.0
    end
    return α, w
end

# TERCRI
# Check if any of the termination criteria are satisfied
# There are convergence criteria and abnormal termination criteria
# Convergence criteria
# 1) ||active_c(x)|| < ε_c (for constraints in the working set)
# 1.5) all inactive constraints must be > 0
# 2) ||active_A^T * λ - ∇f(x)|| < sqrt(ε_rel)(1 + ||∇f(x)||)
# 3) min λ_i >= ε_rel * max |λ_i|
#     i                  i
#            >= ε_rel * (1+||r(x)||^2) (if 1 inequality)
# 4) ||d1||^2 <= ε_x * ||x||
# 5) ||r(x)||^2 <= ε_abs^2
# 6) ||x_previous - x|| < ε_x * ||x||
# 7) sqrt(ε_rel) / ||p_gn|| > 0.25 (gn for Gauss-Newton)
# 8) The last digit in the convergence code has a specific value (TODO : not implemented yet)

# Abnormal termination criterias
# 9) number of iterations exceeds max_iter
# 10) Convergence to a non feasible point
# 11) 2nd derivatives not allowed by the user (TOTO (?) : not implemented yet)
# 12) Newton step fails
# 13) The latest direction is not a descent direction to the merit function (TODO : not implemented yet)

# Returns exit_code, an integer containing whose value gives info about termination
# 0 if no termination criterion is satisfied
# 10000 if criterion 4 satisfied
#  2000 if criterion 5 satisfied
#   300 if criterion 6 satisfied
#    40 if criterion 7 satisfied
#    -2 if criterion 9 satisfied
#    -5 if criterion 12 satisfied
#   -10 if not possible to satisfy the constraints

# exit_code != 0 means the termination of the algorithm

function check_termination_criteria(
    iter::Iteration,
    prev_iter::Iteration,
    W::WorkingSet,
    active_C::Constraint,
    x::Vector{Float64},
    cx::Vector{Float64},
    rx_sum::Float64,
    ∇fx::Vector{Float64},
    max_iter::Int64,
    nb_iter::Int64,
    ε_abs::Float64,
    ε_rel::Float64,
    ε_x::Float64,
    ε_c::Float64)

    exit_code = 0
    alfnoi = ε_rel / (norm(iter.p) + ε_abs)

    # Preliminary conditions
    preliminary_cond = !(iter.restart || (iter.code == -1  && alfnoi <= 0.25))
    if preliminary_cond

        # Check necessary conditions
        grad_res = norm(transpose(active_C.A) * iter.λ - ∇fx)
        necessary_crit = (!iter.del) && (norm(active_C.cx) < ε_c) && (grad_res < sqrt(ε_rel) * (1 + norm(∇fx)))
        if W.l - W.t > 0
            inactive_index = W.inactive[1:W.l - W.t]
            inactive_cx = cx[inactive_index]
            necessary_crit = necessary_crit && (all(>(0), inactive_cx))
        end

        if W.t > W.q
            if W.t == 1
                factor = 1 + rx_sum
            elseif W.t > 1
                factor =  maximum(map(abs, iter.λ))
            end
            lagrange_mult_pos = [iter.λ[i] for i = W.q + 1:W.t if iter.λ[i] > 0]
            sigmin = (isempty(lagrange_mult_pos) ? 0 : minimum(lagrange_mult_pos))
            necessary_crit = necessary_crit && (sigmin >= ε_rel * factor)

        end

        if necessary_crit
            # Check the sufficient conditions
            d1 = @view iter.d_gn[1:iter.dimJ2]
            x_diff = norm(prev_iter.x - x)

            # Criterion 4
            if dot(d1, d1) <= rx_sum * ε_rel^2
                exit_code += 10000
            end
            # Criterion 5
            if rx_sum <= ε_abs^2
                exit_code += 2000
            end
            # Criterion 6
            if x_diff < ε_x * norm(x)
                exit_code += 300
            end
            # Criterion 7
            if alfnoi > 0.25
                exit_code += 40
            end

        end
    end

    if exit_code == 0
        # Check abnormal termination criteria
        x_diff = norm(prev_iter.x - iter.x)
        Atcx_nrm = norm(transpose(active_C.A) * active_C.cx)
        # Criterion 9
        if nb_iter >= max_iter
            exit_code = -2
        # test if impossible to satisfy the constraints
        elseif x_diff <= 10.0 * ε_x && Atcx_nrm <= 10.0 * ε_c
            exit = -10
        end
        # TODO : implement critera 10-12
    end
    return exit_code
end

# OUTPUT
# Print the useful informations at the end of current iteration

function output!(
    io::IOStream,
    iter::Iteration,
    W::WorkingSet,
    nb_iter::Int64,
    β_prev::Float64,
    cx_sum::Float64)

    if norm(W.active, Inf) > 0
        s_act = "("
        for i = 1:W.t
            s_act = (i < W.t ? string(s_act, W.active[i], ",") : string(s_act, W.active[i], ")"))
        end
    else
        s_act = " -"
    end
    speed = (nb_iter == 0 ? 0.0 : iter.β / β_prev)
    @printf(io, "  %2d  %e  %.2e  %9.2e   %.3e  %2d    %2d   %.2e    %.2e     %.2e    %s\n",  nb_iter, dot(iter.rx, iter.rx), cx_sum, iter.progress, norm(iter.p), iter.dimA, iter.dimJ2, iter.α, speed, maximum(iter.w), s_act)
end

function final_output!(
    io::IOStream,
    iter::Iteration,
    W::WorkingSet,
    exit_code::Int64,
    nb_iter::Int64)

    @printf(io, "\nExit code = %d\nNumber of iterations = %d \n\n", exit_code, nb_iter)
    print(io,"Terminated at point :")
    (t -> @printf(io," %e ", t)).(iter.x)
    print(io,"\n\nActive constraints :")
    (i -> @printf(io," %d ", i)).(W.active[1:W.t])
    println(io,"\nConstraint values : ")
    (t -> @printf(io, " %.2e ", t)).(iter.cx)
    println(io,"\nPenalty constants :")
    (t -> @printf(io, " %.2e ", t)).(iter.w)

    @printf(io, "\nSquare sum of residuals = %e\n\n", dot(iter.rx, iter.rx))
end

##### ENLSIP 0.3.0 #####

mutable struct ENLSIP
    sol::Vector
    obj_value::Float64
end


"""
    enlsip(x0, r, c, n, m, q, l, scaling = false, weight_code = 2, MAX_ITER = 100)

Main function for ENLSIP solver. 

Must be called with the following arguments: 

* `x0::Vector{Foat64}` is the departure point

* `r` is a function of type [`ResidualsEval`](@ref) to evaluate residuals values and jacobian

* `c` is a function of type [`ConstraintsEval`] (@ref) to evaluate constraints values and jacobian

* `n::Int64` is the number of parameters

* `m::Int64` is the number of residuals 

* `q::Int64` is the number of equality constraints 

* `l::Int64` is the total number of constraints (equalities and inequalities)

The following arguments are optionnal and have default values:

* `scaling::Bool`

    -  boolean indicating if internal scaling of constraints value and jacobian must be done or not

    - `false` by default
      
* `weight_code::Int64` is an int representing the method used to compute penality weights at each iteration

    - `1` represents maximum norm method

    - `2` (default value) represents euclidean norm method
          
* `MAX_ITER::Int64`
     
    - int defining the maximum number of iterations

    - equals `100` by default

* `name_output::String`

    - string indicating the prefix of the `.txt` file produced at the end of the algorithm that contains informations about each iteration and termination of the algorithm

    - default name given is "output"
"""
function enlsip(x0::Vector{Float64},
    r::ResidualsEval,c::ConstraintsEval,
    n::Int64,m::Int64,q::Int64,l::Int64;
    scaling::Bool=false, weight_code::Int64=2, MAX_ITER::Int64=100)

    output_file = "output.txt"
    io = open(output_file, "w")
    println(io,"\n****************************************")
    println(io,"*                                      *")
    println(io,"*          ENLSIP-JULIA-0.3.0          *")
    println(io,"*                                      *")
    println(io,"****************************************\n")
    println(io,"Starting point : $x0\n")
    println(io,"Number of equality constraints   : $q\nNumber of inequality constraints : $(l-q)")
    println(io, "Constraints internal scaling     : $scaling\n")
    println(io,"iter    objective    cx_sum   reduction     ||p||   dimA  dimJ2     α     conv. speed   max weight   working set")
    ε_float = eps(Float64)
    ε_abs = ε_float
    ε_rel = sqrt(ε_float)
    ε_x = sqrt(ε_float)
    ε_c = sqrt(ε_float)

    nb_iteration = 0
    nb_eval = 0

    # Vector of penalty constants
    K = [zeros(l) for i = 1:4]

    # Evaluate at starting point
    rx, cx = zeros(m), zeros(l)
    J, A = zeros(m, n), zeros(l, n)
    r.ctrl = 1
    c.ctrl = 1
    r(x0, rx, J)
    c(x0, cx, A)
    nb_eval += 1
    # First Iteration
    first_iter = Iteration(x0, zeros(n), rx, cx, l, 1.0, zeros(l), zeros(l), 0, 0, 0, 0, zeros(n), zeros(n), 0., 0., 0., false, true, false, false, 0, 1)

    # Initialization of the working set
    working_set = init_working_set(cx, K, first_iter, q, l)

    first_iter.t = working_set.t

    # Compute jacobians at current point
    new_point!(x0, r, rx, c, cx, J, A, n, m, l)

    active_C = Constraint(cx[working_set.active[1:working_set.t]], A[working_set.active[1:working_set.t],:], scaling, zeros(working_set.t))

    # Gradient of the objective function
    ∇fx = transpose(J) * rx

    p_gn = zeros(n)

    # Estimation of the Lagrange multipliers
    # Computation of the Gauss-Newton search direction
    evaluate_scaling!(active_C)
    P1, L11, Q1, F_L, F_J2 = update_working_set(working_set, rx, A, active_C, ∇fx, J, p_gn, first_iter)
    rx_sum = dot(rx, rx)
    active_cx_sum = dot(cx[working_set.active[1:working_set.t]],cx[working_set.active[1:working_set.t]])
    first_iter.t = working_set.t
    previous_iter = copy(first_iter)
    R11, Q2, P2 = F_L.R, F_L.Q*Matrix(I,working_set.t,working_set.t), F_L.P
    J2 = (J * Q1)[:,first_iter.rankA + 1:end]
    Q3, P3, R22 = F_J2.Q*Matrix(I,m,m), F_J2.P, F_J2.R
    nrm_b1 = norm(first_iter.b_gn[1:first_iter.dimA])
    nrm_d1 = norm(first_iter.d_gn[1:first_iter.dimJ2])
    nrm_d = norm(first_iter.d_gn)

    # Analys of the lastly computed search direction
    search_direction_analys!(previous_iter,first_iter,nb_iteration,x0,c,r,rx,cx,active_C.cx,first_iter.λ,rx_sum,active_cx_sum,p_gn,first_iter.d_gn,first_iter.b_gn,nrm_b1,nrm_d1,nrm_d,J,m,n,working_set,first_iter.rankA,first_iter.rankJ2,P1,Q1,L11,P2,Q2,R11,P3,Q3,R22,first_iter.add,first_iter.del,active_C.scaling,active_C.diag_scale)

    # Computation of penalty constants and steplentgh
    α, w = compute_steplength(first_iter,x0, r, rx, J, first_iter.p, c, cx, A, active_C, previous_iter.w, working_set, K, first_iter.dimA, m, first_iter.index_del, previous_iter.α, previous_iter.rankJ2, first_iter.rankJ2, first_iter.code, weight_code)
    first_iter.α = α
    first_iter.w = w
    x = x0 + α * first_iter.p

    # Evaluate residuals, constraints and compute jacobians at new point

    r.ctrl = 1
    c.ctrl = 1
    r(x, rx, J)
    rx_sum = dot(rx, rx)
    c(x, cx, A)
    new_point!(x, r, rx, c, cx, J, A, n, m, l)
    ∇fx = transpose(J) * rx

    # Check for termination criterias at new point
    exit_code = check_termination_criteria(first_iter, previous_iter, working_set, active_C, x, cx, rx_sum, ∇fx, MAX_ITER, nb_iteration, ε_abs, ε_rel, ε_x, ε_c)

    # Print collected informations about the first iteration
    output!(io,first_iter, working_set, nb_iteration, 0.0, active_cx_sum)

    # Check for violated constraints and add it to the working set
    first_iter.add = evaluate_violated_constraints(cx, working_set)
    active_C.cx = cx[working_set.active[1:working_set.t]]
    active_C.A = A[working_set.active[1:working_set.t],:]

    previous_iter = copy(first_iter)
    first_iter.x = x
    first_iter.rx = rx
    first_iter.cx = cx
    nb_iteration += 1
    iter = copy(first_iter)
    iter.first = false
    iter.add = false
    iter.del = false

    # Main loop for next iterations

    while exit_code == 0
         p_gn = zeros(n)
        # Estimation of the Lagrange multipliers
        # Computation of the Gauss-Newton search direction
        evaluate_scaling!(active_C)
        P1, L11, Q1, F_L, F_J2 = update_working_set(working_set, rx, A, active_C, ∇fx, J, p_gn, iter)
        active_cx_sum = dot(cx[working_set.active[1:working_set.t]],cx[working_set.active[1:working_set.t]])
        iter.t = working_set.t
        R11, Q2, P2 = F_L.R, F_L.Q*Matrix(I,working_set.t,working_set.t), F_L.P
        J2 = (J * Q1)[:, iter.rankA+1:end]
        Q3, P3, R22 = F_J2.Q*Matrix(I,m,m), F_J2.P, F_J2.R
        nrm_b1 = norm(iter.b_gn[1:iter.dimA])
        nrm_d1 = norm(iter.d_gn[1:iter.dimJ2])
        nrm_d = norm(iter.d_gn)

        # Analys of the lastly computed search direction
        search_direction_analys!(previous_iter, iter, nb_iteration, x, c, r, rx, cx, active_C.cx, iter.λ, rx_sum, active_cx_sum, p_gn, iter.d_gn, iter.b_gn, nrm_b1, nrm_d1, nrm_d, J, m, n, working_set, iter.rankA, iter.rankJ2, P1, Q1, L11, P2, Q2, R11, P3, Q3, R22, iter.add, iter.del,active_C.scaling,active_C.diag_scale)

        # Computation of penalty constants and steplentgh
        α, w = compute_steplength(iter,x,r,rx,J,iter.p,c,cx,A,active_C,previous_iter.w,working_set,K,iter.dimA,m,iter.index_del,previous_iter.α,previous_iter.rankJ2,iter.rankJ2,iter.code,weight_code)
        iter.α = α
        iter.w = w
        x = x + α * iter.p

        # Evaluate residuals, constraints and compute jacobians at new point
        r.ctrl = 1
        c.ctrl = 1
        r(x, rx, J)
        rx_sum = dot(rx, rx)
        c(x, cx, A)
        new_point!(x, r, rx, c, cx, J, A, n, m, l)
        ∇fx = transpose(J) * rx

        # Check for termination criterias at new point
        exit_code = check_termination_criteria(iter, previous_iter, working_set, active_C, iter.x, cx, rx_sum, ∇fx, MAX_ITER, nb_iteration, ε_abs, ε_rel, ε_x, ε_c)

        # Print collected informations about current iteration
        exit_code == 0 ? output!(io,iter, working_set, nb_iteration, previous_iter.β, active_cx_sum) : final_output!(io,previous_iter, working_set, exit_code, nb_iteration)

        # Check for violated constraints and add it to the working set
        iter.add = evaluate_violated_constraints(cx, working_set)
        active_C.cx = cx[working_set.active[1:working_set.t]]
        active_C.A = A[working_set.active[1:working_set.t],:]


        nb_iteration += 1
        previous_iter = copy(iter)
        iter.x = x
        iter.rx = rx
        iter.cx = cx
        iter.del = false
        iter.add = false
    end
    # Close the IO Stream and print some collected informations
    close(io)
    for s in readlines(output_file)
        println(s)
    end 
    rm(output_file)
    return ENLSIP(iter.x, dot(rx, rx))
end
