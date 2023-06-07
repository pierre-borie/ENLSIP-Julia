#= Functions used to update penalty weights in the merit function and the steplength=# 

using Polynomials

"""
    psi(x,α,p,r,c,w,m,l,t,active,inactive)

Compute and return the evaluation of the merit function at ``(x+\\alpha p,w)`` with current working set ``\\mathcal{W}`` and the set of inactive constraints ``\\mathcal{I}``

``\\psi(x,w) = \\dfrac{1}{2}\\|r(x)\\|^2 +  \\dfrac{1}{2}\\sum_{i \\in \\mathcal{W}} w_ic_i(x)^2 + \\dfrac{1}{2} \\sum_{j \\in \\mathcal{I}} w_j\\min(0,c_j(x))^2``
"""
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
    for i = 1:l-t
        j = inactive[i]
        if c_new[j] < 0.0
            penalty_constraint_sum += w[j] * c_new[j]^2
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
            for j = 4:-1:ii+1
                K[j][k] = K[j-1][k]
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
        if y_norm != 0.0
            y /= y_norm
        end
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
                        pos_index[j] = pos_index[j+1]
                        y[j] = y[j+1]
                    end
                end
            end
            y_sum *= y_norm * y_norm
            terminated = (n_runch <= 0) || (ctrl == 2) || (i_stop == n_runch)
        end
    end
    return
end


# EUCNRM
# Update the penalty constants using the euclidean norm

function euclidean_norm_weight_update(
    vA::Vector{Float64},
    cx::Vector{Float64},
    active::Vector{Int64},
    t::Int64,
    μ::Float64,
    dimA::Int64,
    previous_w::Vector{Float64},
    K::Array{Array{Float64,1},1})

    # if no active constraints, previous penalty weights are used
    w = previous_w[:]
    if t != 0

        # Compute z = [<∇c_i(x),p>^2]_i for i ∈ active
        z = vA .^ 2
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
    return w
end


# MAXNRM
# Update the penalty weights corresponding to the
# constraints in the current working setb

function max_norm_weight_update!(
    nrm_Ap::Float64,
    rmy::Float64,
    α_w::Float64,
    δ::Float64,
    w::Vector{Float64},
    active::Vector{Int64},
    t::Int64,
    K::Array{Array{Float64,1},1})
    μ = (abs(α_w - 1.0) <= δ ? 0.0 : rmy / nrm_Ap)
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
    work_set::WorkingSet,
    dimA::Int64,
    norm_code::Int64)

    # Data
    δ = 0.25
    active = work_set.active
    t = work_set.t


    nrm_Ap = sqrt(dot(Ap, Ap))
    nrm_cx = (isempty(cx[active[1:dimA]]) ? 0.0 : max(0,maximum(map(abs,cx[active[1:dimA]]))))
    nrm_Jp = sqrt(dot(Jp, Jp))
    nrm_rx = sqrt(dot(rx,rx))

    # Scaling of vectors Jp, Ap, rx and cx
    if nrm_Jp != 0
        Jp = Jp / nrm_Jp
    end

    if nrm_Ap != 0
        Ap = Ap / nrm_Ap
    end

    if nrm_rx != 0
        rx = rx / nrm_rx
    end

    if nrm_cx != 0
        cx = cx / nrm_cx
    end
    

    Jp_rx = dot(Jp, rx) * nrm_Jp * nrm_rx
    

    AtwA = 0.0
    BtwA = 0.0
    if dimA > 0
        for i = 1:dimA
            k = active[i]
            AtwA += w_old[k] * Ap[i]^2
            BtwA += w_old[k] * Ap[i] * cx[k]
        end
    end
    AtwA *= nrm_Ap^2
    BtwA *= nrm_Ap * nrm_cx

    α_w = 1.0
    if abs(AtwA + nrm_Jp^2) > eps(Float64)
        α_w = (-BtwA - Jp_rx) / (AtwA + nrm_Jp^2)
    end

    rmy = (abs(Jp_rx + nrm_Jp^2) / δ) - nrm_Jp^2

    if norm_code == 0
        w = w_old[:]
        max_norm_weight_update!(nrm_Ap, rmy, α_w, δ, w, active, t, K)
    elseif norm_code == 2

        w = euclidean_norm_weight_update(Ap*nrm_Ap, cx*nrm_cx, active, t, rmy, dimA, w_old, K)
    end
    #                               T                       T
    # Computation of ψ'(0) = [J(x)p] r(x)+   Σ      w_i*[∇c_i(x) p]c_i(x)
    #                                     i ∈ active
    BtwA = 0.0
    AtwA = 0.0
    wsum = 0.0
    for i = 1:t
        k = active[i]
        AtwA += w[k] * Ap[i]^2
        BtwA += w[k] * Ap[i] * cx[k]
        wsum += w[k]
    end
    BtwA *= nrm_Ap * nrm_cx
    AtwA *= nrm_Ap^2

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
            v[m+k] = sqrt(w[k]) * cx[k]
        end
    end
    if l != 0
        for j = 1:l-t
            k = inactive[j]
            v[m+k] = (cx[k] > 0 ? 0.0 : sqrt(w[k]) * cx[k])
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
        α, pα = 0.0, 0.0

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
    η, d = 0.1, 1.0
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
    ε, error = 1e-4, 1.0
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
    return u, exit
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
    work_set::WorkingSet,
    ψ0::Float64,
    dψ0::Float64,
    α_low::Float64,
    α_upp::Float64)


    # Data
    m = length(rx)
    l, t = work_set.l, work_set.t
    active, inactive = work_set.active, work_set.inactive

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
    gac_error = false

    # LINC2
    # Computation of v1
    v1 = JpAp
    if t != 0
        for i = 1:t
            k = active[i]
            v1[m+k] = sqrt(w[k]) * v1[m+k]
        end
    end
    if l - t != 0
        for j = 1:l-t
            k = inactive[j]
            v1[m+k] = (cx[k] > 0 ? 0.0 : sqrt(w[k]) * v1[m+k])
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
            α_km1, gac_error = goldstein_armijo_step(ψ0, dψ0, α_min, τ, p_max, x, α_k, p, r, c, w, m, l, t, active, inactive)
        end
    end
    α = α_km1
    return α, gac_error
end

# UPBND
# Determine the upper bound of the steplength

function upper_bound_steplength(
    A::Matrix{Float64},
    cx::Vector{Float64},
    p::Vector{Float64},
    work_set::WorkingSet,
    index_del::Int64)

    # Data
    inactive = work_set.inactive
    t = work_set.t
    l = work_set.l

    α_upper = Inf
    index_α_upp = 0
    if norm(inactive, Inf) > 0
        for i = 1:l-t
            j = inactive[i]
            if j != index_del
                ∇cjTp = dot(A[j, :], p)
                α_j = -cx[j] / ∇cjTp
                if cx[j] > 0 && ∇cjTp < 0 && α_j < α_upper
                    α_upper = α_j
                    index_α_upp = j
                end
            end
        end
    end
    α_upper = min(3.0, α_upper)
    return α_upper, index_α_upp
end


"""
    compute_steplength

Equivalent Fortran77 routine : STPLNG

Update the penalty weights and compute the steplength using the merit function [`psi`](@ref)

If search direction computed with method of Newton, an undamped step is taken, i.e. ``\\alpha =1``

# On return

* `α` : the computed steplength

* `w` : vector of size `l`, containts the computed penalty constants 
"""
function compute_steplength(
    iter::Iteration,
    previous_iter::Iteration,
    x::Vector{Float64},
    r::ResidualsEval,
    rx::Vector{Float64},
    J::Matrix{Float64},
    c::ConstraintsEval,
    cx::Vector{Float64},
    A::Matrix{Float64},
    active_constraint::Constraint,
    work_set::WorkingSet,
    K::Array{Array{Float64,1},1},
    weight_code::Int64)

    # Data
    c1 = 1e-3

    (m,_) = size(J)
    p = iter.p
    dimA = iter.dimA
    rankJ2 = iter.rankJ2
    method_code = iter.code
    ind_constraint_del = iter.index_del
    
    previous_α = previous_iter.α
    prev_rankJ2  = previous_iter.rankJ2
    w_old = previous_iter.w
    
    Jp = J * p
    Ap = A * p
    JpAp = vcat(Jp, Ap)
    active_Ap = (active_constraint.A) * p
    active_index = work_set.active[1:work_set.t]
    if active_constraint.scaling
        active_Ap = active_Ap ./ active_constraint.diag_scale
    end
    

    Ψ_error = 0
    if method_code != 2
        # Compute penalty weights and derivative of ψ at α = 0
        w, dψ0 = penalty_weight_update(w_old, Jp, active_Ap, K, rx, cx, work_set, dimA, weight_code)

        #
        # Compute ψ(0) = 0.5 * [||r(x)||^2 +    Σ     (w_i*c_i(x)^2)]
        #                                   i ∈ active
        ψ0 = 0.5 * (dot(rx, rx) + dot(w[active_index], cx[active_index] .^ 2))
        # check is p is a descent direction
        if dψ0 >= 0
            α = 1.0
            Ψ_error = -1
            iter.index_α_upp = 0
        else

            # Determine upper bound of the steplength
            α_upp, index_α_upp = upper_bound_steplength(A, cx, p, work_set, ind_constraint_del)
            α_low = α_upp / 3000.0

            # Determine a first guess of the steplength
            magfy = (rankJ2 < prev_rankJ2 ? 6.0 : 3.0)
            α0 = min(1.0, magfy * previous_α, α_upp)
            # Compute the steplength
            
            α, gac_error = linesearch_constrained(x, α0, p, r, c, rx, cx, JpAp, w, work_set, ψ0, dψ0, α_low, α_upp)
            if gac_error 
                ψ_k = psi(x,α,p,r,c,w,m,work_set.l,work_set.t,work_set.active,work_set.inactive)
                Ψ_error = check_derivatives(dψ0,ψ0,ψ_k,x,α,p,r,c,w,work_set,m)
            end

            # Compute the predicted linear progress and actual progress
            uppbound = min(1.0, α_upp)
            atwa = dot(w[active_index], active_Ap .^ 2)
            iter.predicted_reduction = uppbound * (-2.0 * dot(Jp, rx) - uppbound * dot(Jp, Jp) + (2.0 - uppbound^2) * atwa)

            # Computation of new point and actual progress
            # Evaluate residuals and constraints at the new point
            r.ctrl = 1
            c.ctrl = 1
            rx_new = zeros(m)
            cx_new = zeros(work_set.l)
            r(x + α * p, rx_new, J)
            c(x + α * p, cx_new, A)
            whsum = dot(w[active_index], cx_new[active_index] .^ 2)
            iter.progress = 2 * ψ0 - dot(rx_new, rx_new) - whsum
            iter.index_α_upp = (index_α_upp != 0 && abs(α - α_upp) > 0.1 ? 0 : index_α_upp)
        end

    else
        # Take an undamped step

        w = w_old
        α_upp = 3.0
        index_α_upp = 0
        α = 1.0
    end
    
    return α, w, Ψ_error
end

function check_derivatives(
    dψ0::Float64,
    ψ0::Float64,
    ψ_k::Float64,
    x_old::Vector{Float64},
    α::Float64,
    p::Vector{Float64},
    r::ResidualsEval,
    c::ConstraintsEval,
    w::Vector{Float64},
    work_set::WorkingSet,
    m::Int64)

    # Data
    l,t = work_set.l, work_set.t
    active, inactive = work_set.active,work_set.inactive

    ctrl = -1
    ψ_mα = psi(x_old,-α,p,r,c,w,m,l,t,active,inactive)
    dψ_forward = (ψ_k - ψ0) / α
    dψ_backward = (ψ0 - ψ_mα) / α
    dψ_central = (ψ_k - ψ_mα) / (2*α)
    max_diff = maximum(map(abs,[dψ_forward-dψ_central , dψ_forward - dψ_backward, dψ_backward - dψ_central]))
    inconsistency = abs(dψ_forward-dψ0) > max_diff && abs(dψ_central-dψ0) > max_diff
    exit = (inconsistency ? -1 : 0)

    return exit

end