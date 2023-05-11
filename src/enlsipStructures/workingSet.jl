"""
    WorkingSet

In ENLSIP, the working-set is a prediction of the set of active constraints at the solution

It is updated at every iteration thanks to a Lagrangian multipliers estimation

Fields of this structure summarize infos about the qualification of the constraints, i.e. :

* `q` : number of equality constraints

* `t` : number of constraints in current working set (all equalities and some inequalities considered to be active at the solution)

* `l` : total number of constraints (i.e. equalities and inequalities)

* active :

    - `Vector` of size `l`

    - first `t` elements are indeces of constraints in working set sorted in increasing order, other elements equal `0`

* inactive : 

    - `Vector` of size `l-q`

    - first `l-t` elements are indeces of constraints not in working set sorted in increasing order, other elements equal `0`

"""
mutable struct WorkingSet
    q::Int64
    t::Int64
    l::Int64
    active::Vector{Int64}
    inactive::Vector{Int64}
end

# Equivalent Fortran : DELETE in dblreduns.f
# Moves the active constraint number s to the inactive set

function delete_constraint!(W::WorkingSet, s::Int64)

    l, t = W.l, W.t

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
# Add the inactive constraint nulber s to the active s

function add_constraint!(W::WorkingSet, s::Int64)

    l, t = W.l, W.t
    # s-th inactive constraint moved from inactive to active set
    W.active[t+1] = W.inactive[s]
    sort!(@view W.active[1:t+1])
    # Inactive set reorganized
    for i = s:l-t-1
        W.inactive[i] = W.inactive[i+1]
    end
    W.inactive[l-t] = 0
    W.t += 1
    return
end