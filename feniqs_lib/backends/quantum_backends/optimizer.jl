#!/usr/bin/env julia

"""
Quantum Circuit Optimizer

Developed for the MIMIQ matrix-product-states simulator.

Author: Shannon Whitlock, Copyright © 2024 University of Strasbourg.
"""

using MimiqCircuitsBase, AbstractQCSs
using Random, LinearAlgebra
using Base.Threads

################################
##  Qubit ordering algorithm  ##
################################
"""
Reorder qubits in a circuit according to the permutation perm
"""
function reorder_qubits(circuit, perm)
    circ =Circuit()
    for inst in circuit
        qtargets = Tuple(perm[t] for t in inst.qtargets)
        push!(circ, Instruction(inst.op, qtargets, inst.ctargets, inst.ztargets))
    end
    return circ
end

"""
Calculate cumulative distance between all pairs of qubits in vector 'pairs'
"""
function calculate_distance(perm::Vector{Int}, pairs::Vector{Tuple{Int,Int,Int}})
    dist = 0
    @inbounds for (i,j,w) in pairs
        dist += abs(perm[i] - perm[j])*w
    end
    return dist    
end

"""
    optimize_ordering(circ)

Reorder qubits in a quantum circuit to minimize distances between two-qubit gates.

Args:
    circ: The quantum circuit to reorder.
    initial_temp: Initial temperature for simulated annealing (default: 1000).
    final_temp: final temperature (default 1)
    max_iter: Maximum number of iterations for simulated annealing (default: (n/2)^3*log(n))
    ntrials: Number of trials to perform using parallelization (default: nthreads/2)

Returns:
    Tuple of (reordered circuit, best_permutation, final distance).
"""
function optimize_ordering(circ; initial_temp=1000, final_temp=1, iterations=nothing, max_iter=1e10, ntrials=nthreads()÷2, rng=Random.GLOBAL_RNG,  kwargs...)
    n = numqubits(circ)

    # empirical estimate for the number of iterations required for simulated_annealing algorithm
    # underestimates a bit the required iterations for large n, but should be ok up to ~1024 qubits
    if isnothing(iterations)
        iterations = min(round(Int,(n/2)^3), max_iter)
    end

    # Precompute a symmetric frequency matrix of all qubit pairs
    pair_matrix = zeros(Int, n, n)
    for inst in circ
        if inst isa Instruction{2}
            q1, q2 = inst.qtargets
            pair_matrix[q1, q2] += 1
            pair_matrix[q2, q1] += 1
        end
    end

    # Convert to a sparse list of weighted pairs
    pairs = Tuple{Int, Int, Int}[(i, j, pair_matrix[i, j]) for i in 1:n for j in i+1:n if pair_matrix[i, j] > 0]
    initial_dist = calculate_distance(collect(1:n), pairs)

    # Run simulated annealing in parallel using available threads
    results = Vector{Tuple{Vector{Int}, Float64}}(undef, ntrials)
    @threads for i in 1:ntrials
        local_rng = Random.MersenneTwister(rand(rng, UInt32))
        perm, final_dist = simulated_annealing(n, pairs, initial_temp, final_temp, iterations; rng=local_rng)
        results[i] = (perm, final_dist)
    end

    # Select the best result
    best_perm, best_dist = argmin(r -> r[2], results)

    if best_dist < initial_dist
        return reorder_qubits(circ, best_perm), best_perm, best_dist
    else
        return circ, collect(1:n), initial_dist
    end
end



"""
    simulated_annealing(n, pairs, initial_temp, final_temp, max_iter; rng=Random.GLOBAL_RNG)

Perform simulated annealing optimization on an integer permutation problem.

# Arguments
- `n::Int`: Number of elements in the permutation.
- `pairs::Vector{Tuple{Int,Int,Int}}`: Vector of (i, j, distance) tuples.
- `initial_temp::Float64`: Initial temperature for the annealing process.
- `final_temp::Float64`: Final temperature for the annealing process.
- `iterations::Int`: Number of iterations.
- `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).

# Returns
- `Tuple{Vector{Int}, Int}`: Best permutation found and its corresponding distance.
"""
function simulated_annealing(n::Int, pairs::Vector{Tuple{Int,Int,Int}}, initial_temp, final_temp, iterations; rng=Random.GLOBAL_RNG)
    perm = collect(1:n)
    best_perm = copy(perm)
    current_distance = calculate_distance(perm, pairs)
    best_distance = current_distance
    
    for s in 1:iterations

        frac = s/iterations
        temperature = initial_temp * (final_temp / initial_temp)^sqrt(frac)

        i = rand(rng, 1:n-1)
        j = rand(rng, i+1:n)
        perm[i], perm[j] = perm[j], perm[i]
        
        new_distance = calculate_distance(perm, pairs)
        
        if new_distance < current_distance || rand(rng) < 1 + (current_distance - new_distance) / temperature
            current_distance = new_distance
            if new_distance < best_distance
                best_distance = new_distance
                best_perm .= perm
            end
        else
            perm[i], perm[j] = perm[j], perm[i]
        end
    end

    return best_perm, best_distance
end


################################
##  Gate fusion algorithm     ##
################################

global id = complex([1.0 0.0; 0.0 1.0])

function fuse(i1::Instruction{2,0,0,GateCX}, i2::Instruction{2,0,0,GateCX})
    q1, q2 = getqubits(i1), getqubits(i2)
    if q1==q2
        return nothing,nothing # identity
    elseif q2 == reverse(q1)
        return Instruction(GateCustom([1 0 0 0; 0 0 1 0; 0 0 0 1; 0 1 0 0]), q1...), nothing
    end
    return i1,i2
end

function fuse(i1::Instruction{2,0,0,T}, i2::Instruction{2,0,0,GateCX}) where T<:GateCustom
    q1, q2 = getqubits(i1), getqubits(i2)
    m = matrix(i1.op)
    if q1==q2
        m[3,:], m[4,:] = m[4,:], m[3,:] # Swap rows 3 and 4
        return i1, nothing
    elseif q2 == reverse(q1)
        m[2,:], m[4,:] = m[4,:], m[2,:] # Swap rows 2 and 4
        return i1, nothing
    end
    return i1,i2
end

""" fuse arbitrary one-qubit gate into GateCustom without tensor products """
function fuse(i1::Instruction{2,0,0,T}, i2::Instruction{1,0,0,T2}) where {T<:GateCustom, T2<:AbstractGate{1}} 
    q1, q2 = getqubits(i1), getqubits(i2)
    m = matrix(i1.op)
    u = matrix(i2.op)

    if q2[1] == q1[1]
        @inbounds for j in 1:4
            m11, m21, m31, m41 = m[1,j], m[2,j], m[3,j], m[4,j]
            m[1,j] = u[1,1] * m11 + u[1,2] * m31
            m[2,j] = u[1,1] * m21 + u[1,2] * m41
            m[3,j] = u[2,1] * m11 + u[2,2] * m31
            m[4,j] = u[2,1] * m21 + u[2,2] * m41
        end
        return i1, nothing
    elseif q2[1] == q1[2]
        @inbounds for j in 1:4
            m11, m21, m31, m41 = m[1,j], m[2,j], m[3,j], m[4,j]
            m[1,j] = u[1,1] * m11 + u[1,2] * m21
            m[2,j] = u[2,1] * m11 + u[2,2] * m21
            m[3,j] = u[1,1] * m31 + u[1,2] * m41
            m[4,j] = u[2,1] * m31 + u[2,2] * m41
        end
        return i1, nothing
    end
    return i1,i2
end

""" fuse arbitrary N-qubit gates -> GateCustom"""
function fuse(inst1::Instruction{N,0,0,T}, inst2::Instruction{N,0,0,T2}) where {N,T,T2}
    q1, q2 = getqubits(inst1), getqubits(inst2)
    m1, m2 = matrix(inst1.op), matrix(inst2.op)

    if q1 == q2
        return Instruction(GateCustom(m2*m1), q1...), nothing
    elseif N == 2 && q2 == reverse(q1)
        u = copy(m2)
        u[2,:], u[3,:] = u[3,:], u[2,:]
        u[:,2], u[:,3] = u[:,3], u[:,2]
        return Instruction(GateCustom(u * m1), q1...), nothing
    end
    return inst1,inst2 # if gates cannot be fused
end 

""" fuse arbitrary one- and two-qubit gates -> GateCustom"""
function fuse(inst1::Instruction{1,0,0,T}, inst2::Instruction{2,0,0,T2}) where {T,T2}
    q1, q2 = getqubits(inst1), getqubits(inst2)
    m1, m2 = matrix(inst1.op), matrix(inst2.op)

    if q2[1] == q1[1]
        return Instruction(GateCustom(m2 * kron(m1, id)), q2...), nothing
    elseif q2[2] == q1[1]
        return Instruction(GateCustom(m2 * kron(id, m1)), q2...), nothing
    end
    return inst1,inst2
end 

function fuse(inst1::Instruction{2,0,0,T}, inst2::Instruction{1,0,0,T2}) where {T,T2}
    q1, q2 = getqubits(inst1), getqubits(inst2)
    m1, m2 = matrix(inst1.op), matrix(inst2.op)

    if q2[1] == q1[1]
        return Instruction(GateCustom(kron(m2, id) * m1), q1...), nothing
    elseif q2[1] == q1[2]
        return Instruction(GateCustom(kron(id, m2) * m1), q1...), nothing
    end
    return inst1,inst2
end 

fuse(inst1, inst2) = inst1,inst2


"""
    compress(circ::Circuit) -> Circuit

Compresses a quantum circuit by fusing adjacent operations that act on the same qubits.

If a fuse is successful, the operations are combined into one. 
The fused or unfused operations are then added to a new, compressed circuit.
This compression algorithm does not commute gates past one another.

# Arguments:
- `circ::Circuit`: A quantum circuit containing quantum instructions.

# Returns:
- `ccirc::Circuit`: A new compressed circuit with fused operations.

# Example:
```julia
compressed_circuit = compress(my_circuit)
"""
function compress(circ::Circuit)
    ccirc = Circuit()
    i = 1
    circ_len = length(circ)

    while i <= circ_len
        current_op = circ[i]
        j = i + 1

        # Try fusing operations until a fuse is not possible
        while j <= circ_len
            current_op, next_op = fuse(current_op, circ[j])

            if next_op !== nothing  # Fusion failed, stop trying
                break
            end
            j += 1  # Move to next operation
        end

        # Add the fused operation to the new circuit if it wasn't removed
        if current_op != nothing
            push!(ccirc, current_op)
        end

        # Move i to j to process the next segment
        i = j
    end

    return ccirc
end
