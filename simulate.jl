using LinearAlgebra
using TensorCast
using ProgressMeter
using Distributions
using Base.Iterators
using Base.Threads

include("initialize_state.jl")
include("hamiltonians.jl")
include("time_evolution.jl")
include("measure.jl")


function simulate(;
    D::Dict{Symbol, Any},
    θ::Dict{Symbol, Any},
    d_s::Dict{Symbol, Any},
    d_c::Float64,
    γ_s::Float64,
    γ_c::Float64,
    B::Float64,
    t::Float64,
    nconfigs::Int,
    nmeasures::Int,
    nqubits::Int,
    T::Float64,
    entanglement::Bool
)
    spin_levels = [0, 1]  # 0 = +, 1 = -
    # Basis of states of the camera
    qubits_states = repeat([spin_levels], nqubits)
    c_basis = map(sort, map(collect, product(qubits_states...)))
    c_basis = reshape(c_basis, length(c_basis))
    projectors = [diagm(row) for row in eachrow((1*I)(2^nqubits))]
    # Initialize the global state
    ρ_0 = initialize_state(nqubits, γ_s, B, T, entanglement)
    # Generate combinations of D, theta and d_s and simulate
    measures = Vector{Vector{Vector{Int64}}}(undef, nconfigs) #, nmeasures, nqubits)
    D_vals = rand(Uniform(D[:low], D[:high]), nconfigs)
    θ_vals = rand(Uniform(θ[:low], θ[:high]), nconfigs)
    d_s_vals = rand(Uniform(d_s[:low], d_s[:high]), nconfigs)
    p = Progress(nconfigs) # Progress bar
    @threads for i in 1:nconfigs
        # Build the hamiltonian
        H = get_H(nqubits, d_c, d_s_vals[i], D_vals[i], θ_vals[i], γ_s, γ_c, B)
        # Time evolution
        U = get_U(H, t)
        ρ_t = time_evolution(ρ_0, U)
        # Measure
        probabilities = get_probabilities(ρ_t, projectors)
        measures[i] = measure(c_basis, probabilities, nmeasures) # @inbounds 
        # Update progress bar
        next!(p)
    end

    return measures
end
