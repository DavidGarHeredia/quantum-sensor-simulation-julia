using LinearAlgebra
using TensorCast
using ProgressMeter
using Distributions
using Base.Iterators
using Base.Threads

include("initialize_state.jl")
include("hamiltonians.jl")
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
    c_basis = create_basis_of_states_of_the_camera(nqubits)
    projectors = [diagm(row) for row in eachrow((1*I)(2^nqubits))]
    D_vals, θ_vals, d_s_vals = generate_params_combinations_for_the_simulation(D, θ, d_s, nconfigs)

    S, S_a, S_b = build_extended_spin_operators(nqubits)
    H_c = build_camera_hamiltonian(S, γ_c, B)
    H_s = build_system_hamiltonian(S_a, S_b, γ_s, B)
    ρ_0 = initialize_global_state(nqubits, γ_s, B, T, entanglement)

    p = Progress(nconfigs) # Progress bar
    measures = Vector{Vector{Vector{Int64}}}(undef, nconfigs) #, nmeasures, nqubits)
    for i in 1:nconfigs
        H = build_hamiltonian(H_c, H_s, S, S_a, S_b, d_c, d_s_vals[i], D_vals[i], θ_vals[i], γ_s, γ_c)
        ρ_t = time_evolution(ρ_0, H, t)
        probabilities = get_probabilities(ρ_t, projectors)
        measures[i] = measure(c_basis, probabilities, nmeasures) # @inbounds
        next!(p)  # Update progress bar
    end

    return measures
end


function time_evolution(ρ_i::Matrix{ComplexF64}, H::Matrix{ComplexF64}, t::Float64)
    ħ = 1.0545718176461565e-34 # J s
    U = exp(-1im*H*t/ħ)
    ρ_f = U*ρ_i*U'
    return ρ_f
end

function build_extended_spin_operators(nqubits::Int)
    S = create_big_spin_operator(nqubits)
    S_a = extended_pauli_operators(dim_left=2^0, dim_right=2^(nqubits+1))
    S_b = extended_pauli_operators(dim_left=2^1, dim_right=2^nqubits)
    return S, S_a, S_b
end

function generate_params_combinations_for_the_simulation(
    D::Dict{Symbol, Any},
    θ::Dict{Symbol, Any},
    d_s::Dict{Symbol, Any},
    nconfigs::Int,
)
    D_vals = rand(Uniform(D[:low], D[:high]), nconfigs)
    θ_vals = rand(Uniform(θ[:low], θ[:high]), nconfigs)
    d_s_vals = rand(Uniform(d_s[:low], d_s[:high]), nconfigs)
    return D_vals, θ_vals, d_s_vals
end

function create_basis_of_states_of_the_camera(nqubits::Int)::Vector{Vector{Int64}}
    spin_levels = [0, 1]  # 0 = +, 1 = -
    qubits_states = repeat([spin_levels], nqubits)
    c_basis = [Vector{Int}()]
    for state in qubits_states
        c_basis = [vcat(x, y) for x in c_basis for y in state]
    end
    return c_basis
end
