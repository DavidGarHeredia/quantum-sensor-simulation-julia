using LinearAlgebra

include("hamiltonians.jl")

function initialize_global_state(
    nqubits::Int,
    γ_s::Float64,
    B::Float64,
    T::Float64,
    entanglement::Bool
)
    ket_0_c = zeros(2^nqubits); ket_0_c[1] = 1.0
    ket_1_c = zeros(2^nqubits); ket_1_c[end] = 1.0
    # System density matrix (thermal state: ρ_s = exp{-β*H_s}/Tr(exp{-β*H_s}))
    k_B = 1.380649e-23 # J K^-1
    β = 1/(T*k_B)
    S_a = extended_pauli_operators(dim_left=2^0, dim_right=2^1)
    S_b = extended_pauli_operators(dim_left=2^1, dim_right=2^0)
    H_s = build_system_hamiltonian(S_a, S_b, γ_s, B)
    ρ_s = exp(-β*H_s)
    ρ_s = ρ_s/tr(ρ_s)
    # Camera density matrix (ρ_c = ket{GHZ}bra{GHZ} or ket{0}bra{0})
    c_state = entanglement ? (ket_0_c + ket_1_c)/sqrt(2) : ket_0_c
    ρ_c = c_state*c_state'
    # Global density matrix
    ρ = kron(ρ_s, ρ_c)
    return ρ
end
