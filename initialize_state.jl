using LinearAlgebra
# using PhysicalConstants

include("hamiltonians.jl")


function initialize_state(
    nqubits::Int,
    γ_s::Float64,
    B::Float64,
    T::Float64,
    entanglement::Bool
)
    c_basis = I(2^nqubits)
    ket_0_c, ket_1_c = c_basis[1, :], c_basis[end, :]
    # System density matrix (thermal state: ρ_s = exp{-β*H_s}/Tr(exp{-β*H_s}))
    β = 1/(T*k_B)
    S_a = extend_operators(ħ/2*σ; dim_left=2^0, dim_right=2^1)
    S_b = extend_operators(ħ/2*σ; dim_left=2^1, dim_right=2^0)
    H_s = get_H_s(S_a, S_b, γ_s, B)
    ρ_s = exp(-β*H_s)
    ρ_s = ρ_s/tr(ρ_s)
    # Camera density matrix (ρ_c = ket{GHZ}bra{GHZ} or ket{0}bra{0})
    c_state = entanglement ? (ket_0_c + ket_1_c)/sqrt(2) : ket_0_c
    ρ_c = c_state*c_state'
    # Global density matrix
    ρ = kron(ρ_s, ρ_c)

    return ρ
end
