using LinearAlgebra
using TensorCast

function get_qubits_coordinates(nqubits::Int, d_c::Float64)
    positions = zeros(nqubits, 2)
    L = (nqubits - 1)*d_c
    positions[:, 1] = LinRange(-L/2, L/2, nqubits)
    return positions
end


function get_particles_coordinates(D::Float64, d_s::Float64, θ::Float64)
    x_a = 0
    y_a = D
    x_b = x_a + d_s*cos(θ)
    y_b = y_a + d_s*sin(θ)
    position_a = [x_a y_a]
    position_b = [x_b y_b]

    return position_a, position_b
end


function cartesian2polar(coordinates::Matrix{Float64})
    r = vec(sqrt.(sum(coordinates.^2; dims=2)))
    θ = atan.(coordinates[:, 2], coordinates[:, 1])
    return r, θ
end


function extended_pauli_operators(; dim_left::Int, dim_right::Int)::Vector{Matrix{ComplexF64}}
    ħ = 1.0545718176461565e-34 # J s
    σ_x = [0 1; 1 0]
    σ_y = [0 -1im; 1im 0]
    σ_z = [1 0; 0 -1]
    A = ħ/2*[σ_x, σ_y, σ_z]
    A_extended = [kron(kron(I(dim_left), A_i), I(dim_right)) for  A_i in A]
    return A_extended
end


function get_H_c(
    S::Matrix{Matrix{ComplexF64}},
    γ_c::Float64,
    B::Float64
)::Matrix{ComplexF64}
    ω = -γ_c*B
    S_x, S_z = S[:, 1], S[:, 3]
    @reduce H_c_0[p,q] := sum(i) ω/2*S_z[i][p,q]
    @reduce H_c_int[p,q] := sum(i,j,r) ω/10*S_x[i][p,r]*S_x[j][r,q] # TODO: case i=j???
    H_c = H_c_0 + H_c_int

    return H_c
end


function get_H_s(
    S_a::Vector{Matrix{ComplexF64}},
    S_b::Vector{Matrix{ComplexF64}},
    γ_s::Float64,
    B::Float64
)::Matrix{ComplexF64}
    ω = -γ_s*B
    S_az, S_bz = S_a[3], S_b[3]
    H_s = ω/2*(S_az + S_bz)
    return H_s
end


function get_H_cs(
    d_c::Float64,
    d_s::Float64,
    D::Float64,
    θ::Float64,
    S_a::Vector{Matrix{ComplexF64}},
    S_b::Vector{Matrix{ComplexF64}},
    S::Matrix{Matrix{ComplexF64}},
    γ_s::Float64,
    γ_c::Float64
)::Matrix{ComplexF64}
    nqubits = size(S, 1)
    μ_0 = 1.25663706212e-6 # N A^-2
    c = μ_0/(4π)*γ_s*γ_c
    # Get positions
    position_a, position_b = get_particles_coordinates(D, d_s, θ)
    positions_qubits = get_qubits_coordinates(nqubits, d_c)
    # Get relative polar coordinates
    r_a_qubits, θ_a_qubits = cartesian2polar(position_a .- positions_qubits)
    r_b_qubits, θ_b_qubits = cartesian2polar(position_b .- positions_qubits)
    # Build dipole-dipole interaction factors
    g_a = c*(1 .- 3*cos.(θ_a_qubits).^2)./r_a_qubits.^3
    g_b = c*(1 .- 3*cos.(θ_b_qubits).^2)./r_b_qubits.^3
    # Build the hamiltonian
    S_ax, S_bx, S_x = S_a[1], S_b[1], S[:, 1]
    @reduce H_cs_a[p, q] := sum(i,r) g_a[i]*S_ax[p,r]*S_x[i][r,q]
    @reduce H_cs_b[p, q] := sum(i,r) g_b[i]*S_bx[p,r]*S_x[i][r,q]
    H_cs = H_cs_a + H_cs_b

    return H_cs

end


function create_big_spin_operator(nqubits::Int)::Matrix{Matrix{ComplexF64}}
    # S = [[s_1x, s_1y, s_1z], ...], [s_nx, s_ny, s_nz]]
    # S es en verdad Vector{Vector{Matrix}}
    nparticles = 2
    n = nparticles + nqubits
    # TODO: (nqubits, 3, 2**n, 2**n)
    S = Matrix{Matrix{ComplexF64}}(undef, nqubits, 3)
    for i in 1:nqubits
        dim_l = 2^(i-1+nparticles)
        dim_r = 2^(nqubits-i)
        S[i, :] = extended_pauli_operators(dim_left=dim_l, dim_right=dim_r) #@inbound
    end
    return S
end


function build_hamiltonian(
    S::Matrix{Matrix{ComplexF64}},
    S_a::Vector{Matrix{ComplexF64}},
    S_b::Vector{Matrix{ComplexF64}},
    d_c::Float64,
    d_s::Float64,
    D::Float64,
    θ::Float64,
    γ_s::Float64,
    γ_c::Float64,
    B::Float64
)
    H_c = get_H_c(S, γ_c, B)
    H_s = get_H_s(S_a, S_b, γ_s, B)
    H_cs = get_H_cs(d_c, d_s, D, θ, S_a, S_b, S, γ_s, γ_c)
    H = H_c + H_s + H_cs
    return H
end
