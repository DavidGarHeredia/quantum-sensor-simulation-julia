using LinearAlgebra
using TensorCast


function partial_trace(A::Matrix{ComplexF64}, subsystems::Vector{Int})
    delta = I(2)
    n = size(A, 1)
    for (k, subsystem) in enumerate(sort(subsystems))
        i = subsystem - k
        dim_left = 2^i
        dim_right = n ÷ 2^(i+1)
        # if ket{psi} = ket{L}ket{P}ket{R}
        # --> A_{ijklmn} = bra{L_i}bra{P_j}bra{R_k} A ket{L_l}ket{P_m}ket{R_n}
        # --> partial_A = delta^{jm} A_{ijklmn}
        new_shape = (dim_left, 2, dim_right, dim_left, 2, dim_right)
        A_super_space = reshape(A, new_shape)
        @reduce A_reduced[i,k,l,n] := sum(j,m) delta[j,m]*A_super_space[i,j,k,l,m,n]
        n = dim_left * dim_right
        A = reshape(A_reduced, (n, n))
    end
    return A
end

function get_probabilities(ρ_f::Matrix{ComplexF64})
    ρ_f_camera = partial_trace(ρ_f, [1, 2])
    probabilities = diag(ρ_f_camera)
    # @reduce probabilities[k] := sum(i, j) ρ_f_camera[i, j] * projectors[k][j, i]
    # probabilities = [tr(ρ_f_c*P_i) for P_i in projectors]
    # probabilities = probabilities/np.sum(probabilities)
    return real(probabilities)
end


function measure(
    states::Vector{Vector{Int}},
    probabilities::Vector{Float64},
    nmeasures::Int
)::Vector{Vector{Int64}}
    # Sample from states according to probabilities
    indices = collect(1:size(states, 1))
    sampled_indices = wsample(indices, probabilities, nmeasures)
    local measures::Vector{Vector{Int64}} = cat(states[sampled_indices]; dims=1)
    return measures
end
