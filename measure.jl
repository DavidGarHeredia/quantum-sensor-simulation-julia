using LinearAlgebra
using TensorCast


function partial_trace(A::Matrix{ComplexF64}, subsystems::Vector{Int})
    partial_A = copy(A)
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
        partial_A = reshape(partial_A, new_shape)
        @reduce partial_A[i,k,l,n] := sum(j,m) delta[j,m]*partial_A[i,j,k,l,m,n]
        n = dim_left * dim_right
        partial_A = reshape(partial_A, (n, n))
    end

    return partial_A
end


function get_probabilities(
    ρ_f::Matrix{ComplexF64},
    projectors::Vector{Matrix{Int}}
)
    ρ_f_camera = partial_trace(ρ_f, [1, 2])
    @reduce probabilities[k] := sum(i, j) ρ_f_camera[i, j] * projectors[k][j, i]
    # probabilities = [tr(ρ_f_c*P_i) for P_i in projectors] # quizas sea más lento por hacer más operaciones
    # probabilities = probabilities/np.sum(probabilities)

    return real(probabilities)
end


function measure(
    states::Vector{Vector{Int}},
    probabilities::Vector{Float64},
    nmeasures::Int
)
    # Sample from states according to probabilities
    indices = collect(1:size(states, 1))
    sampled_indices = wsample(indices, probabilities, nmeasures)
    measures = cat(states[sampled_indices]; dims=1)

    return measures
end