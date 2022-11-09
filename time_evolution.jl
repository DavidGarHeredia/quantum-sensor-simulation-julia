# using PhysicalConstants


function get_U(H::Matrix{ComplexF64}, t::Float64)
    U = exp(-1im*H*t/ħ)

    return U
end


function time_evolution(ρ_i::Matrix{ComplexF64}, U::Matrix{ComplexF64})
    ρ_f = U*ρ_i*U'

    return ρ_f
end