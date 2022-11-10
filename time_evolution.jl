# using PhysicalConstants
include("hamiltonians.jl")


function get_U(H::Matrix{ComplexF64}, t::Float64)
    ħ = 1.0545718176461565e-34 # J s
    U = exp(-1im*H*t/ħ) # TODO: de nuevo, q exponencial quieres calcular???
    return U
end


function time_evolution(ρ_i::Matrix{ComplexF64}, U::Matrix{ComplexF64})
    ρ_f = U*ρ_i*U'
    return ρ_f
end
