using Unitful, Distributions

"""
    ChemicalShiftA

CSA interaction under the Extended Czjzek model

# Fields
- `δᵢₛₒ`
- `Δδ`
- `ηδ`
- `ρσ`
"""
struct ChemicalShiftA <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"ppm")
    Δδ::typeof(1.0u"ppm")
    ηδ::Float64
    ρσ::typeof(1.0u"ppm")
end

"""
    labels(csa)

Get the labels of each parameter for plotting purposes

"""
labels(_::ChemicalShiftA) = ["δᵢₛₒ (ppm)", "Δδ (ppm)", "ηδ", "ρσ (ppm)"]

"""
    length(csa)

Get the number of free parameters of this interaction (4)

"""
@inline Base.length(_::ChemicalShiftA) = 4

"""
    prior(csa, i)

Get the prior distribution of the ith parameter of the CSA interaction

"""
prior(_::ChemicalShiftA, i::Int) = 
    i == 1 ? Uniform(-4000, 4000) :
    i == 2 ? Uniform(-4000, 4000) : 
    i == 3 ? Uniform(0, 1) : 
             Uniform(0, 1000)

"""
    ChemicalShiftA()

Default constructor for the CSA interaction

"""
ChemicalShiftA() = ChemicalShiftA(0.0u"ppm", 0.0u"ppm", 0.0, 0.0u"ppm")

"""
    ChemicalShiftA(δᵢₛₒ, Δδ, ηδ, ρσ)

Construct CSA interaction from floats, assuming ppm as units

"""
ChemicalShiftA(δᵢₛₒ::Float64, Δδ::Float64, ηδ::Float64, ρσ::Float64) = 
    ChemicalShiftA(
        Quantity(δᵢₛₒ, u"ppm"), 
        Quantity(Δδ, u"ppm"), 
        ηδ, 
        Quantity(ρσ, u"ppm")
    )

"""
    get_ν(μ, λ, δᵢₛₒ, Δδ, ηδ)

Get the frequency given the Euler angles (μ, λ) and the parameters (δᵢₛₒ, Δδ, ηδ)

"""
function get_ν(
    μ::Float64, 
    λ::Float64, 
    δᵢₛₒ::typeof(1.0u"ppm"), 
    σ₁₁::typeof(1.0u"ppm"), 
    σ₂₂::typeof(1.0u"ppm"),
    σ₃₃::typeof(1.0u"ppm"),
)
    σ = σ₁₁, σ₂₂, σ₃₃
    σ₁ᵢ = argmin(abs.(σ))
    σ₂ᵢ, σ₃ᵢ = σ₁ᵢ == 1 ? (2, 3) : σ₁ᵢ == 2 ? (1, 3) : (1, 2)
    @inbounds Δδ = σ[σ₁ᵢ]
    @inbounds ηδ = Float64(abs(σ[σ₃ᵢ] - σ[σ₂ᵢ]) / Δδ)
    return δᵢₛₒ + 0.5 * Δδ * (μ^2 * (3 - ηδ * λ) + ηδ * λ - 1)
end

"""
    estimate_powder_pattern(c, N, μs, λs)

Get the estimated powder pattern (a vector of N frequencies) given the CSA interaction and vectors of the Euler angles 
(μs, λs)

"""
function estimate_powder_pattern(
    c::ChemicalShiftA, 
    _::Int64,
    μs::Vector{Float64}, 
    λs::Vector{Float64},
    _::Vector{FPOT},
    U0_rand::Vector{Float64},
    U1_rand::Vector{Float64},
    U5_rand::Vector{Float64},
    _::FPOT,
    ν₀::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz"),
    _::typeof(1.0u"T^-1"),
    _::Symbol
)
    σ = (√3 / 2) * c.ρσ
    U₀ = c.δᵢₛₒ .+ σ .* U0_rand
    U₁ = (-0.5 * c.Δδ) .- (√3 / 3) * σ .* U1_rand
    U₅ = (0.5 * c.Δδ * c.ηδ) .- σ .* U5_rand
    σ₁₁s, σ₂₂s, σ₃₃s = U₁ .+ U₅, U₁ .- U₅, 2U₁
    return (to_Hz.(get_ν.(μs, λs, U₀, σ₁₁s, σ₂₂s, σ₃₃s), ν₀) .- ν₀) ./ ν_step
end
