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

Get the frequency given the Euler angles (μ, λ) 
and the parameters (δᵢₛₒ, Δδ, ηδ)

"""
get_ν(
    μ::Float64, 
    λ::Float64, 
    δᵢₛₒ::typeof(1.0u"ppm"), 
    Δδ::typeof(1.0u"ppm"), 
    ηδ::Float64
) = δᵢₛₒ + 0.5 * Δδ * (μ^2 * (3 - ηδ * λ) + ηδ * λ - 1)

"""
    estimate_static_powder_pattern(c, N, μs, λs)

Get the estimated powder pattern (a vector of N frequencies) given the CSA 
interaction and vectors of the Euler angles (μs, λs)

"""
function estimate_static_powder_pattern(
    c::ChemicalShiftA, 
    N::Int64,
    μs::Vector{Float64}, 
    λs::Vector{Float64},
    _::Vector{FPOT},
    U0_rand::Vector{Float64},
    U1_rand::Vector{Float64},
    U5_rand::Vector{Float64},
    _::FPOT,
    _::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz"),
    _::typeof(1.0u"T^-1")
)
    σ = 0.5 * c.ρσ
    U₀ = c.δᵢₛₒ .+ √3σ .* U0_rand
    U₁ = (0.5 * c.Δδ) .+ σ .* U1_rand
    U₅ = (0.5 * c.Δδ * c.ηδ) .+ √3σ .* U5_rand
    σ₁₁s = U₀ .- U₁ .+ U₅
    σ₂₂s = σ₁₁s .- 2U₅
    σ₃₃s = U₀ .+ 2U₁
    
    Δδs = Array{typeof(1.0u"ppm")}(undef, N)
    ηδs = Array{Float64}(undef, N)
    for i = 1:N
        σᵢ = sort([σ₁₁s[i], σ₂₂s[i], σ₃₃s[i]], 
            by = x -> abs(x - U₀[i]), rev = true)
        Δδs[i] = σᵢ[1] - U₀[i]
        ηδs[i] = Float64((σᵢ[3] - σᵢ[2]) / Δδs[i])
    end

    return get_ν.(μs, λs, U₀, Δδs, ηδs) ./ ν_step
end

"""
    estimate_static_powder_pattern(c, N, exp)

Get the estimated powder pattern (a vector of N frequencies) 
given the CSA interaction

"""
@inline estimate_static_powder_pattern(
    c::ChemicalShiftA, 
    N::Int, 
    exp::ExperimentalSpectrum
) = to_Hz.(estimate_static_powder_pattern(c, N, μ(N), λ(N)), exp.ν₀) .- exp.ν₀