using Unitful, Distributions

struct ChemicalShiftI <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"MHz")
end

ChemicalShiftI() = ChemicalShiftI(0.0u"MHz")
labels(_::ChemicalShiftI) = ["Shift (MHz)"]

ChemicalShiftI(c::Float64) = ChemicalShiftI(Quantity(c, u"MHz"))

prior(_::ChemicalShiftI, _::Int) = Uniform(-1, 1)

Base.length(_::ChemicalShiftI) = 1

get_ν(δᵢₛₒ::Float64) = δᵢₛₒ

@inline estimate_powder_pattern(
    c::ChemicalShiftI, 
    N::Int, 
    _::ExperimentalSpectrum
) = c.δᵢₛₒ .* ones(N)

struct ChemicalShiftA <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"ppm")
    Δδ::typeof(1.0u"ppm")
    ηδ::Float64
    ρσ::typeof(1.0u"ppm")
end

labels(_::ChemicalShiftA) = ["δᵢₛₒ (ppm)", "Δδ (ppm)", "ηδ", "ρσ (ppm)"]

Base.length(_::ChemicalShiftA) = 4

prior(_::ChemicalShiftA, i::Int) = 
    i == 1 ? Uniform(-4000, 4000) :
    i == 2 ? Uniform(0, 400) : 
    i == 3 ? Uniform(0, 1) : 
             Uniform(0, 1000)

ChemicalShiftA() = ChemicalShiftA(0.0u"ppm", 0.0u"ppm", 0.0, 0.0u"ppm")

ChemicalShiftA(δᵢₛₒ::Float64, Δδ::Float64, ηδ::Float64, ρσ::Float64) = 
    ChemicalShiftA(Quantity(δᵢₛₒ, u"ppm"), Quantity(Δδ, u"ppm"), ηδ, Quantity(ρσ, u"ppm"))

get_ν(μ::Float64, λ::Float64, δᵢₛₒ::typeof(1.0u"ppm"), Δδ::typeof(1.0u"ppm"), ηδ::Float64) = 
    δᵢₛₒ + (Δδ / 2) * (3 * μ^2 - 1 + ηδ * (1-μ^2) * λ)

function estimate_powder_pattern(c::ChemicalShiftA, N::Int, 
    μs::Vector{Float64}, λs::Vector{Float64})

    σ = ustrip(c.ρσ / 2)
    μᵤ0 = -3 * ustrip(c.δᵢₛₒ) / √3
    μᵤ1 = ustrip(c.Δδ) / 2
    μᵤ5 = ustrip(c.Δδ) * c.ηδ / 2√3

    U₀ = Quantity.(rand(Normal(μᵤ0, σ), N), unit(c.δᵢₛₒ))
    U₁ = Quantity.(rand(Normal(μᵤ1, σ), N), unit(c.δᵢₛₒ))
    U₅ = Quantity.(rand(Normal(μᵤ5, σ), N), unit(c.δᵢₛₒ))

    σ₁₁s = -3U₀ ./ √3 .- U₁ .+ √3U₅
    σ₂₂s = -3U₀ ./ √3 .- U₁ .- √3U₅
    σ₃₃s = -3U₀ ./ √3 .+ 2U₁

    δᵢₛₒs = (σ₁₁s .+ σ₂₂s .+ σ₃₃s) ./ 3
    
    Δδs = Array{typeof(1.0u"ppm")}(undef, N)
    ηδs = Array{Float64}(undef, N)
    for i = 1:N
        σᵢ = sort([σ₁₁s[i], σ₂₂s[i], σ₃₃s[i]], 
            by = x -> abs(x - δᵢₛₒs[i]), rev = true)
        Δδs[i] = σᵢ[1] - δᵢₛₒs[i]
        ηδs[i] = Float64((σᵢ[3] - σᵢ[2]) / Δδs[i])
    end

    return get_ν.(μs, λs, δᵢₛₒs, Δδs, ηδs)
end

@inline estimate_powder_pattern(
    c::ChemicalShiftA, 
    N::Int, 
    _::ExperimentalSpectrum
) = estimate_powder_pattern(c, N, μ(N), λ(N))