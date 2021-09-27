using Unitful, Distributions

struct ChemicalShiftI <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"MHz")
end

ChemicalShiftI() = ChemicalShiftI(0.0u"MHz")

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
    σδᵢₛₒ::typeof(1.0u"ppm")
    Δδ::typeof(1.0u"ppm")
    σΔδ::typeof(1.0u"ppm")
    ηδ::Float64
    σηδ::Float64
end

Base.length(_::ChemicalShiftA) = 6

get_ν(μ::Float64, λ::Float64, δᵢₛₒ::Float64, Δδ::Float64, ηδ::Float64) = 
    δᵢₛₒ + (Δδ / 2) * (3 * μ^2 - 1 + ηδ * (1-μ^2) * λ)
