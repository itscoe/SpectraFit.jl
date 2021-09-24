using Unitful

struct ChemicalShiftI <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"ppm")
end

Base.size(_::ChemicalShiftI) = (1,)

Base.getindex(C::ChemicalShiftI, _::Int) = C.δᵢₛₒ

get_ν(δᵢₛₒ::Float64) = δᵢₛₒ

struct ChemicalShiftA <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"ppm")
    σδᵢₛₒ::typeof(1.0u"ppm")
    Δδ::typeof(1.0u"ppm")
    σΔδ::typeof(1.0u"ppm")
    ηδ::Float64
    σηδ::Float64
end

Base.size(_::ChemicalShiftA) = (6,)

Base.getindex(C::ChemicalShiftA, i::Int) = 
    i == 1 ? C.δᵢₛₒ : i == 2 ? C.σδᵢₛₒ : i == 3 ? C.Δδ : 
    i == 4 ? C.σΔδ : i == 5 ? C.ηδ : C.σηδ

get_ν(μ::Float64, λ::Float64, δᵢₛₒ::Float64, Δδ::Float64, ηδ::Float64) = 
    δᵢₛₒ + (Δδ / 2) * (3 * μ^2 - 1 + ηδ * (1-μ^2) * λ)
