using Distributions

struct ChemicalShift{T <: AbstractFloat} <: NMRInteraction
    δᵢₛₒ::T
    σδᵢₛₒ::T
    Δδ::T
    σΔδ::T
    ηδ::T
    σηδ::T
end

Base.size(C::ChemicalShift) = (6,)

Base.getindex(C::ChemicalShift, i::Int) = 
    i == 1 ? C.δᵢₛₒ : i == 2 ? C.σδᵢₛₒ : i == 3 ? C.Δδ : 
    i == 4 ? C.σΔδ : i == 5 ? C.ηδ : C.σηδ

get_ν(μ::T, λ::T, δᵢₛₒ::T, Δδ::T, ηδ::T) where {T <: AbstractFloat} = 
    δᵢₛₒ + (Δδ / 2) * (3 * μ^2 - 1 + ηδ * (1-μ^2) * λ)

function estimate_powder_pattern(p::ChemicalShift, 
    μ::Array{AbstractFloat, N}, λ::Array{AbstractFloat, N}) where {N}
    δᵢₛₒ = rand(Normal(p.δᵢₛₒ, p.σδᵢₛₒ), N)
    Δδ = rand(Normal(p.Δδ, p.σΔδ), N)
    ηδ = rand(Normal(p.ηδ, p.σηδ), N)
    return get_ν.(δᵢₛₒ, Δδ, ηδ, μ, λ)
end

@inline estimate_powder_pattern(p::ChemicalShift, N::Int) = 
    estimate_powder_pattern(p, μ(N), λ(N))
