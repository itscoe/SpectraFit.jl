using Distributions

struct ChemicalShift <: Interaction
    δᵢₛₒ::Float64
    σδᵢₛₒ::Float64
    Δδ::Float64
    σΔδ::Float64
    ηδ::Float64
    σηδ::Float64
end

function get_ν(
    μ::Float64,
    ϕ::Float64,
    δᵢₛₒ::Float64,
    Δδ::Float64,
    ηδ::Float64,
)
    σᵢₛₒ + (Δσ / 2) * (3 * μ^2 - 1 + ησ * (1-μ^2) * cos(2 * ϕ));
end

function estimate_powder_pattern(p::ChemicalShift, N::Int64)
    powder_pattern = zeros(N)
    δᵢₛₒ_dist = Normal(p.δᵢₛₒ, p.σδᵢₛₒ)
    Δδ_dist = Normal(p.Δδ, p.σΔδ)
    ηδ_dist = Normal(p.ηδ, p.σηδ)

    @simd for i = 1:N
        powder_pattern[i] = get_ν(
            rand(δᵢₛₒ_dist),
            rand(Δδ_dist),
            rand(ηδ_dist),
            μ[i],
            λ[i],
        )
    end

    return powder_pattern
end

function estimate_powder_pattern(
    p::ChemicalShift,
    N::Int64,
    μ::Array{Float64},
    λ::Array{Float64},
)
    powder_pattern = zeros(N)
    δᵢₛₒ_dist = Normal(p.δᵢₛₒ, p.σδᵢₛₒ)
    Δδ_dist = Normal(p.Δδ, p.σΔδ)
    ηδ_dist = Normal(p.ηδ, p.σηδ)

    @simd for i = 1:N
        powder_pattern[i] = get_ν(
            rand(δᵢₛₒ_dist),
            rand(Δδ_dist),
            rand(ηδ_dist),
            μ[i],
            λ[i],
        )
    end

    return powder_pattern
end