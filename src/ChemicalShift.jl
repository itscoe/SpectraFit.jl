using Distributions

const cos2α_dist = Uniform(-1, 1)
const sinβ_dist = Uniform(-1, 1)

struct ChemicalShift
    σᵢₛₒ::Normal
    Δσ::Normal
    ησ::Truncated{Normal{Float64},Continuous,Float64}
end

function ChemicalShift(p::Array{Float64})
    return ChemicalShift(Normal(p[1], max(0.0, p[2])),
        Normal(p[3], max(0.0, p[4])),
        truncated(Normal(clamp(p[5], 0.001, 0.999), max(0.0, p[6])), 0.0, 1.0))
end

function get_ν(
    α::Float64,
    β::Float64,
    σᵢₛₒ::Normal,
    Δσ::Normal,
    ησ::Truncated{Normal{Float64},Continuous,Float64},
)
    Δσ_sample = rand(Δσ)
    σᵢₛₒ_sample = rand(σᵢₛₒ)
    ησ_sample = rand(ησ)
    σᵢₛₒ_sample - (Δσ_sample / 3) * (3 * (1 - β^2) - 1 - ησ_sample * β^2 * α);
end

function get_powder_pattern(params::ChemicalShift, N::Int64)
    α = rand(cos2α_dist, N)
    β = rand(sinβ_dist, N)
    return get_ν.(α, β, params.σᵢₛₒ, params.Δσ, params.ησ)
end;
