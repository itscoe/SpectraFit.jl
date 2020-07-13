using Distributions

const cos2α_dist = Uniform(-1, 1)
const sinβ_dist = Uniform(-1, 1)

struct ChemicalShift
    σᵢₛₒ::Array{Distribution}
    Δσ::Array{Distribution}
    ησ::Array{Distribution}
    weights::Array{Float64}
end

function ChemicalShift(p::Array{Float64})
    sites = length(p) ÷ 7
    σᵢₛₒ = Array{Distribution}(undef, sites)
    Δσ = Array{Distribution}(undef, sites)
    ησ = Array{Distribution}(undef, sites)
    weights = zeros(sites)
    for i in 1:length(p)
        if i % 7 in [0, 2, 4, 5, 6]
            p[i] < 0 && return missing
        end
        if i % 7 == 5
            p[i] > 1 && return missing
        end
    end
    for i = 1:sites
        σᵢₛₒ[i] = Normal(p[7*(i-1)+1], p[7*(i-1)+2])
        Δσ[i] = Normal(p[7*(i-1)+3], p[7*(i-1)+4])
        ησ[i] = truncated(Normal(p[7*(i-1)+5], p[7*(i-1)+6]), 0.0, 1.0)
        weights[i] = p[7*i]
    end
    weights ./= sum(weights)
    return ChemicalShift(σᵢₛₒ, Δσ, ησ, weights)
end

function get_ν(
    α::Float64,
    β::Float64,
    σᵢₛₒ::Distribution,
    Δσ::Distribution,
    ησ::Distribution,
)
    rand(σᵢₛₒ) - (rand(Δσ) / 3) * (3 * (1 - β^2) - 1 - rand(ησ) * β^2 * α);
end

function estimate_powder_pattern(p::ChemicalShift, N::Int64)
    powder_pattern = zeros(N)
    i = 1
    for j = 1:(length(p.weights) ÷ 7 - 1)
        to_add = floor(Int, p.weights[j] * N)
        α = rand(cos2α_dist, to_add)
        β = rand(sinβ_dist, to_add)
        powder_pattern[i:(i+to_add-1)] = get_ν.(α, β, p.σᵢₛₒ[j], p.Δσ[j], p.ησ[j])
        i += to_add
    end
    α = rand(cos2α_dist, N - i + 1)
    β = rand(sinβ_dist, N - i + 1)
    powder_pattern[i:end] = get_ν.(α, β, p.σᵢₛₒ[end], p.Δσ[end], p.ησ[end])
    if NaN in powder_pattern
        println(p)
    end
    return powder_pattern
end
