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
    sites = length(p) ÷ 7 + 1
    σᵢₛₒ = Array{Distribution}(undef, sites)
    Δσ = Array{Distribution}(undef, sites)
    ησ = Array{Distribution}(undef, sites)
    weights = zeros(sites)

    weights_sum = sum(map(i -> i % 7 == 0 ? p[i] : 0.0, 1:length(p)))
    weights_sum > 1.0 && return missing
    for i in 1:length(p)
        i % 7 in [0, 2, 4, 5, 6] && p[i] < 0 && return missing
        i % 7 == 5 && p[i] > 1 && return missing
        (isnan(p[i]) || isinf(p[i])) && return missing
        (isnothing(p[i]) || ismissing(p[i])) && return missing
    end

    for i = 1:sites
        σᵢₛₒ[i] = Normal(p[7*(i-1)+1], p[7*(i-1)+2])
        Δσ[i] = Normal(p[7*(i-1)+3], p[7*(i-1)+4])
        ησ[i] = truncated(Normal(p[7*(i-1)+5], p[7*(i-1)+6]), 0.0, 1.0)
        weights[i] = i != sites ? p[7*i] : 1 - weights_sum
    end
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
    return powder_pattern
end

function get_chemical_shift_starting_values(sites::Int64)
    [rand(Uniform(-1000, 1000)), rand(Uniform(-100, 100)),
        rand(Uniform(-500, 500)), rand(Uniform(-50, 50)),
        rand(Uniform(0, 1)), rand(Uniform(0, 1))]
    starting_values = zeros(7 * sites - 1)
    starting_values[1:7:end] = rand(Uniform(-1000, 1000), sites)
    starting_values[2:7:end] = rand(Uniform(0, 100), sites)
    starting_values[3:7:end] = rand(Uniform(-500, 500), sites)
    starting_values[4:7:end] = rand(Uniform(0, 50), sites)
    starting_values[5:7:end] = rand(Uniform(0, 1), sites)
    starting_values[6:7:end] = rand(Uniform(0, 1), sites)
    if sites > 1
        starting_values[7:7:end] = rand(Uniform(0, 1 / sites), sites - 1)
    end
    return starting_values
end
