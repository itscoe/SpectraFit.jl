using Distributions, DataFrames

const θ_dist = Uniform(0, π)
const ϕ_dist = Uniform(0, 2π)

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
        i % 7 in [0, 2, 4, 5, 6] && p[i] < 0 && return missing
        (isnan(p[i]) || isinf(p[i])) && return missing
        (isnothing(p[i]) || ismissing(p[i])) && return missing
    end

    for i = 1:sites
        σᵢₛₒ[i] = Normal(p[7*(i-1)+1], p[7*(i-1)+2])
        Δσ[i] = Normal(p[7*(i-1)+3], p[7*(i-1)+4])
        ησ[i] = truncated(Normal(p[7*(i-1)+5], p[7*(i-1)+6]), 0.0, 1.0)
        weights[i] = p[7*i]
    end
    weights /= sum(weights)

    return ChemicalShift(σᵢₛₒ, Δσ, ησ, weights)
end

function get_ν(
    θ::Float64,
    ϕ::Float64,
    σᵢₛₒ::Distribution,
    Δσ::Distribution,
    ησ::Distribution,
)
    rand(σᵢₛₒ) + (rand(Δσ) / 2) * (3 * cos(θ)^2 - 1 - rand(ησ) * sin(θ)^2 * cos(2 * ϕ));
end

function estimate_powder_pattern(p::ChemicalShift, N::Int64)
    powder_pattern = zeros(N)
    i = 1
    for j = 1:(length(p.weights) ÷ 7 - 1)
        to_add = floor(Int, p.weights[j] * N)
        θ = rand(θ_dist, N - i + 1)
        ϕ = rand(ϕ_dist, N - i + 1)
        powder_pattern[i:(i + to_add - 1)] = get_ν.(θ, ϕ, p.σᵢₛₒ[j], p.Δσ[j], p.ησ[j])
        i += to_add
    end
    θ = rand(θ_dist, N - i + 1)
    ϕ = rand(ϕ_dist, N - i + 1)
    powder_pattern[i:end] = get_ν.(θ, ϕ, p.σᵢₛₒ[end], p.Δσ[end], p.ησ[end])
    return powder_pattern
end

function get_chemical_shift_starting_values(sites::Int64)
    starting_values = zeros(7 * sites)
    starting_values[1:7:end] = rand(Uniform(-4000, 4000), sites)
    starting_values[2:7:end] = rand(Uniform(0.000001, sqrt(800)), sites)
    starting_values[3:7:end] = rand(Uniform(-4000, 4000), sites)
    starting_values[4:7:end] = rand(Uniform(0.000001, sqrt(400)), sites)
    starting_values[5:7:end] = rand(Uniform(0, 1), sites)
    starting_values[6:7:end] = rand(Uniform(0.000001, 1), sites)
    starting_values[7:7:end] = map(x -> 1 / sites, 1:sites)
    return starting_values
end

function get_output_table(minimizer::Array{Float64})
    minimizer = transform_params(push!(minimizer,1.0), ChemicalShift)
    df = DataFrame(["σᵢₛₒ" minimizer[1] minimizer[2]
                    "Δσ" minimizer[3] minimizer[4]
                    "ησ"  minimizer[5]  minimizer[6]])
    rename!(df, Symbol.(["Parameter","Mean","St. Dev."]))
    return df
end
