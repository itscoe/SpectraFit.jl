using StatsBase

"""
    get_experimental_ecdf(experimental)

Returns the ecdf of the experimental data over the range of the experimental
data

"""
function get_experimental_ecdf(experimental::Array{Float64,2})
    return cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
end

"""
    get_ν0(experimental, experimental_ecdf)

Calculates the Larmor frequency by finding the approximate mean of the
distribution. Should be passed into the forward model to get theoretical data
that lines up with the experimental

"""
function get_ν0(experimental::Array{Float64,2}, experimental_ecdf)
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    return experimental[end, 1] - riemann_sum
end

function get_ν0(experimental::Array{Float64,2})
    return get_ν0(experimental, get_experimental_ecdf(experimental))
end

"""
    ols_cdf(parameters, experimental, experimental_ecdf, ν0, I)

Compute ordinary least squares comparing the experimental ecdf with the
theoretical cdf, calculated with the NMR parameters, the spin (I), and the
Larmor frequency (ν0) at each x-value in the experimental data

"""
function ols_cdf(
    parameters::Quadrupolar,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1},
    ν0::Float64;
    I::Int64 = 3,
    N::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    th_ecdf = ecdf(estimate_powder_pattern(parameters, N, ν0, I,
        transitions = transitions)).(exp[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]
    return sum((exp_ecdf .- th_ecdf) .^ 2)
end

function ols_cdf(
    parameters::Array{Quadrupolar},
    weights::Array{Float64},
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1},
    ν0::Float64;
    I::Int64 = 3,
    N::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    w = vcat([0], floor.(Int64, cumsum(weights) .* N), [N])
    powder_pattern = zeros(N)
    for i = 1:length(parameters)
        powder_pattern[w[i]+1:w[i+1]] = estimate_powder_pattern(parameters[i],
            w[i+1] - w[i], ν0, I, transitions = transitions)
    end
    th_ecdf = ecdf(powder_pattern).(exp[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]
    return sum((exp_ecdf .- th_ecdf) .^ 2)
end

function ols_cdf(
    parameters::ChemicalShift,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1};
    N::Int64 = 1_000_000,
)
    th_ecdf = ecdf(estimate_powder_pattern(parameters, N)).(exp[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]
    return sum((exp_ecdf .- th_ecdf) .^ 2)
end

function ols_cdf(
    parameters::Array{ChemicalShift},
    weights::Array{Float64},
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1};
    N::Int64 = 1_000_000,
)
    w = vcat([0], floor.(Int64, cumsum(weights) .* N), [N])
    powder_pattern = zeros(N)
    for i = 1:length(parameters)
        powder_pattern[w[i]+1:w[i+1]] = estimate_powder_pattern(parameters[i],
            w[i+1] - w[i])
    end
    th_ecdf = ecdf(powder_pattern).(exp[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]
    return sum((exp_ecdf .- th_ecdf) .^ 2)
end
