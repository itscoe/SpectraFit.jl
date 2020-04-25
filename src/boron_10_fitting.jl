using Distributions, CSV, Optim, StatsBase, KernelDensity

function get_ν(
    qcc::Float64,
    η::Float64,
    μ::Float64,
    λ::Float64,
    m::Int64,
    I::Int64,
    ν0::Float64,
)
    νQ = 3 * qcc / (2 * I * (2 * I - 1))
    β = νQ / ν0
    a = -(3 * μ^2 - 1 + η * λ - η * λ * μ^2)
    c = (0.5 * ((I + 3 / 2) * (I - 1 / 2) - 3 * (m - 1 / 2)^2)) *
        (μ^4 * (3 - η * λ)^2 + 2 * μ^2 * (-9 + 2 * η^2 - η^2 * λ^2) +
         (3 + η * λ)^2) +
        (4 * ((I + 3 / 2) * (I - 1 / 2) - 6 * (m - 1 / 2)^2)) *
        (μ^4 * (3 - η * λ)^2 + μ^2 * (-9 + η^2 + 6 * η * λ - 2 * η^2 * λ^2) +
         (-(η^2) + η^2 * λ^2))
    e = (12 * I * (I + 1) - 40 * m * (m - 1) - 27) * (μ^6 * (3 - η * λ)^3 +
         μ^4 * (-36 + 3 * η^2 + 42 * η * λ - η^3 * λ - 19 * η^2 * λ^2 +
          3 * η^3 * λ^3) +
         μ^2 * (9 - 4 * η^2 - 15 * η * λ + 2 * η^3 * λ + 11 * η^2 * λ^2 -
          3 * η^3 * λ^3) +
         (η^2 - η^3 * λ - η^2 * λ^2 + η^3 * λ^3)) +
        (1 / 2 * (3 * I * (I + 1) - 5 * m * (m - 1) - 6)) *
        (μ^6 * (3 - η * λ)^3 +
         μ^4 * (-63 + 12 * η^2 + 33 * η * λ - 4 * η^3 * λ - 13 * η^2 * λ^2 +
          3 * η^3 * λ^3) +
         μ^2 *
         (45 - 4 * η^2 - 9 * η * λ + 4 * η^3 * λ - η^2 * λ^2 - 3 * η^3 * λ^3) +
         (-9 + 3 * η * λ + 5 * η^2 * λ^2 + η^3 * λ^3)) +
        (8 * I * (I + 1) - 20 * m * (m - 1) - 15) * (μ^6 * (3 - η * λ)^3 +
         μ^4 * (-54 + 9 * η^2 + 36 * η * λ - 3 * η^3 * λ - 15 * η^2 * λ^2 +
          3 * η^3 * λ^3) +
         μ^2 * (27 - 6 * η^2 - 9 * η * λ + 4 * η^3 * λ + 3 * η^2 * λ^2 -
          3 * η^3 * λ^3) +
         (-3 * η^2 - η^3 * λ + 3 * η^2 * λ^2 + η^3 * λ^3))
    return ν0 + (νQ / 2) * (m - 1 / 2) * a + (νQ * β / 72) * c +
           (νQ * β^2 / 144) * e * (m - 1 / 2)
end;

struct nmr_params
    qcc::Array{Distribution}
    η::Array{Distribution}
    weights::Array{Float64}
end

function nmr_params(p::Array{Float64})
    sites = length(p) ÷ 5
    qcc = Array{Distribution}(undef, sites)
    η = Array{Distribution}(undef, sites)
    weights = zeros(sites)
    for i = 1:sites
        qcc[i] = Normal(p[sites*(i-1)+1], max(0, p[sites*(i-1)+2]))
        η[i] = Normal(p[sites*(i-1)+3], max(0, p[sites*(i-1)+4]))
        weights[i] = p[sites*(i-1)+5]
    end
    weights ./= sum(weights)
    return nmr_params(qcc, η, weights)
end;

function estimate_powder_pattern(
    qcc_dist::Distribution,
    η_dist::Distribution,
    N::Int64,
    ν0::Float64,
    I::Int64,
)
    qcc = rand(qcc_dist, N)
    η = rand(η_dist, N)
    m = rand(
        Categorical([3 / 28, 5 / 28, 6 / 28, 6 / 28, 5 / 28, 3 / 28]),
        N,
    ) .- I
    μ = rand(Uniform(0, 1), N)
    λ = rand(Uniform(-1, 1), N)
    return filter(isfinite, filter(!isnan, get_ν.(qcc, η, μ, λ, m, I, ν0)))
end;

function estimate_powder_pattern(p::nmr_params, N::Int64, ν0::Float64, I::Int64)
    powder_pattern = zeros(N)
    i = 1
    for j = 1:(length(p.weights)-1)
        to_add = floor(Int, p.weights[j] * N)
        powder_pattern[i:(i+to_add-1)] = estimate_powder_pattern(
            p.qcc[j],
            p.η[j],
            to_add,
            ν0,
            I,
        )
        i += to_add
    end
    powder_pattern[i:end] = estimate_powder_pattern(
        p.qcc[length(p.weights)],
        p.η[length(p.weights)],
        N - i + 1,
        ν0,
        I,
    )
    return powder_pattern
end;

function ols_cdf(
    p::Array{Float64},
    experimental,
    experimental_ecdf,
    ν0::Float64,
    I::Int64,
)
    parameters = nmr_params(p)
    powder_pattern = estimate_powder_pattern(parameters, 1_000_000, ν0, I)
    theoretical_ecdf = ecdf(powder_pattern).(experimental[:, 1])
    return sum((experimental_ecdf - theoretical_ecdf) .^ 2)
end;

function get_experimental(filename::String, ν0_guess::Float64)
    experimental = CSV.read(filename, delim = "  ", header = false)
    experimental[!, 1] = (parse.(Float64, experimental[:, 1]) .* ν0_guess) /
                         (10^6) .+ ν0_guess
    return [reverse(experimental[:, 1]) reverse(experimental[:, 2])]
end

function fit_nmr(experimental, sites::Int64, iters::Int64)
    experimental_ecdf = cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    ν0 = experimental[end, 1] - riemann_sum
    starting_values = zeros(5 * sites)
    starting_values[1:5:end] = rand(Uniform(0, 9), sites)
    starting_values[2:5:end] = rand(Uniform(0, 1), sites)
    starting_values[3:5:end] = rand(Uniform(0, 1), sites)
    starting_values[4:5:end] = rand(Uniform(0, 1), sites)
    starting_values[5:5:end] = rand(Uniform(0, 1), sites)
    I = 3
    result = optimize(
        x -> ols_cdf(x, experimental, experimental_ecdf, ν0, I),
        starting_values,
        Optim.Options(iterations = iters),
    )
    return result
end;

function generate_theoretical_spectrum(experimental, nmr_params)
    experimental_ecdf = cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    ν0 = experimental[end, 1] - riemann_sum
    I = 3
    powder_pattern = estimate_powder_pattern(nmr_params, 1_000_000, ν0, I)
    k = kde(powder_pattern)
    x = experimental[:, 1]
    ik = InterpKDE(k)
    theoretical = pdf(ik, x)
    return (mean(experimental[:, 2]) / mean(theoretical)) .* theoretical
end;
