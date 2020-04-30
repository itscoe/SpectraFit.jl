using Optim, StatsBase

function ols_cdf(
    parameters::nmr_params,
    experimental,
    experimental_ecdf,
    ν0::Float64,
    I::Int64,
)
    powder_pattern = estimate_powder_pattern(parameters, 1_000_000, ν0, I)
    theoretical_ecdf = ecdf(powder_pattern).(experimental[:, 1])
    return sum((experimental_ecdf - theoretical_ecdf) .^ 2)
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
        x -> ols_cdf(nmr_params(x), experimental, experimental_ecdf, ν0, I),
        starting_values,
        Optim.Options(iterations = iters),
    )
    return result
end
