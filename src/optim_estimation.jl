using Optim, StatsBase

"""
    ols_cdf(parameters, experimental, experimental_ecdf, ν0, I)

Compute ordinary least squares comparing the experimental ecdf with the
theoretical cdf, calculated with the nmr parameters, the spin (I), and the
Larmor frequency (ν0) at each x-value in the experimental data

"""
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

"""
    fit_nmr(experimental, sites, iters)

Fit the NMR parameters, assuming a normal distribution and using the Nelder-Mead
optimization method for with Optim options (options) for iterations,
optimization method, timeout, etc, with the loss function of ordinary least
squares of the CDFs, assuming a specified number of sites (sites, defaults to 1)

"""
function fit_nmr(
    experimental;
    sites::Int64 = 1,
    options = Optim.Options(iterations = 1000),
    method = NelderMead(),
)
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
    if method = SAMIN()
        result = optimize(
            x -> SpectraFit.ols_cdf(
                nmr_params(x),
                experimental,
                experimental_ecdf,
                ν0,
                I,
            ),
            [0.0 0.0 0.0 0.0 0.0],
            [9.0 1.0 1.0 1.0 1.0],
            starting_values,
            SAMIN(),
        )
    else
        result = optimize(
            x -> ols_cdf(nmr_params(x), experimental, experimental_ecdf, ν0, I),
            starting_values,
            options,
        )
    end
    return result
end
