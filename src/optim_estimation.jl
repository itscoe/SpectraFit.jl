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

function get_experimental_ecdf(experimental)
    return cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
end

function get_ν0(experimental, experimental_ecdf)
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    return experimental[end, 1] - riemann_sum
end

"""
    fit_nmr(experimental; sites, iters, options, method)

Fit the NMR parameters, assuming a normal distribution and using the specified
optimization method (currently implementer are Nelder-Mead and Simulated
Annealing) for with Optim options (options) for iterations, timeout, etc, with
the loss function of ordinary least squares of the CDFs, assuming a specified
number of sites (sites, defaults to 1)

"""
function fit_nmr(
    experimental;
    sites::Int64 = 1,
    iters::Int64 = 1000,
    options = Optim.Options(iterations = iters),
    method = NelderMead(),
    I = 3,
)
    experimental_ecdf = get_experimental_ecdf(experimental)
    ν0 =  get_ν0(experimental, experimental_ecdf)

    starting_values = zeros(5 * sites)
    starting_values[1:5:end] = rand(Uniform(0, 9), sites)  # Qcc
    starting_values[2:5:end] = rand(Uniform(0, 1), sites)  # σQcc
    starting_values[3:5:end] = rand(Uniform(0, 1), sites)  # η
    starting_values[4:5:end] = rand(Uniform(0, 1), sites)  # ση
    starting_values[5:5:end] = rand(Uniform(0, 1), sites)  # weights

    if method == SAMIN()
        result = optimize(
            x -> SpectraFit.ols_cdf(  # objective function
                nmr_params(x),
                experimental,
                experimental_ecdf,
                ν0,
                I,
            ),
            [0.0 0.0 0.0 0.0 0.0],  # lower bounds
            [9.0 1.0 1.0 1.0 1.0],  # upper bounds
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
