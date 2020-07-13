using Optim, StatsBase

"""
    ols_cdf(parameters, experimental, experimental_ecdf, ν0, I)

Compute ordinary least squares comparing the experimental ecdf with the
theoretical cdf, calculated with the nmr parameters, the spin (I), and the
Larmor frequency (ν0) at each x-value in the experimental data

"""
function ols_cdf(
    parameters::Quadrupolar,
    experimental,
    experimental_ecdf,
    ν0::Float64;
    I::Int64 = 3,
    samples = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    theoretical_ecdf = ecdf(estimate_powder_pattern(parameters, samples, ν0, I, transitions = transitions)).(experimental[:, 1])
    theoretical_ecdf .-= theoretical_ecdf[1]
    theoretical_ecdf .*= 1 / theoretical_ecdf[end]
    return sum((experimental_ecdf - theoretical_ecdf) .^ 2)
end

function ols_cdf(
    parameters::Missing,
    experimental,
    experimental_ecdf,
    ν0::Float64;
    I::Int64 = 3,
    samples = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    return Inf
end

function ols_cdf(
    parameters::ChemicalShift,
    experimental::Array{Float64, 2},
    experimental_ecdf::Array{Float64, 1};
    samples = 1_000_000,
)
    th_ecdf = ecdf(estimate_powder_pattern(parameters, samples)).(experimental[:, 1])
    return sum((experimental_ecdf .- th_ecdf) .^ 2)
end

function ols_cdf(
    parameters::Missing,
    experimental::Array{Float64, 2},
    experimental_ecdf::Array{Float64, 1};
    samples = 1_000_000,
)
    return Inf
end

"""
    fit_quadrupolar(experimental; sites, iters, options, method)

Fit the NMR parameters, assuming a normal distribution and using the specified
optimization method (currently implementer are Nelder-Mead and Simulated
Annealing) for with Optim options (options) for iterations, timeout, etc, with
the loss function of ordinary least squares of the CDFs, assuming a specified
number of sites (sites, defaults to 1)

"""
function fit_quadrupolar(
    experimental;
    sites::Int64 = 1,
    iters::Int64 = 1000,
    options = Optim.Options(iterations = iters),
    method = NelderMead(),
    I = 3,
    samples = 1_000_000,
    starting_values = get_quadrupolar_starting_values(sites),
    transitions::UnitRange{Int64} = 1:(2*I),
    range::Tuple{Float64,Float64} = (experimental[1, 1], experimental[end, 1]),
)
    range = (findfirst(x -> range[1] < x, experimental[:, 1]),
        findlast(x -> range[2] > x, experimental[:, 1]) - 1)
    experimental = experimental[range[1]:range[2], :]

    experimental_ecdf = get_experimental_ecdf(experimental)
    ν0 =  get_ν0(experimental, experimental_ecdf)

    if method == SAMIN()
        upper_bounds, lower_bounds = zeros(5 * sites), zeros(5 * sites)
        upper_bounds[1:5:end] .= 7  # Qcc
        upper_bounds[2:5:end] .= 1  # σQcc
        upper_bounds[3:5:end] .= 1  # η
        upper_bounds[4:5:end] .= 1  # ση
        upper_bounds[5:5:end] .= 1  # weights
        result = optimize(
            x -> ols_cdf(  # objective function
                Quadrupolar(x),
                experimental,
                experimental_ecdf,
                ν0,
                I = I,
                samples = samples,
                transitions = transitions,
            ),
            lower_bounds,
            upper_bounds,
            starting_values,
            SAMIN(),
            options,
        )
    else
        result = optimize(
            x -> ols_cdf(
                Quadrupolar(x),
                experimental,
                experimental_ecdf,
                ν0,
                I = I,
                samples = samples,
                transitions = transitions,
            ),
            starting_values,
            options,
        )
    end
    return result
end

function fit_chemical_shift(
    experimental;
    sites::Int64 = 1,
    iters::Int64 = 1000,
    options = Optim.Options(iterations = iters),
    method = NelderMead(),
    samples = 1_000_000,
    starting_values = get_chemical_shift_starting_values(sites),
    range::Tuple{Float64,Float64} = (experimental[1, 1], experimental[end, 1]),
)
    range = (findfirst(x -> range[1] < x, experimental[:, 1]),
        findlast(x -> range[2] > x, experimental[:, 1]) - 1)
    experimental = experimental[range[1]:range[2], :]

    experimental_ecdf = get_experimental_ecdf(experimental)

    if method == SAMIN()
        upper_bounds, lower_bounds = zeros(5 * sites), zeros(5 * sites)
        upper_bounds[1:end] .= Inf
        upper_bounds[5:6:end] = 1
        lower_bounds[1:end] .= -Inf
        lower_bounds[2:6:end] .= 0
        lower_bounds[4:6:end] .= 0
        lower_bounds[6:6:end] .= 0
        result = optimize(
            x -> ols_cdf(  # objective function
                ChemicalShift(x),
                experimental,
                experimental_ecdf,
                samples = samples,
            ),
            lower_bounds,
            upper_bounds,
            starting_values,
            SAMIN(),
            options,
        )
    else
        result = optimize(
            x -> ols_cdf(
                ChemicalShift(x),
                experimental,
                experimental_ecdf,
                samples = samples,
            ),
            starting_values,
            options,
        )
    end
    return result
end
