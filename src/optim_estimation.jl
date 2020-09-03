using Optim, StatsBase

"""
    ols_cdf(parameters, experimental, experimental_ecdf, ν0, I)

Compute ordinary least squares comparing the experimental ecdf with the
theoretical cdf, calculated with the nmr parameters, the spin (I), and the
Larmor frequency (ν0) at each x-value in the experimental data

"""
function ols_cdf(
    parameters::Quadrupolar,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1},
    ν0::Float64;
    I::Int64 = 3,
    samples::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
    range::Tuple{Float64,Float64} = (experimental[1, 1], experimental[end, 1]),
)
    range = (findfirst(x -> range[1] < x, exp[:, 1]),
        findlast(x -> range[2] > x, exp[:, 1]) - 1)
    th_ecdf = ecdf(estimate_powder_pattern(parameters, samples, ν0, I,
        transitions = transitions)).(exp[:, 1])
    return sum((exp_ecdf[range[1]:range[2]] .- th_ecdf[range[1]:range[2]]) .^ 2)
end

function ols_cdf(
    parameters::Missing,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1},
    ν0::Float64;
    I::Int64 = 3,
    samples::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    return size(exp)[1] / 10
end

function ols_cdf(
    parameters::ChemicalShift,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1};
    samples::Int64 = 1_000_000,
)
    th_ecdf = ecdf(estimate_powder_pattern(parameters, samples)).(exp[:, 1])
    return sum((exp_ecdf .- th_ecdf) .^ 2)
end

function ols_cdf(
    parameters::Missing,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1};
    samples::Int64 = 1_000_000,
)
    println("oops")
    return size(exp)[1] / 10
end

function transform_params(x::Array{Float64}, T::DataType)
    if T == Quadrupolar
        x = x .^ 2
    elseif T == ChemicalShift
        for i = 1:length(x) ÷ 7
            x[7 * (i - 1) + 2] ^= 2
            x[7 * (i - 1) + 4] ^= 2
            x[7 * (i - 1) + 5] ^= 2
            x[7 * (i - 1) + 6] ^= 2
            x[7 * (i - 1) + 7] ^= 2
        end
    end
    return x
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
    experimental::Array{Float64, 2};
    sites::Int64 = 1,
    iters::Int64 = 1000,
    options = Optim.Options(iterations = iters),
    method = NelderMead(),
    I::Int64 = 3,
    samples::Int64 = 1_000_000,
    starting_values = get_quadrupolar_starting_values(sites),
    transitions::UnitRange{Int64} = 1:(2*I),
    range::Tuple{Float64,Float64} = (experimental[1, 1], experimental[end, 1]),
)
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
                range = range,
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
                Quadrupolar(transform_params(x, Quadrupolar)),
                experimental,
                experimental_ecdf,
                ν0,
                I = I,
                samples = samples,
                transitions = transitions,
                range = range,
            ),
            starting_values,
            options,
        )
    end
    return result
end

function fit_chemical_shift(
    experimental::Array{Float64, 2};
    sites::Int64 = 1,
    iters::Int64 = 1000,
    options = Optim.Options(iterations = iters),
    method = NelderMead(),
    samples::Int64 = 1_000_000,
    starting_values = get_chemical_shift_starting_values(sites),
    range::Tuple{Float64,Float64} = (experimental[1, 1], experimental[end, 1]),
)
    experimental_ecdf = get_experimental_ecdf(experimental)

    if method == SAMIN()
        upper_bounds, lower_bounds = zeros(7 * sites), zeros(7 * sites)
        upper_bounds[1:end] .= Inf
        upper_bounds[5:7:end] .= 1
        upper_bounds[7:7:end] .= 1
        lower_bounds[1:end] .= -Inf
        lower_bounds[2:7:end] .= 0
        lower_bounds[4:7:end] .= 0
        lower_bounds[6:7:end] .= 0
        lower_bounds[7:7:end] .= 0
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
                ChemicalShift([Normal(x[1], x[2] ^ 2)], [Normal(x[3], x[4] ^ 2)],
                    [truncated(Normal(x[5] ^ 2, x[6] ^ 2), 0, 1)], [1.0]),
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
