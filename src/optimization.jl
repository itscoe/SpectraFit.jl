using GalacticOptim, Optim, BlackBoxOptim, StatsBase, Evolutionary, NLopt, Match

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
    N::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
    range::Tuple{Float64,Float64} = (exp[1, 1], exp[end, 1]),
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
    range::Tuple{Float64,Float64} = (exp[1, 1], exp[end, 1]),
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

function quadrupolar_opt_func(x, p)
    if p[6] == 1
        parameters = Quadrupolar(abs(x[1]), abs(x[2]), abs(x[3]), abs(x[4]))
        experimental = p[1]
        experimental_ecdf = get_experimental_ecdf(experimental)
        ν0 =  get_ν0(experimental, experimental_ecdf)

        return ols_cdf(
            parameters,
            experimental,
            experimental_ecdf,
            ν0,
            N = p[2],
            transitions = p[3],
            range = p[4],
            I = p[5],
        )
    else
        parameters = map(i -> Quadrupolar(abs(x[5*(i-1)+1]), abs(x[5*(i-1)+2]), abs(x[5*(i-1)+3]), abs(x[5*(i-1)+4])), 1:p[6])
        if p[6] == 2
            weights = [x[5]]
        else
            weights = x[5:5:end]
        end
        if sum(weights) > 1
            weights ./= sum(weights)
        end
        experimental = p[1]
        experimental_ecdf = get_experimental_ecdf(experimental)
        ν0 =  get_ν0(experimental, experimental_ecdf)
        return ols_cdf(
            parameters,
            weights,
            experimental,
            experimental_ecdf,
            ν0,
            N = p[2],
            transitions = p[3],
            range = p[4],
            I = p[5],
        )
    end
end

function quadrupolar_opt(
    exp::Array{Float64, 2};
    method::Symbol = :BBO,
    I::Int64 = 3,
    N::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
    range::Tuple{Float64,Float64} = (exp[1, 1], exp[end, 1]),
    sites::Int64 = 1,
    starting_values = get_quadrupolar_starting_values(sites)
)
    prob = GalacticOptim.OptimizationProblem(quadrupolar_opt_func, starting_values,
        [exp, N, transitions, range, I, sites], lb = quadrupolar_lb(sites),
        ub = quadrupolar_lb(sites))

    return @match method begin
        :NelderMead              => solve(prob, NelderMead())
        :KyrlovTrustRegion       => solve(prob, Optim.KrylovTrustRegion())
        :SAMIN                   => solve(prob, SAMIN())
        :ParticleSwarm           => solve(prob, ParticleSwarm())
        :GN_DIRECT               => solve(prob, Opt(:GN_DIRECT, 2))
        :GN_DIRECT_L             => solve(prob, Opt(:GN_DIRECT_L, 2))
        :GN_DIRECT_L_RAND        => solve(prob, Opt(:GN_DIRECT_L_RAND, 2))
        :GN_DIRECT_NOSCAL        => solve(prob, Opt(:GN_DIRECT_NOSCAL, 2))
        :GN_DIRECT_L_NOSCAL      => solve(prob, Opt(:GN_DIRECT_L_NOSCAL, 2))
        :GN_DIRECT_L_RAND_NOSCAL => solve(prob, Opt(:GN_DIRECT_L_RAND_NOSCAL, 2))
        :GN_DIRECT_ORIG_DIRECT   => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT, 2))
        :GN_DIRECT_ORIG_DIRECT_L => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT_L, 2))
        :GD_STOGO                => solve(prob, Opt(:GD_STOGO, 2))
        :GD_STOGO_RAND           => solve(prob, Opt(:GD_STOGO_RAND, 2))
        :GN_AGS                  => solve(prob, Opt(:GN_AGS, 2))
        :GN_ISRES                => solve(prob, Opt(:GN_ISRES, 2))
        :GN_ESCH                 => solve(prob, Opt(:GN_ESCH, 2))
        :LN_COBYLA               => solve(prob, Opt(:LN_COBYLA, 2))
        :LN_BOBYQA               => solve(prob, Opt(:LN_BOBYQA, 2))
        :LN_NEWUOA               => solve(prob, Opt(:LN_NEWUOA, 2))
        :LN_NEWUOA_BOUND         => solve(prob, Opt(:LN_NEWUOA_BOUND, 2))
        :LN_PRAXIS               => solve(prob, Opt(:LN_PRAXIS, 2))
        :LN_SBPLX                => solve(prob, Opt(:LN_SBPLX, 2))
        :CMAES                   => solve(prob, CMAES())
        :GA                      => solve(prob, GA())
        :ES                      => solve(prob, ES())
        :SeparableNES            => solve(prob, BBO(:separable_nes))
        :ExponentialNES          => solve(prob, BBO(:xnes))
        :dxNES                   => solve(prob, BBO(:dxnes))
        :AdaptiveDERand1Bin      => solve(prob, BBO(:adaptive_de_rand_1_bin))
        :AdaptiveDERand1BinRS    => solve(prob, BBO(:adaptive_de_rand_1_bin_radiuslimited))
        :DERand1Bin              => solve(prob, BBO(:de_rand_1_bin))
        :DERand1BinRS            => solve(prob, BBO(:de_rand_1_bin_radiuslimited))
        :DERand2Bin              => solve(prob, BBO(:de_rand_2_bin))
        :DERand2BinRS            => solve(prob, BBO(:de_rand_2_bin_radiuslimited))
        :GeneratingSetSearch     => solve(prob, BBO(:generating_set_search))
        :ProbabilisticDescent    => solve(prob, BBO(:probabilistic_descent))
        :RMS                     => solve(prob, BBO(:resampling_memetic_search))
        :RIMS                    => solve(prob, BBO(:resampling_inheritance_memetic_search))
        :SPSA                    => solve(prob, BBO(:simultaneous_perturbation_stochastic_approximation))
        :Random                  => solve(prob, BBO(:random_search))
        _                        => solve(prob, BBO())
    end
end
