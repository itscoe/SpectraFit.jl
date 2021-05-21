using GalacticOptim, Optim, BlackBoxOptim, Evolutionary, NLopt, Match

function quadrupolar_opt_func(x, p)
    if p[5] == 1
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
            I = p[4],
        )
    else
        parameters = map(i -> Quadrupolar(abs(x[5*(i-1)+1]), abs(x[5*(i-1)+2]), abs(x[5*(i-1)+3]), abs(x[5*(i-1)+4])), 1:p[5])
        if p[5] == 2
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
            I = p[4],
        )
    end
end

function quadrupolar_opt(
    exp::Array{Float64, 2};
    method::Symbol = :BBO,
    I::Int64 = 3,
    N::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
    sites::Int64 = 1,
    starting_values = quadrupolar_starting_value(sites),
    lb = quadrupolar_lb(sites),
    ub = quadrupolar_ub(sites),
    max_func_evals = 10_000,
)
    prob = GalacticOptim.OptimizationProblem(quadrupolar_opt_func, starting_values,
        [exp, N, transitions, I, sites], lb = lb, ub = ub)

    return @match method begin
        :NelderMead              => solve(prob, NelderMead(), f_calls_limit = max_func_evals)
        :KyrlovTrustRegion       => solve(prob, Optim.KrylovTrustRegion(), f_calls_limit = max_func_evals)
        :SAMIN                   => solve(prob, SAMIN(), f_calls_limit = max_func_evals)
        :ParticleSwarm           => solve(prob, ParticleSwarm(), f_calls_limit = max_func_evals)
        :GN_DIRECT               => solve(prob, Opt(:GN_DIRECT, 2), maxeval = max_func_evals)
        :GN_DIRECT_L             => solve(prob, Opt(:GN_DIRECT_L, 2), maxeval = max_func_evals)
        :GN_DIRECT_L_RAND        => solve(prob, Opt(:GN_DIRECT_L_RAND, 2), maxeval = max_func_evals)
        :GN_DIRECT_NOSCAL        => solve(prob, Opt(:GN_DIRECT_NOSCAL, 2), maxeval = max_func_evals)
        :GN_DIRECT_L_NOSCAL      => solve(prob, Opt(:GN_DIRECT_L_NOSCAL, 2), maxeval = max_func_evals)
        :GN_DIRECT_L_RAND_NOSCAL => solve(prob, Opt(:GN_DIRECT_L_RAND_NOSCAL, 2), maxeval = max_func_evals)
        :GN_DIRECT_ORIG_DIRECT   => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT, 2), maxeval = max_func_evals)
        :GN_DIRECT_ORIG_DIRECT_L => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT_L, 2), maxeval = max_func_evals)
        :GD_STOGO                => solve(prob, Opt(:GD_STOGO, 2), maxeval = max_func_evals)
        :GD_STOGO_RAND           => solve(prob, Opt(:GD_STOGO_RAND, 2), maxeval = max_func_evals)
        :GN_AGS                  => solve(prob, Opt(:GN_AGS, 2), maxeval = max_func_evals)
        :GN_ISRES                => solve(prob, Opt(:GN_ISRES, 2), maxeval = max_func_evals)
        :GN_ESCH                 => solve(prob, Opt(:GN_ESCH, 2), maxeval = max_func_evals)
        :LN_COBYLA               => solve(prob, Opt(:LN_COBYLA, 2), maxeval = max_func_evals)
        :LN_BOBYQA               => solve(prob, Opt(:LN_BOBYQA, 2), maxeval = max_func_evals)
        :LN_NEWUOA               => solve(prob, Opt(:LN_NEWUOA, 2), maxeval = max_func_evals)
        :LN_NEWUOA_BOUND         => solve(prob, Opt(:LN_NEWUOA_BOUND, 2), maxeval = max_func_evals)
        :LN_PRAXIS               => solve(prob, Opt(:LN_PRAXIS, 2), maxeval = max_func_evals)
        :LN_SBPLX                => solve(prob, Opt(:LN_SBPLX, 2), maxeval = max_func_evals)
        :CMAES                   => solve(prob, CMAES(), iterations = max_func_evals ÷ 15)
        :GA                      => solve(prob, GA(), iterations = max_func_evals ÷ 50)
        :ES                      => solve(prob, ES(), iterations = max_func_evals)
        :SeparableNES            => solve(prob, BBO(:separable_nes), MaxFuncEvals = max_func_evals)
        :ExponentialNES          => solve(prob, BBO(:xnes), MaxFuncEvals = max_func_evals)
        :dxNES                   => solve(prob, BBO(:dxnes), MaxFuncEvals = max_func_evals)
        :AdaptiveDERand1Bin      => solve(prob, BBO(:adaptive_de_rand_1_bin), MaxFuncEvals = max_func_evals)
        :AdaptiveDERand1BinRS    => solve(prob, BBO(:adaptive_de_rand_1_bin_radiuslimited), MaxFuncEvals = max_func_evals)
        :DERand1Bin              => solve(prob, BBO(:de_rand_1_bin), MaxFuncEvals = max_func_evals)
        :DERand1BinRS            => solve(prob, BBO(:de_rand_1_bin_radiuslimited), MaxFuncEvals = max_func_evals)
        :DERand2Bin              => solve(prob, BBO(:de_rand_2_bin), MaxFuncEvals = max_func_evals)
        :DERand2BinRS            => solve(prob, BBO(:de_rand_2_bin_radiuslimited), MaxFuncEvals = max_func_evals)
        :GeneratingSetSearch     => solve(prob, BBO(:generating_set_search), MaxFuncEvals = max_func_evals)
        :ProbabilisticDescent    => solve(prob, BBO(:probabilistic_descent), MaxFuncEvals = max_func_evals)
        :RMS                     => solve(prob, BBO(:resampling_memetic_search), MaxFuncEvals = max_func_evals)
        :RIMS                    => solve(prob, BBO(:resampling_inheritance_memetic_search), MaxFuncEvals = max_func_evals)
        :SPSA                    => solve(prob, BBO(:simultaneous_perturbation_stochastic_approximation), MaxFuncEvals = max_func_evals)
        :Random                  => solve(prob, BBO(:random_search), MaxFuncEvals = max_func_evals)
        _                        => solve(prob, BBO(:adaptive_de_rand_1_bin_radiuslimited), MaxFuncEvals = max_func_evals)
    end
end

function chemical_shift_opt_func(x, p)
    if p[3] == 1
        parameters = ChemicalShift(x[1], abs(x[2]), x[3], abs(x[4]), abs(x[5]), abs(x[6]))
        experimental = p[1]
        experimental_ecdf = get_experimental_ecdf(experimental)

        return ols_cdf(
            parameters,
            experimental,
            experimental_ecdf,
            ν0,
            N = p[2],
        )
    else
        parameters = map(i -> ChemicalShift(x[7*(i-1)+1], abs(x[7*(i-1)+2]),
            x[7*(i-1)+3], abs(x[7*(i-1)+4]), abs(x[7*(i-1)+5]),
            abs(x[7*(i-1)+6])), 1:p[3])
        if p[3] == 2
            weights = [x[7]]
        else
            weights = x[7:7:end]
        end
        if sum(weights) > 1
            weights ./= sum(weights)
        end
        experimental = p[1]
        experimental_ecdf = get_experimental_ecdf(experimental)
        return ols_cdf(
            parameters,
            weights,
            experimental,
            experimental_ecdf,
            N = p[2],
        )
    end
end

function chemical_shift_opt(
    exp::Array{Float64, 2};
    method::Symbol = :BBO,
    N::Int64 = 1_000_000,
    sites::Int64 = 1,
    starting_values = CSA_starting_value(sites),
    lb = CSA_lb(sites),
    ub = CSA_ub(sites),
    max_func_evals = 10_000,
)
    prob = GalacticOptim.OptimizationProblem(chemical_shift_opt_func, starting_values,
        [exp, N, sites], lb = lb, ub = ub)

    return @match method begin
        :NelderMead              => solve(prob, NelderMead(), f_calls_limit = max_func_evals)
        :KyrlovTrustRegion       => solve(prob, Optim.KrylovTrustRegion(), f_calls_limit = max_func_evals)
        :SAMIN                   => solve(prob, SAMIN(), f_calls_limit = max_func_evals)
        :ParticleSwarm           => solve(prob, ParticleSwarm(), f_calls_limit = max_func_evals)
        :GN_DIRECT               => solve(prob, Opt(:GN_DIRECT, 2), maxeval = max_func_evals)
        :GN_DIRECT_L             => solve(prob, Opt(:GN_DIRECT_L, 2), maxeval = max_func_evals)
        :GN_DIRECT_L_RAND        => solve(prob, Opt(:GN_DIRECT_L_RAND, 2), maxeval = max_func_evals)
        :GN_DIRECT_NOSCAL        => solve(prob, Opt(:GN_DIRECT_NOSCAL, 2), maxeval = max_func_evals)
        :GN_DIRECT_L_NOSCAL      => solve(prob, Opt(:GN_DIRECT_L_NOSCAL, 2), maxeval = max_func_evals)
        :GN_DIRECT_L_RAND_NOSCAL => solve(prob, Opt(:GN_DIRECT_L_RAND_NOSCAL, 2), maxeval = max_func_evals)
        :GN_DIRECT_ORIG_DIRECT   => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT, 2), maxeval = max_func_evals)
        :GN_DIRECT_ORIG_DIRECT_L => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT_L, 2), maxeval = max_func_evals)
        :GD_STOGO                => solve(prob, Opt(:GD_STOGO, 2), maxeval = max_func_evals)
        :GD_STOGO_RAND           => solve(prob, Opt(:GD_STOGO_RAND, 2), maxeval = max_func_evals)
        :GN_AGS                  => solve(prob, Opt(:GN_AGS, 2), maxeval = max_func_evals)
        :GN_ISRES                => solve(prob, Opt(:GN_ISRES, 2), maxeval = max_func_evals)
        :GN_ESCH                 => solve(prob, Opt(:GN_ESCH, 2), maxeval = max_func_evals)
        :LN_COBYLA               => solve(prob, Opt(:LN_COBYLA, 2), maxeval = max_func_evals)
        :LN_BOBYQA               => solve(prob, Opt(:LN_BOBYQA, 2), maxeval = max_func_evals)
        :LN_NEWUOA               => solve(prob, Opt(:LN_NEWUOA, 2), maxeval = max_func_evals)
        :LN_NEWUOA_BOUND         => solve(prob, Opt(:LN_NEWUOA_BOUND, 2), maxeval = max_func_evals)
        :LN_PRAXIS               => solve(prob, Opt(:LN_PRAXIS, 2), maxeval = max_func_evals)
        :LN_SBPLX                => solve(prob, Opt(:LN_SBPLX, 2), maxeval = max_func_evals)
        :CMAES                   => solve(prob, CMAES(), iterations = max_func_evals ÷ 15)
        :GA                      => solve(prob, GA(), iterations = max_func_evals ÷ 50)
        :ES                      => solve(prob, ES(), iterations = max_func_evals)
        :SeparableNES            => solve(prob, BBO(:separable_nes), MaxFuncEvals = max_func_evals)
        :ExponentialNES          => solve(prob, BBO(:xnes), MaxFuncEvals = max_func_evals)
        :dxNES                   => solve(prob, BBO(:dxnes), MaxFuncEvals = max_func_evals)
        :AdaptiveDERand1Bin      => solve(prob, BBO(:adaptive_de_rand_1_bin), MaxFuncEvals = max_func_evals)
        :AdaptiveDERand1BinRS    => solve(prob, BBO(:adaptive_de_rand_1_bin_radiuslimited), MaxFuncEvals = max_func_evals)
        :DERand1Bin              => solve(prob, BBO(:de_rand_1_bin), MaxFuncEvals = max_func_evals)
        :DERand1BinRS            => solve(prob, BBO(:de_rand_1_bin_radiuslimited), MaxFuncEvals = max_func_evals)
        :DERand2Bin              => solve(prob, BBO(:de_rand_2_bin), MaxFuncEvals = max_func_evals)
        :DERand2BinRS            => solve(prob, BBO(:de_rand_2_bin_radiuslimited), MaxFuncEvals = max_func_evals)
        :GeneratingSetSearch     => solve(prob, BBO(:generating_set_search), MaxFuncEvals = max_func_evals)
        :ProbabilisticDescent    => solve(prob, BBO(:probabilistic_descent), MaxFuncEvals = max_func_evals)
        :RMS                     => solve(prob, BBO(:resampling_memetic_search), MaxFuncEvals = max_func_evals)
        :RIMS                    => solve(prob, BBO(:resampling_inheritance_memetic_search), MaxFuncEvals = max_func_evals)
        :SPSA                    => solve(prob, BBO(:simultaneous_perturbation_stochastic_approximation), MaxFuncEvals = max_func_evals)
        :Random                  => solve(prob, BBO(:random_search), MaxFuncEvals = max_func_evals)
        _                        => solve(prob, BBO(:adaptive_de_rand_1_bin_radiuslimited), MaxFuncEvals = max_func_evals)
    end
end
