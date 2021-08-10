using GalacticOptim, Optim, BlackBoxOptim, Evolutionary, NLopt, Match

spectra_opt_func(x, p) = ols_cdf(x, p[1], p[2], ν0 = p[3], N = p[4], transitions = p[5], I = p[6])

function optimize_spectra(
    exp::Array{Float64, 2}, 
    spectra::Spectra;
    method::Symbol = :BBO,
    N::Int64 = 1_000_000,
    max_func_evals = 10_000,
    lb = lower_bounds(spectra),
    ub = upper_bounds(spectra),
    I::Int64 = 3,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    exp_ecdf = get_experimental_ecdf(exp)
    ν0 =  get_ν0(exp, exp_ecdf)
    prob = GalacticOptim.OptimizationProblem(spectra_opt_func, spectra,
        [exp, exp_ecdf, ν0, N, transitions, I, sites], lb = lb, ub = ub)

    return @match method begin
        :NelderMead              => solve(prob, NelderMead(), f_calls_limit = max_func_evals)
        :KyrlovTrustRegion       => solve(prob, Optim.KrylovTrustRegion(), f_calls_limit = max_func_evals)
        :SAMIN                   => solve(prob, SAMIN(), f_calls_limit = max_func_evals)
        :ParticleSwarm           => solve(prob, ParticleSwarm(), f_calls_limit = max_func_evals)
        :GN_DIRECT               => solve(prob, Opt(:GN_DIRECT, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_L             => solve(prob, Opt(:GN_DIRECT_L, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_L_RAND        => solve(prob, Opt(:GN_DIRECT_L_RAND, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_NOSCAL        => solve(prob, Opt(:GN_DIRECT_NOSCAL, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_L_NOSCAL      => solve(prob, Opt(:GN_DIRECT_L_NOSCAL, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_L_RAND_NOSCAL => solve(prob, Opt(:GN_DIRECT_L_RAND_NOSCAL, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_ORIG_DIRECT   => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT, 5 * sites - 1), maxeval = max_func_evals)
        :GN_DIRECT_ORIG_DIRECT_L => solve(prob, Opt(:GN_DIRECT_ORIG_DIRECT_L, 5 * sites - 1), maxeval = max_func_evals)
        :GD_STOGO                => solve(prob, Opt(:GD_STOGO, 5 * sites - 1), maxeval = max_func_evals)
        :GD_STOGO_RAND           => solve(prob, Opt(:GD_STOGO_RAND, 5 * sites - 1), maxeval = max_func_evals)
        :GN_AGS                  => solve(prob, Opt(:GN_AGS, 5 * sites - 1), maxeval = max_func_evals)
        :GN_ISRES                => solve(prob, Opt(:GN_ISRES, 5 * sites - 1), maxeval = max_func_evals)
        :GN_ESCH                 => solve(prob, Opt(:GN_ESCH, 5 * sites - 1), maxeval = max_func_evals)
        :LN_COBYLA               => solve(prob, Opt(:LN_COBYLA, 5 * sites - 1), maxeval = max_func_evals)
        :LN_BOBYQA               => solve(prob, Opt(:LN_BOBYQA, 5 * sites - 1), maxeval = max_func_evals)
        :LN_NEWUOA               => solve(prob, Opt(:LN_NEWUOA, 5 * sites - 1), maxeval = max_func_evals)
        :LN_NEWUOA_BOUND         => solve(prob, Opt(:LN_NEWUOA_BOUND, 5 * sites - 1), maxeval = max_func_evals)
        :LN_PRAXIS               => solve(prob, Opt(:LN_PRAXIS, 5 * sites - 1), maxeval = max_func_evals)
        :LN_SBPLX                => solve(prob, Opt(:LN_SBPLX, 5 * sites - 1), maxeval = max_func_evals)
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
