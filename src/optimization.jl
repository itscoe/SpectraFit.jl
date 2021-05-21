using GalacticOptim, Optim, BlackBoxOptim, StatsBase, Evolutionary, NLopt, Match

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
    starting_values = get_quadrupolar_starting_values(sites),
    lb = quadrupolar_lb(sites),
    ub = quadrupolar_ub(sites),
    max_func_evals = 10_000,
)
    prob = GalacticOptim.OptimizationProblem(quadrupolar_opt_func, starting_values,
        [exp, N, transitions, range, I, sites], lb = lb, ub = ub)

    return @match method begin
        :NelderMead              => solve(prob, NelderMead(), Optim.Options(f_calls_limit = max_func_evals))
        :KyrlovTrustRegion       => solve(prob, Optim.KrylovTrustRegion(), Optim.Options(f_calls_limit = max_func_evals))
        :SAMIN                   => solve(prob, SAMIN(), Optim.Options(f_calls_limit = max_func_evals))
        :ParticleSwarm           => solve(prob, ParticleSwarm(), Optim.Options(f_calls_limit = max_func_evals))
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
        :CMAES                   => solve(prob, CMAES(), Evolutionary.Options(iters = max_func_evals ÷ 15))
        :GA                      => solve(prob, GA(), Evolutionary.Options(iters = max_func_evals ÷ 50))
        :ES                      => solve(prob, ES(), Evolutionary.Options(iters = max_func_evals))
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
