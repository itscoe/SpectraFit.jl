using BlackBoxOptim

function fit_quadrupolar_bb(
    experimental::Array{Float64, 2};
    sites::Int64 = 1,
    max_func_evals::Int64 = 10000,
    method::Symbol = :seperable_nes,
    I::Int64 = 3,
    samples::Int64 = 1_000_000,
    starting_values = get_quadrupolar_starting_values(sites),
    transitions::UnitRange{Int64} = 1:(2*I),
    range::Tuple{Float64,Float64} = (experimental[1, 1], experimental[end, 1]),
)
    experimental_ecdf = get_experimental_ecdf(experimental)
    ν0 =  get_ν0(experimental, experimental_ecdf)
    search_range = [(0.0, 9.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    fitness_progress_history = Array{Tuple{Int, Float64},1}()
    callback = oc -> push!(fitness_progress_history,
        (BlackBoxOptim.num_func_evals(oc), best_fitness(oc)))

    res =  bboptimize(x -> SpectraFit.ols_cdf(  # objective function
                Quadrupolar(x),
                experimental,
                experimental_ecdf,
                ν0,
                I = I,
                samples = 1_000_000,
                transitions = transitions,
                range = range,
            ), SearchRange = search_range, MaxFuncEvals = max_func_evals,
            CallbackFunction = callback, CallbackInterval = 0.0)

    return res, fitness_progress_history
end