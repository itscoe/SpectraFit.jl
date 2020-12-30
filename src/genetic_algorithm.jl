using Distributions, ProgressMeter

function cross_over(survivors)
    size = length(survivors) * 2
    population = Array{Array{Float64}}(undef, size)
    for i = 1:size
        parents = rand(DiscreteUniform(1, length(survivors)), 2)
        genes = rand(DiscreteUniform(1, 2), 4)
        population[i] = [survivors[parents[genes[1]]][1],
                         survivors[parents[genes[2]]][2],
                         survivors[parents[genes[3]]][3],
                         survivors[parents[genes[4]]][4]]
    end
    return population
end

function mutate(population;
    mutation_probability = 0.25,
    qcc_σ = 0.5,
    σqcc_σ = 0.05,
    η_σ = 0.1,
    ση_σ = 0.05
)
    for i = 1:length(population)
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][1] = max(0, rand(Normal(population[i][1], qcc_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][2] = max(0, rand(Normal(population[i][2], σqcc_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][3] = max(0, rand(Normal(population[i][3], η_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][4] = max(0, rand(Normal(population[i][4], ση_σ)))
        end
    end
    return population
end

function genetic_algorithm(N, experimental; gens = 1_000)
    mins, maxes, means = zeros(gens), zeros(gens), zeros(gens)
    population = [get_quadrupolar_starting_values.(ones(Int, N)) zeros(N)]
    best = (population[1, 1], Inf)
    experimental_ecdf = get_experimental_ecdf(experimental)
    ν0 =  get_ν0(experimental, experimental_ecdf)
    @showprogress for i = 1:gens
        population[:, 2] = ols_cdf.(Quadrupolar.(population[:, 1]),
            Ref(experimental), Ref(experimental_ecdf), ν0)
        population = sortslices(population, by=x->x[2], dims = 1, rev = false)
        mins[i], maxes[i] = minimum(population[:, 2]), maximum(population[:, 2])
        if mins[i] < best[2]
            best = population[argmin(population[:, 2]), 1], mins[i]
        end
        means[i] = mean(population[:, 2])
        population[:, 1] = cross_over(population[1:Int(N / 2), 1])
        population[:, 1] = mutate(population[:, 1])
        map(x -> x[4] = max(eps(Float64), x[4]), population[:, 1])
    end
    return best, population, mins, means, maxes
end

function cross_over_csa(survivors)
    size = length(survivors) * 2
    population = Array{Array{Float64}}(undef, size)
    for i = 1:size
        parents = rand(DiscreteUniform(1, length(survivors)), 2)
        genes = rand(DiscreteUniform(1, 2), 6)
        population[i] = [survivors[parents[genes[1]]][1],
                         survivors[parents[genes[2]]][2],
                         survivors[parents[genes[3]]][3],
                         survivors[parents[genes[4]]][4],
                         survivors[parents[genes[3]]][5],
                         survivors[parents[genes[4]]][6],
                         1.0]
    end
    return population
end

function mutate_csa(population;
    mutation_probability = 0.25,
    σᵢₛₒ_σ = 50,
    σσᵢₛₒ_σ = 10,
    Δσ_σ = 50,
    σΔσ_σ = 10,
    ησ_σ = 0.1,
    σησ_σ = 0.05
)
    for i = 1:length(population)
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][1] = max(0, rand(Normal(population[i][1], σᵢₛₒ_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][2] = max(0, rand(Normal(population[i][2], σσᵢₛₒ_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][3] = max(0, rand(Normal(population[i][3], Δσ_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][4] = max(0, rand(Normal(population[i][4], σΔσ_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][5] = max(0, rand(Normal(population[i][5], ησ_σ)))
        end
        if rand(Uniform(0, 1)) < mutation_probability
            population[i][6] = max(0, rand(Normal(population[i][6], σησ_σ)))
        end
    end
    return population
end

function genetic_algorithm_csa(N, experimental; gens = 1_000)
    mins, maxes, means = zeros(gens), zeros(gens), zeros(gens)
    population = [get_chemical_shift_starting_values.(ones(Int, N)) zeros(N)]
    best = (population[1, 1], Inf)
    experimental_ecdf = get_experimental_ecdf(experimental)
    @showprogress for i = 1:gens
        population[:, 2] = ols_cdf.(ChemicalShift.(population[:, 1]),
            Ref(experimental), Ref(experimental_ecdf))
        population = sortslices(population, by=x->x[2], dims = 1, rev = false)
        mins[i], maxes[i] = minimum(population[:, 2]), maximum(population[:, 2])
        if mins[i] < best[2]
            best = population[argmin(population[:, 2]), 1], mins[i]
        end
        means[i] = mean(population[:, 2])
        population[:, 1] = cross_over_csa(population[1:Int(N / 2), 1])
        population[:, 1] = mutate_csa(population[:, 1])
        map(x -> x[2] = max(eps(Float64), x[2]), population[:, 1])
        map(x -> x[4] = max(eps(Float64), x[4]), population[:, 1])
        map(x -> x[5] = max(eps(Float64), x[5]), population[:, 1])
        map(x -> x[6] = max(eps(Float64), x[6]), population[:, 1])
        map(x -> x[5] = min(1.0, x[5]), population[:, 1])
    end
    return best, population, mins, means, maxes
end
