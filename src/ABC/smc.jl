using Distributions, KissABC, StatsBase, Unitful, ProgressMeter

"""
    prior(s)

Gets the prior distribution of a spectrum in the form of s

"""
function prior(s::Spectrum{N, M, C}) where {N, M, C}
    dists = Array{Distribution}(undef, length(s))
    p = 1
    for i = 1:N, interaction in s.components[i], j = 1:length(interaction)
        dists[p] = prior(interaction, j)
        p += 1
    end
    for _ = 1:N-1
        dists[p] = Uniform(0, 1)
        p += 1
    end
    return Factored(dists...)
end

"""
    ecdf(X, exp)

Higher order function that creates the ecdf function from a sample 
constructed and the ExperimentalSpectrum. This is a simple extension 
of the StatsBase.ecdf function. 

"""
function ecdf(X::Vector{Quantity{Float64, Y1, Z1}}, 
  exp::ExperimentalSpectrum) where {Y1, Z1}
    function ef(v::Vector{Quantity{Float64, Y2, Z2}}) where {Y2, Z2}
        ef_func = StatsBase.ecdf(ustrip.(to_Hz.(X, exp.ν₀)))
        return ef_func(ustrip.(to_Hz.(v, exp.ν₀)))
    end
    return ef
end

"""
    get_wasserstein(s₀, exp)

Higher order function that creates wasserstein distance function 
for the form of the spectrum s₀ and the ExperimentalSpectrum exp

"""
function get_wasserstein(s₀::Spectrum{N, M, C}, 
  exp::ExperimentalSpectrum; type::Symbol = :static) where {N, M, C}
    function wasserstein(p::NTuple{Nₚ, Float64}) where {Nₚ}
        ν_step = (exp.ν[end] - exp.ν[1]) / length(exp.ν)
        ν_start = exp.ν[1] - ν_step / 2
        ν_stop = exp.ν[end] + ν_step / 2

        s = Spectrum(s₀, p)
        weights_sum = N == 1 ? 0. : sum(s.weights)
        weights_sum > 1.0 && return 1.0

        th_cdf = zeros(length(exp.ν))
        for c = 1:N
            weight = c == N ? 1. - weights_sum : s.weights[c]
            powder_pattern = filter(
                x -> to_Hz(ν_start, exp.ν₀) <= x <= to_Hz(ν_stop, exp.ν₀), 
                type == :static ? 
                    estimate_static_powder_pattern(
                        s.components[c], 1_000_000, exp) : 
                    estimate_mas_powder_pattern(
                        s.components[c], 1_000_000, exp)
            )
            isempty(powder_pattern) && return 1.0
            th_cdf .+= weight .* ecdf(powder_pattern, exp)(exp.ν .+ ν_step / 2)
        end

        return mean(abs.(th_cdf .- exp.ecdf))
    end

    return wasserstein
end 

"""
    abc_smc(s₀, exp)

Run the approximate Bayesian computations through sequential 
Monte Carlo given the experimental data and the spectrum of 
the functional form for the model selected

"""
abc_smc(
    s₀::Spectrum, 
    exp::ExperimentalSpectrum; 
    type::Symbol = :static,
    prior::KissABC.Factored = prior(s₀),
    cost::Function = get_wasserstein(s₀, exp, type = type),
    parallel::Bool = false,
    nparticles::Int = 100,
    M::Int = 1,
    alpha::Float64 = 0.95,
    mcmc_retrys::Int = 0,
    mcmc_tol::Float64 = 0.015,
    epstol::Float64 = 0.0,
    r_epstol::Float64 = (1 - alpha)^1.5 / 50,
    min_r_ess::Float64 = alpha^2,
    max_stretch::Float64 = 2.0,
) = smc(
    prior, 
    cost, 
    parallel = parallel,
    nparticles = nparticles,
    M = M,
    alpha = alpha,
    mcmc_retrys = mcmc_retrys,
    mcmc_tol = mcmc_tol,
    epstol = epstol,
    r_epstol = r_epstol,
    min_r_ess = min_r_ess,
    max_stretch = max_stretch
)

"""
    abc_smc(s₀, exp)

Run the approximate Bayesian computations through sequential 
Monte Carlo given the experimental data (in a series) and the 
spectrum of the functional form for the model selected

"""
abc_smc(
    s₀::Spectrum, 
    exp::ExperimentalSeries; 
    type::Symbol = :static,
    prior::KissABC.Factored = prior(s₀),
    parallel::Bool = false,
    nparticles::Int = 100,
    M::Int = 1,
    alpha::Float64 = 0.95,
    mcmc_retrys::Int = 0,
    mcmc_tol::Float64 = 0.015,
    epstol::Float64 = 0.0,
    r_epstol::Float64 = (1 - alpha)^1.5 / 50,
    min_r_ess::Float64 = alpha^2,
    max_stretch::Float64 = 2.0,
) = @showprogress map(x -> abc_smc(
    s₀, 
    x, 
    prior = prior,
    type = type,
    cost = get_wasserstein(s₀, x, type = type),
    parallel = parallel,
    nparticles = nparticles,
    M = M,
    alpha = alpha,
    mcmc_retrys = mcmc_retrys,
    mcmc_tol = mcmc_tol,
    epstol = epstol,
    r_epstol = r_epstol,
    min_r_ess = min_r_ess,
    max_stretch = max_stretch,
), exp.spectra)

