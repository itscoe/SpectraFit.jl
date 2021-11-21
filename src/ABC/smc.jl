using Distributions, KissABC, StatsBase, Unitful, ProgressMeter

"""
    ecdf(X, exp)

Higher order function that creates the ecdf function from a sample 
constructed and the ExperimentalSpectrum. This is a simple extension 
of the StatsBase.ecdf function. 

"""
function get_ecdf(ν::Vector{Float64})
    m = length(ν)
    function ecdf(X::Vector{Float64})
        # modified from StatsBase.jl
        sort!(X)
        weightsum = length(X)
        r = zeros(m)
        i = 1
        for (j, x) in enumerate(X)
            while i <= m && x > ν[i]
                r[i] = j - 1
                i += 1
            end
            i > m && break
        end
        while i <= m
            r[i] = weightsum
            i += 1
        end
        return r / weightsum
    end
    return ecdf
end

"""
    get_wasserstein(s₀, exp)

Higher order function that creates wasserstein distance function 
for the form of the spectrum s₀ and the ExperimentalSpectrum exp

"""
function get_wasserstein(
    s₀::Spectrum{N, M, C}, 
    exp::ExperimentalSpectrum; 
    type::Symbol = :static
) where {N, M, C}
    n = 1_000_000
    μs = μ(n)
    λs = λ(n)
    ν_step = (exp.ν[end] - exp.ν[1]) / length(exp.ν)
    ν_start = to_Hz(exp.ν[1] - ν_step / 2, exp.ν₀)
    ν_stop = to_Hz(exp.ν[end] + ν_step / 2, exp.ν₀)
    ecdf = get_ecdf(ustrip.(to_Hz.(exp.ν .+ ν_step / 2, exp.ν₀)))

    if type == :static
        if N == 1
            function wasserstein(p::NTuple{Nₚ, Float64}) where {Nₚ}
                s = Spectrum(s₀, p)
                powder_pattern = filter(
                    x -> ν_start <= x <= ν_stop, 
                    estimate_static_powder_pattern(
                        s.components[c], n, μs, λs, exp))
                isempty(powder_pattern) && return 1.0
                th_cdf = ecdf(ustrip.(to_Hz.(powder_pattern, exp.ν₀)))
                return mean(abs.(th_cdf .- exp.ecdf))
            end
        else
            function wasserstein(p::NTuple{Nₚ, Float64}) where {Nₚ}
                s = Spectrum(s₀, p)
                weights_sum = sum(s.weights)
                weights_sum > 1.0 && return 1.0
        
                th_cdf = zeros(length(exp.ν))
                for c = 1:N
                    weight = c == N ? 1. - weights_sum : s.weights[c]
                    powder_pattern = filter(
                        x -> ν_start <= x <= ν_stop, 
                            estimate_static_powder_pattern(
                                s.components[c], n, μs, λs, exp))
                    isempty(powder_pattern) && return 1.0
                    th_cdf .+= weight .* 
                        ecdf(ustrip.(to_Hz.(powder_pattern, exp.ν₀)))
                end
        
                return mean(abs.(th_cdf .- exp.ecdf))
            end
        end
    else
        if N == 1 
            function wasserstein(p::NTuple{Nₚ, Float64}) where {Nₚ}
                s = Spectrum(s₀, p)
                powder_pattern = filter(
                    x -> ν_start <= x <= ν_stop, 
                    estimate_mas_powder_pattern(
                        s.components[c], n, μs, λs, exp))
                isempty(powder_pattern) && return 1.0
                th_cdf = ecdf(ustrip.(to_Hz.(powder_pattern, exp.ν₀)))
                return mean(abs.(th_cdf .- exp.ecdf))
            end
        else
            function wasserstein(p::NTuple{Nₚ, Float64}) where {Nₚ}
                s = Spectrum(s₀, p)
                weights_sum = sum(s.weights)
                weights_sum > 1.0 && return 1.0
        
                th_cdf = zeros(length(exp.ν))
                for c = 1:N
                    weight = c == N ? 1. - weights_sum : s.weights[c]
                    powder_pattern = filter(
                        x -> ν_start <= x <= ν_stop, 
                        type == :static ? 
                            estimate_static_powder_pattern(
                                s.components[c], n, μs, λs, exp) : 
                            estimate_mas_powder_pattern(
                                s.components[c], n, μs, λs, exp)
                    )
                    isempty(powder_pattern) && return 1.0
                    th_cdf .+= weight .* 
                        ecdf(ustrip.(to_Hz.(powder_pattern, exp.ν₀)))
                end
        
                return mean(abs.(th_cdf .- exp.ecdf))
            end
        end
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

