using Distributions, KissABC, Unitful, ProgressMeter



"""
    get_wasserstein(s₀, exp)

Higher order function that creates wasserstein distance function for the form of the spectrum s₀ and the 
ExperimentalSpectrum exp

"""
function get_wasserstein(
    s₀::Spectrum{N, M, C}, 
    exp::ExperimentalSpectrum{N2}; 
    exp_type::Symbol = :static, 
    n::Int64 = 1_000_000
) where {N, M, C, N2}
    μs = μ(n)
    λs = λ(n)
    I₀ = I(exp.isotope)
    ν₀ = exp.ν₀
    ν_start = exp.ν_start
    ν_step = exp.ν_step

    ms = I₀.d == 0x0000 ? get_m.(rand(1:m_sum(I₀), n), I₀) : 
        FPOT(0, 0x0000) * ones(Int64, n)
    vQ_c = ((1.5 * e * 0.0845e-28u"m^2" / h) / Float64(I₀ * (2 * I₀ - 1))) |> 
        u"T^-1"
    U0_rand = randn(n)
    U1_rand = randn(n)
    U5_rand = randn(n)
    exp_ecdf = exp.ecdf

    function get_ecdf(x::Array{Int64})
        # modified from StatsBase.jl add_counts
        r = zeros(Int64, N2)
        @simd for i in 1:length(x)
            @inbounds r[x[i]] += 1
        end
        return cumsum(r) ./ length(x)
    end

    if N == 1
        return @inline function (p::NTuple{Nₚ, Float64}) where {Nₚ}
            s = Spectrum(s₀, p)
            powder_pattern = filter(x -> 1 <= x <= N2, 
                ceil.(Int64, estimate_static_powder_pattern(s.components[1], n, 
                μs, λs, ms, U0_rand, U1_rand, U5_rand, I₀, ν₀, ν_step, vQ_c, 
                ν_start, exp_type)))
            isempty(powder_pattern) && return 1.0
            th_cdf = get_ecdf(powder_pattern)
            return sum(abs.(th_cdf .- exp_ecdf)) / N2
        end
    else
        return @inline function (p::NTuple{Nₚ, Float64}) where {Nₚ}
            s = Spectrum(s₀, p)
            weights_sum = sum(s.weights)
            weights_sum > 1.0 && return 1.0
            th_cdf = zeros(length(exp.ν))
            for c = 1:N
                weight = c == N ? 1. - weights_sum : s.weights[c]
                powder_pattern = filter(x -> 1 <= x <= N2, 
                    ceil.(Int64, estimate_powder_pattern(s.components[c], n, μs,
                    λs, ms, U0_rand, U1_rand, U5_rand, I₀, ν₀, ν_step, vQ_c, 
                    ν_start, exp_type)))
                isempty(powder_pattern) && return 1.0
                th_cdf .+= weight .* get_ecdf(powder_pattern)
            end
            return  sum(abs.(th_cdf .- exp_ecdf)) / N2
        end
    end
end 

"""
    abc_smc(s₀, exp)

Run the approximate Bayesian computations through sequential Monte Carlo given the experimental data and the spectrum of 
the functional form for the model selected

"""
@inline abc_smc(
    s₀::Spectrum, 
    exp::ExperimentalSpectrum; 
    exp_type::Symbol = :static,
    prior::KissABC.Factored = prior(s₀),
    n::Int64 = 1_000_000,
    cost::Function = get_wasserstein(s₀, exp, exp_type = exp_type, n = n),
    parallel::Bool = false,
    nparticles::Int64 = 100,
    M::Int64 = 1,
    alpha::Float64 = 0.95,
    mcmc_retrys::Int64 = 0,
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

Run the approximate Bayesian computations through sequential Monte Carlo given the experimental data (in a series) and 
the spectrum of the functional form for the model selected

"""
@inline abc_smc(
    s₀::Spectrum, 
    exp::ExperimentalSeries; 
    exp_type::Symbol = :static,
    n::Int64 = 1_000_000,
    prior::KissABC.Factored = prior(s₀),
    parallel::Bool = false,
    nparticles::Int64 = 100,
    M::Int64 = 1,
    alpha::Float64 = 0.95,
    mcmc_retrys::Int64 = 0,
    mcmc_tol::Float64 = 0.015,
    epstol::Float64 = 0.0,
    r_epstol::Float64 = (1 - alpha)^1.5 / 50,
    min_r_ess::Float64 = alpha^2,
    max_stretch::Float64 = 2.0,
) = @showprogress map(x -> abc_smc(
    s₀, 
    x, 
    prior = prior,
    exp_type = exp_type,
    n = n,
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

