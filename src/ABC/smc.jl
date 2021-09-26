using Distributions, KissABC, StatsBase, Unitful

function prior(s::Spectra{N, M, C}) where {N, M, C}
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

function ecdf(X::Vector{Quantity{Float64, Y1, Z1}}, exp::ExperimentalSpectra) where {Y1, Z1}
    function ef(v::Vector{Quantity{Float64, Y2, Z2}}) where {Y2, Z2}
        ef_func = StatsBase.ecdf(ustrip.(to_ppm.(X, exp.ν₀)))
        return ef_func(ustrip.(to_ppm.(v, exp.ν₀)))
    end
    return ef
end

function get_wasserstein(s₀::Spectra{N, M, C}, 
  exp::ExperimentalSpectra) where {N, M, C}
    function wasserstein(p::NTuple{Nₚ, Float64}) where {Nₚ}
        ν_step = (exp.ν[end] - exp.ν[1]) / length(exp.ν)
        ν_start = exp.ν[1] - ν_step / 2
        ν_stop = exp.ν[end] + ν_step / 2

        s = Spectra(s₀, p)
        weights_sum = N == 1 ? 0. : sum(s.weights)
        weights_sum > 1.0 && return 1.0

        th_cdf = zeros(length(exp.ν))
        for c = 1:N
            weight = c == N ? 1. - weights_sum : s.weights[c]
            powder_pattern = filter(
                x -> to_ppm(ν_start, exp.ν₀) <= x <= to_ppm(ν_stop, exp.ν₀), 
                to_ppm.(estimate_powder_pattern(
                    s.components[c], 1_000_000, exp), exp.ν₀)
            )
            isempty(powder_pattern) && return 1.0
            th_cdf .+= weight .* ecdf(powder_pattern, exp)(exp.ν .+ ν_step / 2)
        end

        return mean(abs.(th_cdf .- exp.ecdf))
    end

    return wasserstein
end 

abc_smc(s₀::Spectra, exp::ExperimentalSpectra; parallel::Bool = false) = 
    smc(prior(s₀), get_wasserstein(s₀, exp), parallel = parallel)