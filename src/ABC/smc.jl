using Unitful, StatsBase, KissABC

function abc_smc(
    s₀::Spectra{N, I}, 
    exp::ExperimentalSpectra{Quantity{Float64, NoDims, Unitful.FreeUnits{(ppm,), NoDims, nothing}}}; 
    parallel = false
) where {N, I}
    function wasserstein(p)
        ν_step = exp.ν[end] - exp.ν[1] / length(exp.ν)
        ν_start = exp.ν[1] - ν_step / 2
        ν_stop = exp.ν[end] + ν_step / 2

        s = Spectra(s₀, p)
        th_cdf = zeros(length(exp.ν))

        for i = 1:N
            powder_pattern = filter(x -> ν_stop <= x <= ν_start, 
                to_ppm.(estimate_powder_pattern(s.components[c], 
                1_000_000), exp.ν₀))
            isempty(powder_pattern) && return 1.0
            th_cdf .+= s.weights[i] .* ((exp.ν .+ (ν_step / 2)) .|> Float64 
                .|> ecdf(powder_pattern .|> Float64))
        end

        return mean(abs.(th_cdf .- exp.ecdf))
    end

    return smc(prior, wasserstein, parallel = parallel)
end