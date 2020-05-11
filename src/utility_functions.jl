using CSV, KernelDensity, Plots

function get_experimental(filename::String, ν0_guess::Float64)
    experimental = CSV.read(filename, delim = "  ", header = false)
    experimental[!, 1] = (parse.(Float64, experimental[:, 1]) .* ν0_guess) /
                         (10^6) .+ ν0_guess
    return [reverse(experimental[:, 1]) reverse(experimental[:, 2])]
end

function generate_theoretical_spectrum(experimental, nmr_params)
    experimental_ecdf = cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    ν0 = experimental[end, 1] - riemann_sum
    I = 3
    powder_pattern = estimate_powder_pattern(nmr_params, 1_000_000, ν0, I)
    k = kde(powder_pattern)
    x = experimental[:, 1]
    ik = InterpKDE(k)
    theoretical = pdf(ik, x)
    return (mean(experimental[:, 2]) / mean(theoretical)) .* theoretical
end

function compare_theoreticals(experimental, old_nmr_params, new_nmr_params)
    plot(experimental[:, 1], experimental[:, 2], label = "experimental")
    theoretical = generate_theoretical_spectrum(experimental, old_nmr_params)
    plot!(experimental[:, 1], theoretical, width = 2, label = "old theoretical")
    theoretical = generate_theoretical_spectrum(experimental, new_nmr_params)
    plot!(
        experimental[:, 1],
        theoretical,
        width = 2,
        label = "new theoretical",
        title = "Theoretical vs. Experimental",
        xlabel = "Frequency (MHz)",
        ylabel = "Intensity",
    )
end
