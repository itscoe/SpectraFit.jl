using KernelDensity, Plots, StatsBase, StatsPlots, Distributions

function plot_parameters(exp::ExperimentalSeries, s::Spectrum, res)
    nₚ = length(res[1].P)
    nₛ = length(exp.spectra)
    ps = Array{Plots.Plot}(undef, nₚ)
    labels_s = labels(s)
    
    for i = 1:nₚ
        ps[i] = violin(
            map(x -> repeat([exp.ind_var[x]], length(res[x].P[i].particles)), 1:nₛ), 
            map(x -> res[x].P[i].particles, 1:nₛ), 
            label = "", linewidth = 0, bar_width = 1 / (nₛ + 1), 
            color = palette(:default)[1])
        scatter!(ps[i], exp.ind_var, map(x -> mean(res[x].P[i].particles), 1:nₛ), 
            label = "", grid = false, xlabel = exp.ind_var_label, ylabel = labels_s[i], 
            markersize = 3, color = palette(:default)[1], markerstrokecolor = :white)
    end

    return plot(ps..., dpi = 300, format = :png)
end

function generate_theoretical_spectrum(
    exp::ExperimentalSpectrum,
    c::Tuple{Vararg{NMRInteraction}},
)
    ν_step = (exp.ν[end] - exp.ν[1]) / length(exp.ν)
    ν_start = exp.ν[1] - ν_step / 2
    ν_stop = exp.ν[end] + ν_step / 2
    powder_pattern = filter(x -> to_ppm(ν_start, exp.ν₀) <= x <= to_ppm(ν_stop, exp.ν₀), 
        to_ppm.(estimate_powder_pattern(c, 1_000_000, exp), exp.ν₀))
    k = kde(ustrip.(powder_pattern))
    ik = InterpKDE(k)
    theoretical = pdf(ik, ustrip.(to_ppm.((exp.ν .+ ν_step / 2), exp.ν₀)))
    return (mean(exp.i) / mean(theoretical)) .* theoretical
end

function plot_fits(exp::ExperimentalSeries, s::Spectrum, res; units = u"MHz")
    nₛ = length(exp.spectra)
    unit_label = exp.ind_var_label[findfirst(
            isequal('('), exp.ind_var_label):findlast(
            isequal(')'), exp.ind_var_label)] 
    plt = plot(ylabel = "", yaxis = false, yticks = [], xlabel = "Frequency (MHz)", grid = false, 
        dpi = 300, format = :png)
    for i = 1:length(exp.spectra)
        ν_step = (exp.spectra[i].ν[end] - exp.spectra[i].ν[1]) / length(exp.spectra[i].ν)
        N, M = length(exp.spectra[i].ν), length(res[i].P[1].particles)
        th_spectra = zeros(N, M)
        for j in 1:M
            th_spectra[:, j] = generate_theoretical_spectrum(exp.spectra[i], 
                Spectrum(s, (map(x -> x.particles[j], res[i].P)...,)).components[1]
            )
        end
        th_spectra_collected = hcat(map(x -> mean(th_spectra[x, :]), 1:N), 
            map(x -> minimum(th_spectra[x, :]), 1:N), map(x -> maximum(th_spectra[x, :]), 1:N))
        
        ν = (exp.spectra[i].ν .+ ν_step / 2, exp.spectra[i].ν₀)
        ν = units == u"ppm" ? to_ppm(ν, exp.ν₀) : to_Hz(ν, exp.ν₀)

        plot!(plt, ustrip.(ν), 
                exp.spectra[i].i ./ maximum(exp.spectra[i].i) .+ i, 
                label = "", color = palette(:default)[1])
        plot!(plt, ustrip.(ν), 
                th_spectra_collected[:, 1] .* 
                mean(exp.spectra[i].i ./ maximum(exp.spectra[i].i)) ./ 
                mean(th_spectra_collected[:, 1]) .+ i,
                ribbon = (th_spectra_collected[:, 1] ./ maximum(th_spectra_collected[:, 1]) .- 
                        th_spectra_collected[:, 2] ./ maximum(th_spectra_collected[:, 1]), 
                    th_spectra_collected[:, 3] ./ maximum(th_spectra_collected[:, 1]) .- 
                        th_spectra_collected[:, 1] ./ maximum(th_spectra_collected[:, 1])),
                label = "", color = palette(:default)[2])
        annotate!(minimum(ustrip.(ν)), i + 0.4, 
            text("$(exp.ind_var[i]) $(unit_label)", :black, :left, 10))
    end
    return plt
end
