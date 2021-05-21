using KernelDensity, Plots

"""
    generate_theoretical_spectrum(experimental, Quadrupolar)

Given the experimental data (experimental) and a set of corresponding
nmr parameters, constructs a powder pattern (spectrum) by interpolating the
kernel density estimate of the distribution produced by the nmr parameters

# Examples
```julia-repl
julia> generate_theoretical_spectrum(
julia>     get_experimental("B2O3FastCool.txt", 32.239),
julia>     Quadrupolar([5.5, 0.1, 0.12, 0.03, 1.0])
9992-element Array{Float64,1}:
 20431.55828260406
 20423.94683368574
 20415.98848956408
 20408.319911200455
 20400.302057265526
 20392.58886476615
 20384.5664157466
 20376.877421554564
 20369.200721475398
 20361.217235488795
 20353.565635884566
 20345.621287321344
 20338.045108945793
     ⋮
 22608.80291238082
 22607.488079524057
 22606.341148346466
 22605.26654097029
 22604.350227759416
 22603.546870237795
 22602.831135390592
 22602.261303660813
 22601.78989055206
 22601.454608723958
 22601.22751832443
 22601.126786321012
```
"""
function generate_theoretical_spectrum(
    experimental::Array{Float64,2},
    Quadrupolar::Quadrupolar;
    I::Int64 = 3,
    transitions = 1:(2*I),
)
    ν0 = get_ν0(experimental, get_experimental_ecdf(experimental))
    powder_pattern = estimate_powder_pattern(Quadrupolar, 1_000_000, ν0, I,
        transitions = transitions)
    k = kde(powder_pattern)
    x = experimental[:, 1]
    ik = InterpKDE(k)
    theoretical = pdf(ik, x)
    return (mean(experimental[:, 2]) / mean(theoretical)) .* theoretical
end

function generate_theoretical_spectrum(
    experimental::Array{Float64,2},
    ChemicalShift::ChemicalShift,
)
    powder_pattern = estimate_powder_pattern(ChemicalShift, 1_000_000)
    k = kde(powder_pattern)
    x = experimental[:, 1]
    ik = InterpKDE(k)
    theoretical = pdf(ik, x)
    return (mean(experimental[:, 2]) / mean(theoretical)) .* theoretical
end

function plot_experimental(
    experimental::Array{Float64,2};
    relative_ν = false,
    unit = "MHz",
)
    to_plot = relative_ν ?
        experimental[:, 1] .- get_ν(experimental) :
        experimental[:, 1]
    x_label = relative_ν ? "Δν ($(unit))" : "ν ($(unit))"
    plot(
        to_plot,
        experimental[:, 2],
        label = "experimental",
        xlabel = x_label,
        ylabel = "Intensity",
    )
end

function plot_theoretical(
    experimental::Array{Float64,2},
    params::Quadrupolar;
    relative_ν = false,
)
    to_plot = relative_ν ?
        experimental[:, 1] .- get_ν(experimental) :
        experimental[:, 1]
    x_label = relative_ν ? "Δν (MHz)" : "ν (MHz)"
    theoretical = generate_theoretical_spectrum(experimental, params)
    plot(
        to_plot,
        theoretical,
        width = 2,
        label = "theoretical",
        xlabel = x_label,
        ylabel = "Intensity",
    )
end

function plot_theoretical(
    experimental::Array{Float64,2},
    params::ChemicalShift;
    relative_ν = false,
    unit = "MHz",
)
    to_plot = relative_ν ?
        experimental[:, 1] .- get_ν(experimental) :
        experimental[:, 1]
    x_label = relative_ν ? "Δν ($(unit))" : "ν ($(unit))"
    theoretical = generate_theoretical_spectrum(experimental, params)
    plot(
        to_plot,
        theoretical,
        width = 2,
        label = "theoretical",
        xlabel = x_label,
        ylabel = "Intensity",
    )
end

"""
    compare_theoreticals(experimental, old_Quadrupolar, new_Quadrupolar)

Plots the experimental data along with two theoretical powder patterns, for the
sake of comparison

"""
function compare_theoreticals(
    experimental::Array{Float64,2},
    old_Quadrupolar::Quadrupolar,
    new_Quadrupolar::Quadrupolar;
    I = 3,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    plot(experimental[:, 1], experimental[:, 2], label = "experimental")
    theoretical = generate_theoretical_spectrum(experimental, old_Quadrupolar,
        transitions = transitions)
    plot!(experimental[:, 1], theoretical, width = 2, label = "old theoretical")
    theoretical = generate_theoretical_spectrum(experimental, new_Quadrupolar,
        transitions = transitions)
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

function plot_chemical_shift(
    experimental::Array{Float64,2},
    params::ChemicalShift
)
    plot(experimental[:, 1], experimental[:, 2], label = "experimental")
    theoretical = generate_theoretical_spectrum(experimental, params)
    plot!(
        experimental[:, 1],
        theoretical,
        width = 2,
        label = "theoretical",
        title = "Theoretical vs. Experimental",
        xlabel = "Frequency (ppm)",
        ylabel = "Intensity",
    )
end

function plot_chemical_shift(
    experimental::Array{Float64,2},
    params::Array{Float64}
)
    return plot_chemical_shift(experimental,
        ChemicalShift(transform_params(push!(params, 1.0), ChemicalShift)))
end
