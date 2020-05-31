using CSV, KernelDensity, Plots

"""
    get_experimental(filename, ν0_guess)

Get the experimental data from the filename specified and with the guess for the
larmor frequency (ν0_guess). Returns data of type Array{Float64,2} assuming good
data in the specified file.

# Examples
```julia-repl
julia> experimental = get_experimental("B2O3FastCool.txt", 32.239)
9992×2 Array{Float64,2}:
 30.99     8158.06
 30.9902   7660.22
 30.9905   7280.4
 30.9907   7220.96
 30.991    7524.46
 30.9912   8042.03
 30.9915   8545.74
 30.9917   8843.87
 30.992    8826.42
 30.9922   8468.01
 30.9924   7816.08
 30.9927   6973.43
 30.9929   6075.45
  ⋮
 33.4273  16466.1
 33.4276  15877.6
 33.4278  15079.3
 33.4281  14120.7
 33.4283  13077.5
 33.4285  12044.6
 33.4288  11128.3
 33.429   10433.0
 33.4293  10042.0
 33.4295   9993.03
 33.4298  10261.3
 33.43    10764.6
```
"""
function get_experimental(
    filename::String,
    ν0_guess::Float64;
    header::Bool = false,
    delim = ",",
    convert_ppm_to_mhz::Bool = true,
    reverse_data::Bool = true,
)
    if delim == ","
        experimental = CSV.read(filename, header = header)
    else
        experimental = CSV.read(filename, header = header, delim = delim)
    end
    if typeof(experimental[1, 1]) != Float64
        experimental[!, 1] = parse.(Float64, experimental[:, 1])
    end
    if convert_ppm_to_mhz
        experimental[!, 1] = experimental[:, 1] .* (ν0_guess /
            (10^6)) .+ ν0_guess
    end
    if reverse_data
        experimental = [reverse(experimental[:, 1]) reverse(experimental[:, 2])]
    else
        experimental = [experimental[:, 1] experimental[:, 2]]
    end
    return experimental
end

"""
    generate_theoretical_spectrum(experimental, nmr_params)

Given the experimental data (experimental) and a set of corresponding
nmr parameters, constructs a powder pattern (spectrum) by interpolating the
kernel density estimate of the distribution produced by the nmr parameters

# Examples
```julia-repl
julia> generate_theoretical_spectrum(
julia>     get_experimental("B2O3FastCool.txt", 32.239),
julia>     nmr_params([5.5, 0.1, 0.12, 0.03, 1.0])
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
    nmr_params::nmr_params;
    I::Int64 = 3,
)
    ν0 = get_ν0(experimental, get_experimental_ecdf(experimental))
    powder_pattern = estimate_powder_pattern(nmr_params, 1_000_000, ν0, I)
    k = kde(powder_pattern)
    x = experimental[:, 1]
    ik = InterpKDE(k)
    theoretical = pdf(ik, x)
    return (mean(experimental[:, 2]) / mean(theoretical)) .* theoretical
end

function plot_experimental(experimental::Array{Float64,2}; relative_ν = false)
    ENV["GKS_ENCODING"] = "utf-8"
    to_plot = relative_ν ?
        experimental[:, 1] .- get_ν(experimental) :
        experimental[:, 1]
    x_label = relative_ν ? "Δν (MHz)" : "ν (MHz)"
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
    params::nmr_params;
    relative_ν = false,
)
    ENV["GKS_ENCODING"] = "utf-8"
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

"""
    compare_theoreticals(experimental, old_nmr_params, new_nmr_params)

Plots the experimental data along with two theoretical powder patterns, for the
sake of comparison

"""
function compare_theoreticals(
    experimental::Array{Float64,2},
    old_nmr_params::nmr_params,
    new_nmr_params::nmr_params,
)
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

"""
    get_experimental_ecdf(experimental)

Returns the ecdf of the experimental data over the range of the experimental
data

"""
function get_experimental_ecdf(experimental::Array{Float64,2})
    return cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
end

"""
    get_ν0(experimental, experimental_ecdf)

Calculates the Larmor frequency by finding the approximate mean of the
distribution. Should be passed into the forward model to get theoretical data
that lines up with the experimental

"""
function get_ν0(experimental::Array{Float64,2}, experimental_ecdf)
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    return experimental[end, 1] - riemann_sum
end

function get_ν0(experimental::Array{Float64,2})
    return get_ν0(experimental, get_experimental_ecdf(experimental))
end

function get_data(filename::String)
    return joinpath(dirname(pathof(SpectraFit)), "..", "data", filename)
end
