using Unitful

get_example_data(filename::String) = 
    joinpath(dirname(pathof(SpectraFit)), "..", "data", filename)

"""
    get_experimental_ecdf(experimental)

Returns the ecdf of the experimental data over the range of the experimental
data

"""
get_exp_ecdf(intensity::Vector{Float64}) = cumsum(intensity) ./ sum(intensity)


struct ExperimentalSpectrum{U}
    isotope::Isotope
    B::typeof(1.0u"T")
    ν₀::typeof(1.0u"MHz")
    ν::Vector{U}
    i::Vector{Float64}
    ecdf::Vector{Float64}
end

function ExperimentalSpectrum(
    filename::String;
    freq_unit::Unitful.Unitlike = u"ppm",
    isotope::Isotope = Isotope(elements["Boron"], 10),
    B::typeof(1.0u"T") = 7.0u"T",
    header::Int64 = 0,
    delim::String = ",",
    range::Tuple{Quantity, Quantity} = (Quantity(-Inf, freq_unit), 
        Quantity(Inf, freq_unit))
)
    ν₀ = γ(isotope) * B
    data = map(x -> parse.(Float64, x), 
        split.(readlines(filename)[(header + 1):end], delim))
    data = sortslices(hcat(map(x -> x[1], data), map(x -> x[2], data)), 
        dims = 1)
    start_i = findfirst(x -> 
        to_Hz(Quantity(x, freq_unit), ν₀) > to_Hz(range[1], ν₀), data[:, 1])
    stop_i = findlast(x -> 
        to_Hz(Quantity(x, freq_unit), ν₀) < to_Hz(range[2], ν₀), data[:, 1])
    return ExperimentalSpectrum(
        isotope, 
        B, 
        ν₀, 
        Quantity.(data[start_i:stop_i, 1], freq_unit), 
        data[start_i:stop_i, 2], 
        get_exp_ecdf(data[start_i:stop_i, 2]),
    )
end
