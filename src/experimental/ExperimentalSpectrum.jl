using Unitful

"""
    get_example_data(filename)

Gets the filepath on the computer calling it to access the example data 
from the paper

"""
get_example_data(filename::String) = 
    joinpath(dirname(pathof(SpectraFit)), "..", "data", filename)

"""
    get_experimental_ecdf(experimental)

Returns the ecdf of the experimental data over the range of the experimental
data

"""
get_exp_ecdf(intensity::Vector{Float64}) = cumsum(intensity) ./ sum(intensity)

"""
    ExperimentalSpectrum

An ExperimentalSpectrum holds the information of a NMR experiment in terms of 
the frequency (in a specific unit, like ppm or MHz) and intensity at that 
frequency, as well as the relavent experimental parameters, such as isotope 
and magnet strength

# Fields
- `isotope`
- `B`
- `ν₀`
- `ν`
- `i`
- `ecdf`
"""
struct ExperimentalSpectrum{N}
    isotope::Isotope
    B::typeof(1.0u"T")
    ν₀::typeof(1.0u"MHz")
    ν_start::typeof(1.0u"MHz")
    ν_step::typeof(1.0u"MHz")
    ecdf::NTuple{N, Float64}
end


"""
    ExperimentalSpectrum(filename; freq_unit, isotope, B, header, delim, range)

Construct the ExperimentalSpectrum from a text file and the relevant 
experimental parameters

"""
function ExperimentalSpectrum(
    file::String;
    freq_unit::Unitful.Unitlike = u"ppm",
    isotope::Isotope = Isotope(elements["Boron"], 10),
    B::typeof(1.0u"T") = 7.0u"T",
    header::Int64 = 0,
    delim::String = ",",
    range::Tuple{Quantity, Quantity} = (Quantity(-Inf, freq_unit), 
        Quantity(Inf, freq_unit))
)
    ν₀ = γ(isotope) * B

    if !isdir(file)
        data = map(x -> parse.(Float64, x), 
            split.(readlines(file)[(header + 1):end], delim))
        data = sortslices(hcat(map(x -> x[1], data), map(x -> x[2], data)), 
            dims = 1)
    else
        dic, data = nmrglue.fileio.bruker.read_pdata("10/pdata/1")
        data = nmrglue.fileio.bruker.scale_pdata(dic, data)
        udic = nmrglue.fileio.bruker.guess_udic(dic, data)
        udic[0]["sw"] = dic["procs"]["SW_p"]
        udic[0]["obs"] = dic["procs"]["SF"]
        uc = nmrglue.fileiobase.uc_from_udic(udic)
        ppm_scale = uc.ppm_scale()
        ppm_scale .+= dic["procs"]["OFFSET"] - ppm_scale[1]
        data = sortslices(hcat(ppm_scale, data), dims = 1)
    end

    start_i = findfirst(x -> 
        to_Hz(Quantity(x, freq_unit), ν₀) > to_Hz(range[1], ν₀), data[:, 1])
    stop_i = findlast(x -> 
        to_Hz(Quantity(x, freq_unit), ν₀) < to_Hz(range[2], ν₀), data[:, 1])

    N = stop_i - start_i + 1
    ν_step = 
        to_Hz(Quantity((data[stop_i, 1] - data[start_i, 1]), freq_unit), ν₀) / N
    ν_start = to_Hz(Quantity(data[start_i, 1], freq_unit), ν₀) - (ν_step / 2)
    
    return ExperimentalSpectrum{N}(
        isotope, 
        B, 
        ν₀, 
        ν_start, 
        ν_step, 
        (get_exp_ecdf(data[start_i:stop_i, 2])...,),
    )
end
