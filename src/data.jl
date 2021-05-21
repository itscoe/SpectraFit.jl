using CSV, DataFrames

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
    filename::String;
    ν0_guess::Float64 = 0.0,
    header::Bool = false,
    delim = ",",
    convert_ppm_to_mhz::Bool = ν0_guess != 0,
)
    experimental = CSV.read(filename, DataFrame, header = header, delim = delim)
    rename!(experimental, [:Frequency, :Intensity])
    if typeof(experimental[1, 1]) <: AbstractString
        experimental[!, 1] = parse.(Float64, experimental[:, 1])
    end
    if convert_ppm_to_mhz
        experimental[!, 1] = experimental[:, 1] .* (ν0_guess / (10^6)) .+ ν0_guess
    end
    sort!(experimental, [:Frequency])

    return Matrix(experimental)
end

function get_data(filename::String)
    return joinpath(dirname(pathof(SpectraFit)), "..", "data", filename)
end
