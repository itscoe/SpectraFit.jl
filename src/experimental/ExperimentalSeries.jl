"""
    ExperimentalSeries

An ExperimentalSeries is a struct for holding a series of experimental spectra 
that vary over an independent variable (composition, thermal history, etc.)

# Fields
- `spectra`
- `ind_var`
- `ind_var_label`
"""

struct ExperimentalSeries
    spectra::Vector{ExperimentalSpectrum}
    ind_var::Vector{Float64}
    ind_var_label::String
end

