"""
    Dipolar

Dipolar broadening is currently represented as Gaussian broadening, with σ as 
the standard deviation (and mean of zero)

# Fields
- `σ`
"""
struct Dipolar <: NMRInteraction
    σ::typeof(1.0u"MHz")
end

"""
    length(d)

Get the number of free parameters of this interaction (1)

"""
@inbounds length(_::Dipolar) = 1

"""
    prior(csi, i)

Get the prior distribution of the ith parameter of the dipolar interaction

"""
prior(_::Dipolar, i::Int) = Uniform(0, 1)

"""
    Dipolar()

Default constructor for the dipolar interaction

"""
Dipolar() = Dipolar(0.0u"MHz")

"""
    labels(d)

Get the labels of each parameter for plotting purposes

"""
labels(_::Dipolar) = ["σ (MHz)"]

"""
    Dipolar(σ)

Construct dipolar interaction from float, assuming MHz as units

"""
Dipolar(σ::Float64) = Dipolar(Quantity(σ, u"MHz"))

"""
    estimate_powder_pattern(d, N, exp)

Get the estimated powder pattern (a vector of N frequencies) given the 
dipolar interaction

"""
estimate_powder_pattern(d::Dipolar, N::Int, _::ExperimentalSpectrum) = 
    d.σ * randn(N)