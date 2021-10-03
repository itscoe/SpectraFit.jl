using Unitful, Distributions

"""
    ChemicalShiftI

Isotropic chemical shift interaction (simple shift in frequency)

# Fields
- `δᵢₛₒ`
"""
struct ChemicalShiftI <: NMRInteraction
    δᵢₛₒ::typeof(1.0u"MHz")
end

"""
    ChemicalShiftI()

Default constructor for the isotropic chemical shift interaction

"""
ChemicalShiftI() = ChemicalShiftI(0.0u"MHz")

"""
    labels(csi)

Get the labels of each parameter for plotting purposes

"""
labels(_::ChemicalShiftI) = ["Shift (MHz)"]

"""
    ChemicalShiftI(c)

Construct isotropic chemical shift interaction from float, assuming MHz as units

"""
ChemicalShiftI(c::Float64) = ChemicalShiftI(Quantity(c, u"MHz"))

"""
    prior(csi, i)

Get the prior distribution of the ith parameter of the isotropic chemical shift 
interaction

"""
prior(_::ChemicalShiftI, _::Int) = Uniform(-1, 1)

"""
    length(csi)

Get the number of free parameters of this interaction (1)

"""
@inline Base.length(_::ChemicalShiftI) = 1

"""
    get_ν(δᵢₛₒ)

Get the frequency given the parameters, δᵢₛₒ

"""
get_ν(δᵢₛₒ::Float64) = δᵢₛₒ

"""
    estimate_powder_pattern(c, N, exp)

Get the estimated powder pattern (a vector of N frequencies) given the 
isotropic chemical shift interaction

"""
@inline estimate_powder_pattern(
    c::ChemicalShiftI, 
    N::Int, 
    _::ExperimentalSpectrum
) = c.δᵢₛₒ .* ones(N)