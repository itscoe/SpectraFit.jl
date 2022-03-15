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
    estimate_static_powder_pattern(c, N, exp)

Get the estimated static powder pattern (a vector of N frequencies) given the 
isotropic chemical shift interaction

"""
@inline estimate_static_powder_pattern(
    c::ChemicalShiftI, 
    N::Int, 
    exp::ExperimentalSpectrum
) = (c.δᵢₛₒ / exp.ν_step) .* ones(N)

@inline estimate_static_powder_pattern(
    c::ChemicalShiftI, 
    N::Int, 
    _::Vector{typeof(Quantity(1.0, (u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, (u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, √(u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, √(u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, √(u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, ∛(u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, ∛(u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, ∛(u"MHz m^2 ZV^-1")))},
    _::Vector{typeof(Quantity(1.0, ∛(u"MHz m^2 ZV^-1")))},
    _::Vector{Float64}, 
    _::Vector{Float64},
    _::Vector{Float64},
    _::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz")
) = (c.δᵢₛₒ / ν_step) .* ones(N)

"""
    estimate_mas_powder_pattern(c, N, exp)

Get the estimated MAS powder pattern (a vector of N frequencies) given the 
isotropic chemical shift interaction

"""
@inline estimate_mas_powder_pattern(
    c::ChemicalShiftI, 
    N::Int, 
    exp::ExperimentalSpectrum
) = (c.δᵢₛₒ / exp.ν_step) .* ones(N)

@inline estimate_mas_powder_pattern(
    c::ChemicalShiftI, 
    N::Int, 
    _::Vector{Float64}, 
    _::Vector{Float64},
    _::Vector{FPOT},
    _::FPOT,
    _::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz"),
    ν_start::typeof(1.0u"MHz")
) = (c.δᵢₛₒ / ν_step) .* ones(N)