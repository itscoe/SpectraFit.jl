using Unitful

"""
    Zeeman

Zeeman interaction (parameterless, and centers spectrum on Larmor frequency)

"""
struct Zeeman <: NMRInteraction end

"""
    length(z)

Get the number of free parameters of this interaction (0)

"""
@inline Base.length(_::Zeeman) = 0

"""
    labels(z)

Get the labels of each parameter for plotting purposes

"""
labels(_::Zeeman) = Vector{String}()

"""
    estimate_powder_pattern

Get the estimated static powder pattern (a vector of N frequencies) given the Zeeman interaction and the 
ExperimentalSpectrum

"""
@inline estimate_powder_pattern(
    _::Zeeman,
    N::Int64,
    _::Vector{Float64}, 
    _::Vector{Float64},
    _::Vector{FPOT},
    _::Vector{Float64},
    _::Vector{Float64},
    _::Vector{Float64},
    _::FPOT,
    ν₀::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz"),
    _::typeof(1.0u"T^-1"),
    _::Symbol
) = (ν₀ / ν_step) .* ones(N)
