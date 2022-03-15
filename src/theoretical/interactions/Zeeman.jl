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
    estimate_static_powder_pattern(z, N, exp)

Get the estimated static powder pattern (a vector of N frequencies) given the 
Zeeman interaction and the ExperimentalSpectrum

"""
@inline estimate_static_powder_pattern(_::Zeeman, N::Int, 
    exp::ExperimentalSpectrum) = 
    (exp.ν₀ / exp.ν_step) .* ones(N)

    estimate_static_powder_pattern(i, N, coefs, u0, u1, u5, ν_step)

@inline estimate_static_powder_pattern(
    _::Zeeman, 
    N::Int, 
    _,
    _::Vector{Float64}, 
    _::Vector{Float64},
    _::Vector{Float64},
    ν₀::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz"),
) = (ν₀ / ν_step) .* ones(N)

"""
    estimate_mas_powder_pattern(z, N, exp)

Get the estimated MAS powder pattern (a vector of N frequencies) given the 
Zeeman interaction and the ExperimentalSpectrum

"""
@inline estimate_mas_powder_pattern(_::Zeeman, N::Int, 
    exp::ExperimentalSpectrum) = 
    (exp.ν₀ / exp.ν_step) .* ones(N)

@inline estimate_mas_powder_pattern(
    _::Zeeman, 
    N::Int, 
    _::Vector{Float64}, 
    _::Vector{Float64},
    _::Vector{FPOT},
    _::FPOT,
    ν₀::typeof(1.0u"MHz"),
    ν_step::typeof(1.0u"MHz"),
    ν_start::typeof(1.0u"MHz")
) = (ν₀ / ν_step) .* ones(N)