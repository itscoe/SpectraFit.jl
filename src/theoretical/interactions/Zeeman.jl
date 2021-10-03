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
@inbounds length(_::Zeeman) = 0

"""
    labels(z)

Get the labels of each parameter for plotting purposes

"""
labels(_::Zeeman) = Vector{String}()

"""
    estimate_powder_pattern(z, N, ν₀)

Get the estimated powder pattern (a vector of N frequencies) given the 
Zeeman interaction and the Larmor frequency

"""
@inbounds estimate_powder_pattern(_::Zeeman, N::Int, ν₀::typeof(1.0u"MHz")) = 
    ν₀ .* ones(N)

"""
    estimate_powder_pattern(z, N, ν₀)

Get the estimated powder pattern (a vector of N frequencies) given the 
Zeeman interaction and the Larmor frequency

"""
@inbounds estimate_powder_pattern(_::Zeeman, N::Int, ν₀) = 
    (ν₀ |> u"mHz") .* ones(N)

"""
    estimate_powder_pattern(z, N, exp)

Get the estimated powder pattern (a vector of N frequencies) given the 
Zeeman interaction and the ExperimentalSpectrum

"""
@inbounds estimate_powder_pattern(_::Zeeman, N::Int, exp::ExperimentalSpectrum) = 
    exp.ν₀ .* ones(N)