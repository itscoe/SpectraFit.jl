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
    exp::ExperimentalSpectrum) = exp.ν₀ .* ones(N)

@inline estimate_static_powder_pattern(_::Zeeman, N::Int, μs::Vector{Float64}, 
    λs::Vector{Float64}, exp::ExperimentalSpectrum) = exp.ν₀ .* ones(N)

"""
    estimate_mas_powder_pattern(z, N, exp)

Get the estimated MAS powder pattern (a vector of N frequencies) given the 
Zeeman interaction and the ExperimentalSpectrum

"""
@inline estimate_mas_powder_pattern(_::Zeeman, N::Int, 
    exp::ExperimentalSpectrum) = exp.ν₀ .* ones(N)

@inline estimate_mas_powder_pattern(_::Zeeman, N::Int, μs::Vector{Float64}, 
    λs::Vector{Float64}, exp::ExperimentalSpectrum) = exp.ν₀ .* ones(N)