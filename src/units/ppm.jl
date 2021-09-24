using Unitful

@unit ppm "ppm" ppm 1. false
Unitful.register(@__MODULE__)

to_ppm(ν::Quantity{Float64, 𝐓^-1, X}, 
    ν₀::Quantity{Float64, 𝐓^-1, Y}) where {X, Y} = (ν / ν₀ - 1) * 1e6u"ppm"
to_ppm(ν::typeof(1.0u"ppm"), _::Quantity{Float64, 𝐓^-1, Y}) where {Y} = ν

to_Hz(ν::typeof(1.0u"ppm"), ν₀::Quantity{Float64, 𝐓^-1, Y}) where {Y} = 
    (1 + ν / 1e6) * ν₀
to_Hz(ν::Quantity{Float64, 𝐓^-1, X}, 
    _::Quantity{Float64, 𝐓^-1, Y}) where {X, Y} = ν