"""
    Spectrum

Theoretical spectrum made up of N components that are weighted with floats that 
add up to one, and the components are NMRInteractions

# Fields
- `components`
- `weights`
"""
struct Spectrum{N, M, C}
    components::NTuple{N, C}
    weights::NTuple{M, Float64}
end

"""
    Spectrum(N, x)

"Easy" Spectrum constructor that allows the user to specify the number of 
sites, and the interactions that make up each component. This is used for 
specifying the form of the model

"""
function Spectrum(N::Int64, x::Vararg{DataType, M}) where {M}
    components = (map(j -> (map(i -> i(), x)...,), 1:N)...,)
    weights = (map(i -> 1 / N, 1:N - 1)...,)
    return Spectrum(components, weights)
end

"""
    Spectrum(s, p)

Construct a Spectrum given an existing Spectrum as the functional form and a 
tuple of parameters as floats

"""
function Spectrum(s::Spectrum{N, M, C}, 
  p::NTuple{Nₚ, Float64}) where {N, M, C, Nₚ}
    pᵢ = 1
    components = Array{typeof(s.components[1])}(undef, N)
    for i = 1:N
        c = s.components[i]
        interactions = Array{NMRInteraction}(undef, length(c))

        for j in 1:length(c)
            nᵢ = length(c[j])
            if !iszero(nᵢ)
                interactions[j] = typeof(c[j])(p[pᵢ:(pᵢ + nᵢ - 1)]...)
            else
                interactions[j] = typeof(c[j])()
            end
            pᵢ += nᵢ
        end

        components[i] = (interactions...,)
    end
    return Spectrum{N, M, C}((components...,), N == 1 ? () : (p[pᵢ:end]...,))
end

"""
    prior(s)

Gets the prior distribution of a spectrum in the form of s

"""
function prior(s::Spectrum{N, M, C}) where {N, M, C}
    dists = Array{Distribution}(undef, length(s))
    p = 1
    for i = 1:N, interaction in s.components[i], j = 1:length(interaction)
        dists[p] = prior(interaction, j)
        p += 1
    end
    for _ = 1:N-1
        dists[p] = Uniform(0, 1)
        p += 1
    end
    return Factored(dists...)
end

"""
    length(s)

Get the number of free parameters in the Spectrum

"""
Base.length(s::Spectrum{N, M, C}) where {N, M, C} = 
    (mapreduce(length, +, s.components[1]) + 1)N - 1

"""
    estimate_static_powder_pattern(c, N, exp)

Get the estimate powder pattern (a vector of N frequencies), given the 
component, and the ExperimentalSpectrum

"""
estimate_static_powder_pattern(
    c::Tuple{Vararg{NMRInteraction}}, 
    N::Int, 
    exp::ExperimentalSpectrum
) = mapreduce(i -> 
    to_Hz.(estimate_static_powder_pattern(i, N, exp), exp.ν₀), .+, c)

estimate_static_powder_pattern(
    c::Tuple{Vararg{NMRInteraction}}, 
    N::Int, 
    μs::Vector{Float64}, 
    λs::Vector{Float64},
    ms::Vector{FPOT},
    I₀::FPOT,
    ν₀::typeof(1.0u"MHz")
) = mapreduce(i -> 
    to_Hz.(estimate_static_powder_pattern(i, N, μs, λs, ms, I₀, ν₀), ν₀), 
        .+, c)

estimate_mas_powder_pattern(
    c::Tuple{Vararg{NMRInteraction}}, 
    N::Int, 
    exp::ExperimentalSpectrum
) = mapreduce(i -> 
    to_Hz.(estimate_mas_powder_pattern(i, N, exp), exp.ν₀), .+, c)

estimate_mas_powder_pattern(
    c::Tuple{Vararg{NMRInteraction}}, 
    N::Int, 
    μs::Vector{Float64}, 
    λs::Vector{Float64},
    ms::Vector{FPOT},
    I₀::FPOT,
    ν₀::typeof(1.0u"MHz")
) = mapreduce(i -> 
    to_Hz.(estimate_mas_powder_pattern(i, N, μs, λs, ms, I₀, ν₀), ν₀), 
        .+, c)

"""
    labels(s)

Get the labels of the free parameters of the Spectrum

"""
function labels(s::Spectrum{N, M, C}) where {N, M, C}
    labels_s = []
    for i = 1:N, c in s.components[i]
        labels_s = vcat(labels_s, labels(c))
    end
    return labels_s
end