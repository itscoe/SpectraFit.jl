using Distributions

struct Spectra{N::Int64, I <: Vararg{NMRInteraction}} <: AbstractVector{Float64}
    components::NTuple{Tuple{I}, N}
    weights::NTuple{Float64, N}
end

Base.size(S::Spectra{N, I}) where {N, I} = 
    ((mapreduce(length, +, S.components[1]) + 1)N, )

function prior(s::Spectra{N, I}) where {N, I}
    dists = Array{Distribution}(undef, length(s))
    p = 1
    for i = 1:N
        for interaction in s.components[i], j = 1:length(interaction)
            dists[p] = prior(interaction, j)
            p += 1
        end
        dists[p] = Uniform(0, 1)
        p += 1
    end
    return Factored(dists...)
end

Base.IndexStyle(::Type{<:Spectra}) = IndexLinear()

function Base.getindex(S::Spectra, i::Int)
    j, k = 0, 1
    while length(S.components[k]) < i - j + 1
        j += length(S.components[k])
        k += 1
    end
    return i - j <= length(S.components[k]) ? 
        S.components[k][i - j] : weights[k]
end

estimate_powder_pattern(c::Tuple{Vararg{NMRInteraction}}, N::Int) = 
    mapreduce(i -> estimate_powder_pattern(i, N), .+, c)
    