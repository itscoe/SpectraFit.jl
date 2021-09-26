struct Spectra{N, M, C}
    components::NTuple{N, C}
    weights::NTuple{M, Float64}
end

function Spectra(N::Int64, x::Vararg{DataType, M}) where {M}
    components = (map(j -> (map(i -> i(), x)...,), 1:N)...,)
    weights = (map(i -> 1 / N, 1:N - 1)...,)
    return Spectra(components, weights)
end

function Spectra(s::Spectra{N, M, C}, p::NTuple{Nₚ, Float64}) where {N, M, C, Nₚ}
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
    return Spectra{N, M, C}((components...,), N == 1 ? () : (p[pᵢ:end]...,))
end

Base.length(S::Spectra{N, M, C}) where {N, M, C} = 
    (mapreduce(length, +, S.components[1]) + 1)N - 1

estimate_powder_pattern(
    c::Tuple{Vararg{NMRInteraction}}, 
    N::Int, 
    exp::ExperimentalSpectra) = 
    mapreduce(i -> estimate_powder_pattern(i, N, exp), .+, c)
