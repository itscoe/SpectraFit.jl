struct Spectra <: AbstractArray{Float64, 1}
    interactions::Tuple{Vararg{NMRInteraction}}
end

Base.size(S::Spectra) = (mapreduce(size[1], +, S.interactions), )

Base.IndexStyle(::Type{<:Spectra}) = IndexLinear()

function Base.getindex(S::Spectra, i::Int)
    j, k = 0, 1
    while size(S.interactions[k])[1] < i - j
        j += size(S.interactions[k])[1]
        k += 1
    end
    return S.interactions[k][i - j]
end