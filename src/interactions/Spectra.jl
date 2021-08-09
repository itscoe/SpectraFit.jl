struct Spectra <: AbstractArray{Float64, 1}
    interactions::Tuple{Vararg{NMRInteraction, N}}
    weights::Tuple{Vararg{AbstractFloat, N}}
end

Base.size(S::Spectra) = (mapreduce(x -> size(x)[1], +, S.interactions) + 
    length(S.interactions), )

Base.IndexStyle(::Type{<:Spectra}) = IndexLinear()

function Base.getindex(S::Spectra, i::Int)
    j, k = 0, 1
    while size(S.interactions[k])[1] < i - j + 1
        j += size(S.interactions[k])[1]
        k += 1
    end
    return i - j <= size(S.interactions[k]) ? S.interactions[k][i - j] : weights[k]
end

function lower_bounds(S::Spectra)
    lb = Array{Float64}(undef, 0)
    for i in S.interactions
        lb = vcat(lb, lower_bounds(i), 0.)
    end
    return lb
end

function upper_bounds(S::Spectra)
    ub = Array{Float64}(undef, 0)
    for i in S.interactions
        ub = vcat(ub, upper_bounds(i), 1.)
    end
    return ub
end

function tolerance(S::Spectra)
    tol = Array{Float64}(undef, 0)
    for i in S.interactions
        tol = vcat(tol, tolerance(i), .05)
    end
    return tol
end