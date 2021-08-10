struct Spectra{N} <: AbstractArray{Float64, 1}
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

"""
    ols_cdf(parameters, experimental, experimental_ecdf, ν0, I)

Compute ordinary least squares comparing the experimental ecdf with the
theoretical cdf, calculated with the NMR parameters, the spin (I), and the
Larmor frequency (ν0) at each x-value in the experimental data

"""
function ols_cdf(
    parameters::Spectra,
    exp::Array{Float64, 2},
    exp_ecdf::Array{Float64, 1};
    ν0::Float64,
    I::Int64 = 3,
    N::Int64 = 1_000_000,
    transitions::UnitRange{Int64} = 1:(2*I),
)
    th_ecdf = ecdf(estimate_powder_pattern(parameters, N, ν0, I,
        transitions = transitions)).(exp[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]
    return sum((exp_ecdf .- th_ecdf) .^ 2)
end

function estimate_powder_pattern(
    S::Spectra,
    μ::Array{AbstractFloat, N},
    λ::Array{AbstractFloat, N};
    transitions::UnitRange{Int64} = 1:(2*I),
    ν0::AbstractFloat,
    I::Int64,
) where {N}
    tol = Array{Float64}(undef, 0)
    for i in S.interactions
        
    end
    estimate_powder_pattern(
    p::Spectra,
    ν0::AbstractFloat,
    I::Int64,
    μ::Array{AbstractFloat, N},
    λ::Array{AbstractFloat, N};
    transitions::UnitRange{Int64} = 1:(2*I),
)
    qcc = rand(Normal(p.qcc, p.σqcc), N)
    η = rand(Normal(p.η, p.ση), N)
    m_arr = [3, 5, 6, 6, 5, 3]
    m = rand(Categorical(m_arr[transitions] ./ sum(m_arr[transitions])), N) .-
        (length(transitions) ÷ 2)
    return get_ν.(Ref(p.position), qcc, η, μ, λ, m, Ref(I), Ref(ν0))
end