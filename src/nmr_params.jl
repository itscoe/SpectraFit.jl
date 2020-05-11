using Distributions

struct nmr_params
    qcc::Array{Distribution}
    η::Array{Distribution}
    weights::Array{Float64}
end

function nmr_params(p::Array{Float64})
    sites = length(p) ÷ 5
    qcc = Array{Distribution}(undef, sites)
    η = Array{Distribution}(undef, sites)
    weights = zeros(sites)
    for i = 1:sites
        qcc[i] = truncated(
            Normal(p[sites*(i-1)+1], max(0, p[sites*(i-1)+2])),
            0.0,
            Inf,
        )
        η[i] = truncated(
            Normal(p[sites*(i-1)+3], max(0, p[sites*(i-1)+4])),
            0.0,
            1.0,
        )
        weights[i] = p[sites*(i-1)+5]
    end
    weights ./= sum(weights)
    return nmr_params(qcc, η, weights)
end
