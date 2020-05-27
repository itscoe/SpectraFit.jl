using Distributions


"""
    nmr_params

A structure for holding information about the quadrupolar parameters of the NMR
spectra

# Fields
- `qcc::Array{Distribution}`: an array of distributions of the quantum coupling
constant
- `η::Array{Distribution}`: an array of distributions of the asymmetry parameter
- `weights::Array{Float64}`: an array of relative weights of the above
distributions
"""
struct nmr_params
    qcc::Array{Distribution}
    η::Array{Distribution}
    weights::Array{Float64}
end

"""
    nmr_params(p)

A constructor for the nmr_parameters given an array of floating point numbers.
Order is qcc, σqcc, η, ση, weight, and repeat for each site.

# Examples
```julia-repl
julia> nmr_params([5.5, 0.1, 0.12, 0.03, 1.0])
nmr_params(Distributions.Distribution[Truncated(Distributions.Normal{Float64}
(μ=5.5, σ=0.1), range=(0.0, Inf))],
Distributions.Distribution[Truncated(Distributions.Normal{Float64}(μ=0.12,
σ=0.03), range=(0.0, 1.0))], [1.0])
```
"""
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
