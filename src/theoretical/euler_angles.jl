using Distributions

const ϕ_dist = Uniform(0, π)
const λ_dist = Arcsine(-1, 1; check_args = false)

"""
    μ(N)

Return N randomly sampled μ

"""
@inline μ(N::Int) = rand(N)

"""
    ϕ(N)

Return N randomly sampled ϕ

"""
@inline ϕ(N::Int) = rand(ϕ_dist, N)

"""
    λ(N)

Return N randomly sampled λ

"""
@inline λ(N::Int) = rand(λ_dist, N)