"""
    μ(N)

Return N randomly sampled μ

"""
@inline μ(N::Int) = rand(N)

"""
    ϕ(N)

Return N randomly sampled ϕ

"""
@inline ϕ(N::Int) = rand(Uniform(0, π), N)

"""
    λ(N)

Return N randomly sampled λ

"""
@inline λ(N::Int) = rand(Arcsine(-1, 1; check_args = false), N)