"""
    μ(N)

Return N randomly sampled μ

"""
@inline μ(N::Int) = rand(N)

"""
    ϕ(N)

Return N randomly sampled ϕ

"""
@inline ϕ(N::Int) = rand(N) .* π

"""
    λ(N)

Return N randomly sampled λ

"""
@inline λ(N::Int) = cos.(2 .* ϕ(N))
