@inline μ(N::Int) = rand(N)
@inline ϕ(N::Int) = rand(N) .* π
@inline λ(N::Int) = cos.(2 .* ϕ(N))
