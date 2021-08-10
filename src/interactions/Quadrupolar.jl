using Distributions

"""
    Quadrupolar

A structure for holding information about the quadrupolar parameters of the NMR
spectra

# Fields
- 'position'
- `qcc`
- `σqcc`
- `η`
- `ση`
"""
struct Quadrupolar{T <: AbstractFloat} <: NMRInteraction
    position::T
    qcc::T
    σqcc::T
    η::T
    ση::T
end

Base.size(Q::Quadrupolar) = (5,)

Base.getindex(Q::Quadrupolar, i::Int) = 
    i == 1 ? Q.qcc : i == 2 ? Q.σqcc : i == 3 ? Q.η : Q.ση

function Quadrupolar() 
    return Quadrupolar(
        rand(Uniform(-5., 5.)),
        rand(Uniform(0.,  9.)),
        rand(Uniform(0.,  1.)),
        rand(Uniform(0.,  1.)),
        rand(Uniform(0.,  1.)),
    )
end

lower_bounds(Q::Quadrupolar) = [-5., 0., 0., 0., 0.]
upper_bounds(Q::Quadrupolar) = [5.,  9., 1., 1., 1.]
tolerance(Q::Quadrupolar)    = [.1,  .5, .1, .1, .1]

"""
    get_ν(qcc, η, μ, λ, m, I, ν0)

Compute the value of ν with the third order perturbation described in

>Jellison Jr, G. E., Feller, S. A., & Bray, P. J. (1977). NMR powder patterns
>for integer spin nuclei in the presence of asymmetric quadrupole effects.
>Journal of Magnetic Resonance (1969), 27(1), 121-132.

# Arguments
- `qcc::AbstractFloat`: the quantum coupling constant
- `η::AbstractFloat`: the asymmetry parameter (between 0 and 1 inclusive)
- `μ::AbstractFloat`: cos(θ), where θ is the spherical coordinate angle
- `λ::AbstractFloat`: cos(2ϕ), where ϕ is the spherical coordinate angle
- `m::Int64`: the quantum number m, which can be integers from -I to I - 1
- `I::Int64`: spin (3 in the case of Boron-10)
- `ν0::AbstractFloat`: the Larmor frequency

# Examples
```julia-repl
julia> get_ν(5.5, 0.12, 0.1, 0.2, -1, 3, 32.239)
31.8515444235865
```
"""
function get_ν(position::T1, qcc::T1, η::T1, μ::T1, λ::T1, 
    m::T2, I::T2, ν0::T1) where {T1 <: AbstractFloat, T2 <: Int}
    νQ = 3 * qcc / (2 * I * (2 * I - 1))
    β = νQ / ν0
    a = -(3 * μ^2 - 1 + η * λ - η * λ * μ^2)
    c = (0.5 * ((I + 3 / 2) * (I - 1 / 2) - 3 * (m - 1 / 2)^2)) *
        (μ^4 * (3 - η * λ)^2 + 2 * μ^2 * (-9 + 2 * η^2 - η^2 * λ^2) +
        (3 + η * λ)^2) + (4 * ((I + 3 / 2) * (I - 1 / 2) - 6 * (m - 1 / 2)^2)) *
        (μ^4 * (3 - η * λ)^2 + μ^2 * (-9 + η^2 + 6 * η * λ - 2 * η^2 * λ^2) +
        (-(η^2) + η^2 * λ^2))
    e = (12 * I * (I + 1) - 40 * m * (m - 1) - 27) * (μ*μ*μ*μ*μ*μ *
        (3 - η * λ)^3 + μ^4 * (-36 + 3 * η^2 + 42 * η * λ - η^3 * λ - 19 * η^2 *
        λ^2 + 3 * η^3 * λ^3) + μ^2 * (9 - 4 * η^2 - 15 * η * λ + 2 * η^3 * λ +
        11 * η^2 * λ^2 - 3 * η^3 * λ^3) + (η^2 - η^3 * λ - η^2 * λ^2 + η^3 *
        λ^3)) + (1 / 2 * (3 * I * (I + 1) - 5 * m * (m - 1) - 6)) *
        (μ*μ*μ*μ*μ*μ * (3 - η * λ)^3 + μ^4 * (-63 + 12 * η^2 + 33 * η * λ - 4 *
        η^3 * λ - 13 * η^2 * λ^2 + 3 * η^3 * λ^3) + μ^2 * (45 - 4 * η^2 - 9 *
        η * λ + 4 * η^3 * λ - η^2 * λ^2 - 3 * η^3 * λ^3) + (-9 + 3 * η * λ + 5 *
        η^2 * λ^2 + η^3 * λ^3)) + (8 * I * (I + 1) - 20 * m * (m - 1) - 15) *
        (μ*μ*μ*μ*μ*μ * (3 - η * λ)^3 + μ^4 * (-54 + 9 * η^2 + 36 * η * λ - 3 *
        η^3 * λ - 15 * η^2 * λ^2 + 3 * η^3 * λ^3) + μ^2 * (27 - 6 * η^2 - 9 *
        η * λ + 4 * η^3 * λ + 3 * η^2 * λ^2 - 3 * η^3 * λ^3) + (-3 * η^2 - η^3 *
        λ + 3 * η^2 * λ^2 + η^3 * λ^3))
    return ν0 + position + (νQ / 2) * (m - 1 / 2) * a + (νQ * β / 72) * c + 
        (νQ * β^2 / 144) * e * (m - 1 / 2)
end

function estimate_powder_pattern(
    p::Quadrupolar,
    ν0::AbstractFloat,
    I::Int64,
    μ::Array{AbstractFloat, N},
    λ::Array{AbstractFloat, N};
    transitions::UnitRange{Int64} = 1:(2*I),
) where {N}
    qcc = rand(Normal(p.qcc, p.σqcc), N)
    η = rand(Normal(p.η, p.ση), N)
    m_arr = [3, 5, 6, 6, 5, 3]
    m = rand(Categorical(m_arr[transitions] ./ sum(m_arr[transitions])), N) .-
        (length(transitions) ÷ 2)
    return get_ν.(Ref(p.position), qcc, η, μ, λ, m, Ref(I), Ref(ν0))
end

"""
    estimate_powder_pattern(p, N, ν0, I)

Compute N frequencies, whose distribution forms what is referred to as the
powder pattern, for one or more given distributions (found in p) for the quantum
coupling constant (qcc) and asymmetry parameter (η), and constant Larmor
frequency (ν0) and spin (I).

# Examples
```julia-repl
julia> estimate_powder_pattern(Quadrupolar([5.5, 0.1, 0.12, 0.03, 1.0]), 1000, 32.239, 3)
1000-element Array{Float64,1}:
 32.18456766333233
 32.2091872593358
 32.32362143777174
 32.33050236817988
 31.950161601754655
 32.338682082979666
  ⋮
 32.42199530095508
 32.26831269193921
 32.12305522098913
 32.04062886358494
 32.15411486178574
```
"""
estimate_powder_pattern(p::Quadrupolar, N::Int, ν0::AbstractFloat, I::Int; 
    transitions::UnitRange{Int} = 1:(2*I)) = 
    estimate_powder_pattern(p, ν0, I, μ(N), λ(N), transitions = transitions)


