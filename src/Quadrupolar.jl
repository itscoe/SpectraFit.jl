using Distributions

const m_arr = [3, 5, 6, 6, 5, 3]
const μ_dist = Uniform(0, 1)
const λ_dist = Uniform(-1, 1)

"""
    Quadrupolar

A structure for holding information about the quadrupolar parameters of the NMR
spectra

# Fields
- `qcc::Array{Distribution}`: an array of distributions of the quantum coupling
constant
- `η::Array{Distribution}`: an array of distributions of the asymmetry parameter
- `weights::Array{Float64}`: an array of relative weights of the above
distributions
"""
struct Quadrupolar
    qcc::Array{Distribution}
    η::Array{Distribution}
    weights::Array{Float64}
end

"""
    Quadrupolar(p)

A constructor for the nmr_parameters given an array of floating point numbers.
Order is qcc, σqcc, η, ση, weight, and repeat for each site.

# Examples
```julia-repl
julia> Quadrupolar([5.5, 0.1, 0.12, 0.03, 1.0])
Quadrupolar(Distributions.Distribution[Truncated(Distributions.Normal{Float64}
(μ=5.5, σ=0.1), range=(0.0, Inf))],Distributions.Distribution[Truncated(
Distributions.Normal{Float64}(μ=0.12, σ=0.03), range=(0.0, 1.0))], [1.0])
```
"""
function Quadrupolar(p::Array{Float64})
    sites = length(p) ÷ 5 + 1
    qcc = Array{Distribution}(undef, sites)
    η = Array{Distribution}(undef, sites)
    weights = zeros(sites)

    weights_sum = sum(map(i -> i % 5 == 0 ? p[i] : 0.0, 1:length(p)))
    weights_sum > 1.0 && return missing
    for i in 1:length(p)
        p[i] < 0 && return missing
        (isnan(p[i]) || isinf(p[i])) && return missing
        (isnothing(p[i]) || ismissing(p[i])) && return missing
    end
    for i = 1:sites
        qcc[i] = truncated(Normal(p[5*(i-1)+1], p[5*(i-1)+2]), 0.0, Inf)
        η[i] = truncated(Normal(p[5*(i-1)+3], p[5*(i-1)+4]), 0.0, 1.0)
        weights[i] = i != sites ? p[5*i] : 1 - weights_sum
    end

    return Quadrupolar(qcc, η, weights)
end

"""
    get_ν(qcc, η, μ, λ, m, I, ν0)

Compute the value of ν with the third order perturbation described in

>Jellison Jr, G. E., Feller, S. A., & Bray, P. J. (1977). NMR powder patterns
>for integer spin nuclei in the presence of asymmetric quadrupole effects.
>Journal of Magnetic Resonance (1969), 27(1), 121-132.

# Arguments
- `qcc::Float64`: the quantum coupling constant
- `η::Float64`: the asymmetry parameter (between 0 and 1 inclusive)
- `μ::Float64`: cos(θ), where θ is the spherical coordinate angle
- `λ::Float64`: cos(2ϕ), where ϕ is the spherical coordinate angle
- `m::Int64`: the quantum number m, which can be integers from -I to I - 1
- `I::Int64`: spin (3 in the case of Boron-10)
- `ν0::Float64`: the Larmor frequency

# Examples
```julia-repl
julia> get_ν(5.5, 0.12, 0.1, 0.2, -1, 3, 32.239)
31.8515444235865
```
"""
function get_ν(
    qcc::Float64,
    η::Float64,
    μ::Float64,
    λ::Float64,
    m::Int64,
    I::Int64,
    ν0::Float64,
)
    νQ = 3 * qcc / (2 * I * (2 * I - 1))
    β = νQ / ν0
    a = -(3 * μ^2 - 1 + η * λ - η * λ * μ^2)
    c = (0.5 * ((I + 3 / 2) * (I - 1 / 2) - 3 * (m - 1 / 2)^2)) *
        (μ^4 * (3 - η * λ)^2 + 2 * μ^2 * (-9 + 2 * η^2 - η^2 * λ^2) +
         (3 + η * λ)^2) +
        (4 * ((I + 3 / 2) * (I - 1 / 2) - 6 * (m - 1 / 2)^2)) *
        (μ^4 * (3 - η * λ)^2 + μ^2 * (-9 + η^2 + 6 * η * λ - 2 * η^2 * λ^2) +
         (-(η^2) + η^2 * λ^2))
    e = (12 * I * (I + 1) - 40 * m * (m - 1) - 27) * (μ^6 * (3 - η * λ)^3 +
         μ^4 * (-36 + 3 * η^2 + 42 * η * λ - η^3 * λ - 19 * η^2 * λ^2 +
          3 * η^3 * λ^3) +
         μ^2 * (9 - 4 * η^2 - 15 * η * λ + 2 * η^3 * λ + 11 * η^2 * λ^2 -
          3 * η^3 * λ^3) +
         (η^2 - η^3 * λ - η^2 * λ^2 + η^3 * λ^3)) +
        (1 / 2 * (3 * I * (I + 1) - 5 * m * (m - 1) - 6)) *
        (μ^6 * (3 - η * λ)^3 +
         μ^4 * (-63 + 12 * η^2 + 33 * η * λ - 4 * η^3 * λ - 13 * η^2 * λ^2 +
          3 * η^3 * λ^3) +
         μ^2 *
         (45 - 4 * η^2 - 9 * η * λ + 4 * η^3 * λ - η^2 * λ^2 - 3 * η^3 * λ^3) +
         (-9 + 3 * η * λ + 5 * η^2 * λ^2 + η^3 * λ^3)) +
        (8 * I * (I + 1) - 20 * m * (m - 1) - 15) * (μ^6 * (3 - η * λ)^3 +
         μ^4 * (-54 + 9 * η^2 + 36 * η * λ - 3 * η^3 * λ - 15 * η^2 * λ^2 +
          3 * η^3 * λ^3) +
         μ^2 * (27 - 6 * η^2 - 9 * η * λ + 4 * η^3 * λ + 3 * η^2 * λ^2 -
          3 * η^3 * λ^3) +
         (-3 * η^2 - η^3 * λ + 3 * η^2 * λ^2 + η^3 * λ^3))
    return ν0 + (νQ / 2) * (m - 1 / 2) * a + (νQ * β / 72) * c +
           (νQ * β^2 / 144) * e * (m - 1 / 2)
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
function estimate_powder_pattern(
    p::Quadrupolar,
    N::Int64,
    ν0::Float64,
    I::Int64;
    transitions::UnitRange{Int64} = 1:(2*I),
)
    sites = rand(Categorical(p.weights), N)
    powder_pattern = get_ν.(
        rand.(map(i -> p.qcc[i], sites)),
        rand.(map(i -> p.η[i], sites)),
        rand(μ_dist, N),
        rand(λ_dist, N),
        rand(Categorical(m_arr[transitions] ./
            sum(m_arr[transitions])), N) .- (length(transitions) ÷ 2),
        I,
        ν0,
    )
    return powder_pattern
end

function get_quadrupolar_starting_values(sites::Int64)
    starting_values = zeros(5 * sites - 1)
    starting_values[1:5:end] = rand(Uniform(0, 3), sites)  # √Qcc
    starting_values[2:5:end] = rand(Uniform(0, 1), sites)  # √σQcc
    starting_values[3:5:end] = rand(Uniform(0, 1), sites)  # √η
    starting_values[4:5:end] = rand(Uniform(0, 1), sites)  # √ση
    if sites > 1
        starting_values[5:5:end] = rand(Uniform(0, 1 / sites), sites - 1)
    end
    return starting_values
end
