using Unitful, Distributions

"""
    Quadrupolar

A structure for holding information about the quadrupolar parameters of the NMR
Spectrum utilizing the Extended Czjzek Model (ECM) for parameterization

# Fields
- `qcc`
- `η`
- `ρ`
"""
struct Quadrupolar <: NMRInteraction
    Vzz::typeof(1.0u"ZV/m^2")
    η::Float64
    ρσ::typeof(1.0u"ZV/m^2")
end

"""
    prior(csi, i)

Get the prior distribution of the ith parameter of the quadrupolar interaction

"""
prior(_::Quadrupolar, i::Int) = 
    i == 1 ? Uniform(0.0, 4.5) : 
    i == 2 ? Uniform(0, 1) : 
             Uniform(0, 1)

"""
    Quadrupolar()

Default constructor for the quadrupolar interaction

"""
Quadrupolar() = Quadrupolar(0.0u"ZV/m^2", 0., 0.0u"ZV/m^2")

"""
    labels(d)

Get the labels of each parameter for plotting purposes

"""
labels(_::Quadrupolar) = ["|Vzz| (ZV/m²)", "η", "ρσ (ZV/m²)"]

"""
    Quadrupolar(Vzz, η, ρσ)

Construct quadrupolar interaction from floats, assuming MHz as units

"""
Quadrupolar(Vzz::Float64, η::Float64, ρσ::Float64) = 
    Quadrupolar(Quantity(Vzz, u"ZV/m^2"), η, Quantity(ρσ, u"ZV/m^2"))

"""
    length(d)

Get the number of free parameters of this interaction (3)

"""
@inbounds length(_::Quadrupolar) = 3

"""
    get_ν(qcc, η, μ, λ, m, I, ν0)

Compute the value of ν with the third order perturbation described in

>Jellison Jr, G. E., Feller, S. A., & Bray, P. J. (1977). NMR powder patterns
>    for integer spin nuclei in the presence of asymmetric quadrupole effects.
>    Journal of Magnetic Resonance (1969), 27(1), 121-132.

# Arguments
- `qcc::typeof(1.0u"MHz")`: the quantum coupling constant
- `η::AbstractFloat`: the asymmetry parameter (between 0 and 1 inclusive)
- `μ::AbstractFloat`: cos(θ), where θ is the spherical coordinate angle
- `λ::AbstractFloat`: cos(2ϕ), where ϕ is the spherical coordinate angle
- `m::Int64`: the quantum number m, which can be integers from -I to I - 1
- `I::Int64`: spin (3 in the case of Boron-10)
- `ν0::typeof(1.0u"MHz")`: the Larmor frequency

# Examples
```julia-repl
julia> get_ν(5.5, 0.12, 0.1, 0.2, -1, 3, 32.239)
31.8515444235865
```
"""
function get_ν(qcc::typeof(1.0u"MHz"), η::Float64, μ::Float64, λ::Float64, 
    m::Int64, I::Int64, ν₀::typeof(1.0u"MHz"))

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

    νQ = 3 * qcc / (2 * I * (2 * I - 1))

    return (νQ / 2) * (m - 1 / 2) * a + 
        (νQ ^ 2 / 72ν₀) * c + 
        (νQ ^ 3 / 144ν₀ ^ 2) * e * (m - 1 / 2)
end

"""
    estimate_powder_pattern(q, N, μs, λs, isotope, ν₀)

Get the estimated powder pattern (a vector of N frequencies) given the 
quadrupolar interaction, vectors of the Euler angles, the isotope, 
and the Larmor frequency

"""
function estimate_powder_pattern(q::Quadrupolar, N::Int, 
    μs::Vector{Float64}, λs::Vector{Float64}, isotope::Isotope, 
    ν₀::typeof(1.0u"MHz"))

    σ = ustrip(q.ρσ / 2)
    μᵤ1 = ustrip(q.Vzz) / 2
    μᵤ5 = q.η * ustrip(q.Vzz) / 2√3

    U1 = Quantity.(rand(Normal(μᵤ1, σ), N), unit(q.Vzz))
    U5 = Quantity.(rand(Normal(μᵤ5, σ), N), unit(q.Vzz))

    Vxx, Vyy, Vzz = -U1 .+ √3U5, -U1 .- √3U5, 2U1

    Qccs = (e * abs(Q(isotope)) * abs.(Vzz) / h) .|> u"MHz"
    ηs = (Vyy .- Vxx) ./ Vzz

    I₀ = Int64(I(isotope))
    ms = rand(Categorical([3, 5, 6, 6, 5, 3] ./ 28), N) .- I₀

    return get_ν.(Qccs, ηs, μs, λs, ms, I₀, ν₀)
end

"""
    estimate_powder_pattern(q, N, μs, λs, isotope, ν₀)

Get the estimated powder pattern (a vector of N frequencies) given the 
quadrupolar interaction, vectors of the Euler angles, the isotope, 
and the Larmor frequency

"""
@inline estimate_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    μs::Vector{Float64}, 
    λs::Vector{Float64}, 
    isotope::Isotope, 
    ν₀
) = estimate_powder_pattern(q, N, μs, λs, isotope, ν₀ |> u"MHz")

"""
    estimate_powder_pattern(q, N, isotope, ν₀)

Get the estimated powder pattern (a vector of N frequencies) given the 
quadrupolar interaction, the isotope, and the Larmor frequency

"""
@inline estimate_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    isotope::Isotope, 
    ν₀
) = estimate_powder_pattern(q, N, μ(N), λ(N), isotope, ν₀)

"""
    estimate_powder_pattern(q, N, exp)

Get the estimated powder pattern (a vector of N frequencies) given the 
quadrupolar interaction and the ExperimentalSpectrum

"""
@inline estimate_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    exp::ExperimentalSpectrum
) = estimate_powder_pattern(q, N, μ(N), λ(N), exp.isotope, exp.ν₀)