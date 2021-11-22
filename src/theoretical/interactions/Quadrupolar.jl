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
@inline Base.length(_::Quadrupolar) = 3

"""
    get_ν1(qcc, η, μ, λ, m, I, ν0)

Compute the 1st order component of ν with the third order perturbation 
described in

>Jellison Jr, G. E., Feller, S. A., & Bray, P. J. (1977). NMR powder patterns
>    for integer spin nuclei in the presence of asymmetric quadrupole effects.
>    Journal of Magnetic Resonance (1969), 27(1), 121-132.

# Arguments
- `qcc::typeof(1.0u"MHz")`: the quantum coupling constant
- `η::AbstractFloat`: the asymmetry parameter (between 0 and 1 inclusive)
- `μ::AbstractFloat`: cos(θ), where θ is the spherical coordinate angle
- `λ::AbstractFloat`: cos(2ϕ), where ϕ is the spherical coordinate angle
- `m::Rational`: the quantum number m, which can be integers from -I to I - 1
- `I::Rational`: spin (3 in the case of Boron-10)

# Examples
```julia-repl
julia> get_ν(5.5, 0.12, 0.1, 0.2, -1, 3, 32.239)
31.8515444235865
```
"""
@inline function get_ν1(νQ::typeof(1.0u"MHz"), η::Float64, μ::Float64, 
    λ::Float64, m::FPOT)
    return (νQ / 2) * Float64(m - FPOT(1, 1)) * (-3μ^2 + ((μ^2 - 1)λ)η + 1)
end

"""
    get_ν2(qcc, η, μ, λ, m, I, ν0)

Compute the 2nd order component of ν with the third order perturbation 
described in

>Jellison Jr, G. E., Feller, S. A., & Bray, P. J. (1977). NMR powder patterns
>    for integer spin nuclei in the presence of asymmetric quadrupole effects.
>    Journal of Magnetic Resonance (1969), 27(1), 121-132.

# Arguments
- `qcc::typeof(1.0u"MHz")`: the quantum coupling constant
- `η::AbstractFloat`: the asymmetry parameter (between 0 and 1 inclusive)
- `μ::AbstractFloat`: cos(θ), where θ is the spherical coordinate angle
- `λ::AbstractFloat`: cos(2ϕ), where ϕ is the spherical coordinate angle
- `m::Rational`: the quantum number m, which can be integers from -I to I - 1
- `I::Rational`: spin (3 in the case of Boron-10)

# Examples
```julia-repl
julia> get_ν(5.5, 0.12, 0.1, 0.2, -1, 3, 32.239)
31.8515444235865
```
"""
@inline function get_ν2(νQ::typeof(1.0u"MHz"), η::Float64, μ::Float64, 
    λ::Float64, m::FPOT, I::FPOT, ν₀::typeof(1.0u"MHz"))
    return (νQ ^ 2 / 72ν₀) * (
        Float64(FPOT(1, 1) * ((I + FPOT(3, 1)) * (I - FPOT(1, 1)) - 
            3 * (m - FPOT(1, 1))^2)) *
            ((3 + η * λ)^2 + (-18 + (4 - 2λ^2)η^2 + ((3 - η * λ)^2)μ^2)μ^2) + 
        Float64(4 * ((I + FPOT(3, 1)) * (I - FPOT(1, 1)) - 
            6 * (m - FPOT(1, 1))^2)) *
            ((λ^2 - 1)η^2 + (-9 + (6λ + (1 - 2λ^2)η)η + ((3 - η * λ)^2)μ^2)μ^2)
    )
end

"""
    get_ν3(qcc, η, μ, λ, m, I, ν0)

Compute the 3rd order component of ν with the third order perturbation 
described in

>Jellison Jr, G. E., Feller, S. A., & Bray, P. J. (1977). NMR powder patterns
>    for integer spin nuclei in the presence of asymmetric quadrupole effects.
>    Journal of Magnetic Resonance (1969), 27(1), 121-132.

# Arguments
- `qcc::typeof(1.0u"MHz")`: the quantum coupling constant
- `η::AbstractFloat`: the asymmetry parameter (between 0 and 1 inclusive)
- `μ::AbstractFloat`: cos(θ), where θ is the spherical coordinate angle
- `λ::AbstractFloat`: cos(2ϕ), where ϕ is the spherical coordinate angle
- `m::Rational`: the quantum number m, which can be integers from -I to I - 1
- `I::Rational`: spin (3 in the case of Boron-10)

# Examples
```julia-repl
julia> get_ν(5.5, 0.12, 0.1, 0.2, -1, 3, 32.239)
31.8515444235865
```
"""
@inline function get_ν3(νQ::typeof(1.0u"MHz"), η::Float64, μ::Float64, 
    λ::Float64, m::Int64, I::Int64, ν₀::typeof(1.0u"MHz"))
    return (νQ^3 / 144ν₀^2) * Float64(m - FPOT(1, 1)) * (
        Float64(12 * (I + 1)I - 40 * (m - 1)m - 27) * 
            ((1 - λ^2 + ((λ^2 - 1)λ)η)η^2 + 
            (9 + (-15λ * (-4 + 11λ^2 + ((2 - 3λ^2)λ)η)η)η + 
            (-36 + (42λ + (3 - 19λ^2 + ((3λ^2 - 1)λ)η)η)η + 
            ((3 - η * λ)^3)μ^2)μ^2)μ^2) +
        Float64((3 * (I + 1)I - 5 * (m - 1)m - 6) / 2) *
            (-9 + (3λ + (5λ^2 + (λ^3)η)η)η +
            (45 - (9λ + (4 - λ^2 + (4 - 3λ^2)η)η)η + 
            (-63 + (33λ + (12 - 13λ^2 + ((3λ^2 - 4)λ)η)η)η + 
            ((3 - η * λ)^3)μ^2)μ^2)μ^2) +
        Float64(8 * (I + 1)I - 20 * (m - 1)m - 15) *
            ((-3 + (-η + (3 + (η)λ)λ)λ)η^2 + 
            (27 + (-9λ + (3λ^2 - 6 + ((4 - 3λ^2)λ)η)η)η +
            (-54 + (36λ + (9 - 15λ^2 + ((3λ^2 - 3)λ)η)η)η + 
            ((3 - η * λ)^3)μ^2)μ^2)μ^2
        )
    )
end

@inline function get_m(i::Int64, m_vec::Vector{Int64}, I::FPOT)
    @inbounds j = m_vec[1] 
    i <= j && return -I + 1
    for k = 2:length(m_vec) - 1
        @inbounds j += m_vec[k]
        i <= j && return k - I
    end
    return length(m_vec) - I
end

"""
    estimate_static_powder_pattern(q, N, μs, λs, isotope, ν₀)

Get the estimated static powder pattern (a vector of N frequencies) given the 
quadrupolar interaction, vectors of the Euler angles, the isotope, 
and the Larmor frequency

"""
function estimate_static_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    μs::Vector{Float64}, 
    λs::Vector{Float64},
    exp::ExperimentalSpectrum
)
    I₀ = I(exp.isotope)
    
    U1 = (q.Vzz / 2) .+ (q.ρσ / 2) .* randn(N)
    
    νQ_c = 2e * 0.0845e-28u"m^2" / (2h/3 * Float64(I₀ * (2 * I₀ - 1)))
    
    νQs = abs.(U1 .* νQ_c) .|> u"MHz"
    ηs = -√3 * (q.η * q.Vzz / 2√3) ./ U1 .+ (q.ρσ / 2) .* randn(N) ./ U1

    m_vec = map(m -> I₀ * (I₀ + 1) - m * (m - 1), (-I₀ + 1):I₀)
    ms = get_m.(rand(1:sum(m_vec), N), Ref(m_vec), I₀)
    
    return get_ν1.(νQs, ηs, μs, λs, ms) .+ 
           get_ν2.(νQs, ηs, μs, λs, ms, I₀, exp.ν₀) .+
           get_ν3.(νQs, ηs, μs, λs, ms, I₀, exp.ν₀)
end

"""
    estimate_static_powder_pattern(q, N, exp)

Get the estimated powder pattern (a vector of N frequencies) given the 
quadrupolar interaction and the ExperimentalSpectrum

"""
@inline estimate_static_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    exp::ExperimentalSpectrum
) = estimate_static_powder_pattern(q, N, μ(N), λ(N), exp)

"""
    estimate_mas_powder_pattern(q, N, μs, λs, isotope, ν₀)

Get the estimated static powder pattern (a vector of N frequencies) given the 
quadrupolar interaction, vectors of the Euler angles, the isotope, 
and the Larmor frequency

"""
function estimate_mas_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    μs::Vector{Float64}, 
    λs::Vector{Float64},
    exp::ExperimentalSpectrum
)
    I₀ = I(exp.isotope)
    
    U1 = (q.Vzz / 2) .+ (q.ρσ / 2) .* randn(N)
    
    νQ_c = (2e * 0.0845e-28u"m^2" / (2h/3 * I₀ * (2 * I₀ - 1)))
    
    νQs = abs.(U1 .* νQ_c) .|> u"MHz"
    ηs = -√3 * (q.η * q.Vzz / 2√3) ./ U1 .+ (q.ρσ / 2) .* randn(N) ./ U1

    m_vec = map(m -> I₀ * (I₀ + 1) - m * (m - 1), (-I₀ + 1):I₀)
    ms = get_m.(rand(1:sum(m_vec), N), Ref(m_vec), I₀)
    
    return get_ν2.(νQs, ηs, μs, λs, ms, I₀, exp.ν₀) .+
           get_ν3.(νQs, ηs, μs, λs, ms, I₀, exp.ν₀)
end

"""
    estimate_mas_powder_pattern(q, N, exp)

Get the estimated powder pattern (a vector of N frequencies) given the 
quadrupolar interaction and the ExperimentalSpectrum

"""
@inline estimate_mas_powder_pattern(
    q::Quadrupolar, 
    N::Int, 
    exp::ExperimentalSpectrum
) = estimate_mas_powder_pattern(q, N, μ(N), λ(N), exp)