using Distributions

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

function estimate_powder_pattern(
    qcc::Float64,
    η::Float64,
    N::Int64,
    ν0::Float64,
    I::Int64,
)
    m = rand(Categorical([3, 5, 6, 6, 5, 3] ./ 28), N) .- I
    μ = rand(Uniform(0, 1), N)
    λ = rand(Uniform(-1, 1), N)
    return filter(isfinite, filter(!isnan, get_ν.(qcc, η, μ, λ, m, I, ν0)))
end

function estimate_powder_pattern(
    qcc_dist::Distribution,
    η_dist::Distribution,
    N::Int64,
    ν0::Float64,
    I::Int64,
)
    qcc = rand(qcc_dist, N)
    η = rand(η_dist, N)
    m = rand(Categorical([3, 5, 6, 6, 5, 3] ./ 28), N) .- I
    μ = rand(Uniform(0, 1), N)
    λ = rand(Uniform(-1, 1), N)
    return filter(isfinite, filter(!isnan, get_ν.(qcc, η, μ, λ, m, I, ν0)))
end

function estimate_powder_pattern(p::nmr_params, N::Int64, ν0::Float64, I::Int64)
    powder_pattern = zeros(N)
    i = 1
    for j = 1:(length(p.weights)-1)
        to_add = floor(Int, p.weights[j] * N)
        powder_pattern[i:(i+to_add-1)] = estimate_powder_pattern(
            p.qcc[j],
            p.η[j],
            to_add,
            ν0,
            I,
        )
        i += to_add
    end
    powder_pattern[i:end] = estimate_powder_pattern(
        p.qcc[length(p.weights)],
        p.η[length(p.weights)],
        N - i + 1,
        ν0,
        I,
    )
    return powder_pattern
end
