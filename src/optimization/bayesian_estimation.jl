using ProgressMeter

"""
    likelihood(yhat, experimental_ecdf, experimental, I, ν0)

Find the likelihood of a specific set of parameters (yhat), which is based on
the ordinary least squares of the experimental vs. theoretical CDFs

# Arguments
- `yhat`: an array containing parameters for qcc, η, and the standard deviation
- `experimental_ecdf`: the estimated ECDF of the experimental data
- `experimental`: the experimental data
- `I`: spin (3 in the case of Boron-10)
- `ν0`: the Larmor frequency

"""
function likelihood_quad(ŷ, experimental_ecdf, experimental, I, ν0)
    sites = length(ŷ) ÷ 5 - 1

    # We must choose a distribution for the error!
    ŷ[end] < 0 && return 0
    likelihood_dist = Normal(0, ŷ[end])

    N = 100_000
    powder_pattern = zeros(N)
    if sites == 1
        powder_pattern .= estimate_powder_pattern(Quadrupolar(ŷ[1], ŷ[2], ŷ[3],
            ŷ[4]), N, ν0, I)
    else
        if sites == 2
            weights = [ŷ[5]]
        else
            weights = ŷ[5:5:end-1]
        end
        w = vcat([0], floor.(Int64, cumsum(weights) .* N), [N])
        for i = 1:sites
            powder_pattern[w[i]+1:w[i+1]] = estimate_powder_pattern(map(i ->
                Quadrupolar(ŷ[5*(i-1)+1], ŷ[5*(i-1)+2], ŷ[5*(i-1)+3],
                ŷ[5*(i-1)+4]), 1:sites), w[i+1] - w[i], ν0, I)
        end
    end

    # Here we're generating the theoretical sample and converting it to a CDF
    th_ecdf = ecdf(powder_pattern).(experimental[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]

    # Then we return the likelihood, based on two CDFs' differences
    return sum(logpdf.(likelihood_dist, experimental_ecdf .- th_ecdf))
end

function likelihood_CSA(ŷ, experimental_ecdf, experimental)
    sites = length(ŷ) ÷ 7 - 1

    # We must choose a distribution for the error!
    ŷ[end] < 0 && return 0
    likelihood_dist = Normal(0, ŷ[end])

    N = 100_000
    powder_pattern = zeros(N)
    if sites == 1
        powder_pattern .= estimate_powder_pattern(ChemicalShift(ŷ[1], ŷ[2], ŷ[3],
            ŷ[4], ŷ[5], ŷ[6]), N)
    else
        if sites == 2
            weights = [ŷ[7]]
        else
            weights = ŷ[7:7:end-1]
        end
        w = vcat([0], floor.(Int64, cumsum(weights) .* N), [N])
        for i = 1:sites
            powder_pattern[w[i]+1:w[i+1]] = estimate_powder_pattern(map(i ->
                ChemicalShift(ŷ[7*(i-1)+1], ŷ[7*(i-1)+2], ŷ[7*(i-1)+3],
                ŷ[7*(i-1)+4], ŷ[7*(i-1)+5], ŷ[7*(i-1)+6]), 1:sites), w[i+1] -
                w[i])
        end
    end

    # Here we're generating the theoretical sample and converting it to a CDF
    th_ecdf = ecdf(powder_pattern).(experimental[:, 1])
    th_ecdf .-= th_ecdf[1]
    th_ecdf ./= th_ecdf[end]

    # Then we return the likelihood, based on two CDFs' differences
    return sum(logpdf.(likelihood_dist, experimental_ecdf .- th_ecdf))
end

"""
    metropolis_hastings(experimental)

An implementation of the Metropolis-Hastings algorithm for Monte Carlo Markov
Chain (MCMC) Bayesian estimates, with optional keyword arguments for the number
of samples (default 1,000,000) and tolerances for each parameter (default 0.1
for qcc, 0.1 for η, 0.05 for σ)

"""
function mh_quad(experimental;
    N = 1_000_000,
    sites = 1,
    I = 3,
    lb = quadrupolar_lb(sites),
    ub = quadrupolar_ub(sites),
    tol = quadrupolar_tol(sites),
)
    experimental_ecdf = get_experimental_ecdf(experimental)
    ν0 = get_ν0(experimental, experimental_ecdf)

    samples = zeros(N, 5 * sites)  # initialize zero array for samples

    prior_dist_qcc, prior_dist_σqcc, prior_dist_η, prior_dist_ση =
        Uniform(lb[1], ub[1]), Uniform(lb[2], ub[2]),
        Uniform(lb[3], ub[3]), Uniform(lb[4], ub[4])
        if sites != 1
            prior_dist_w = Uniform(lb[5], ub[5])
        end
        prior_dist_σ = Uniform(0, 1)

    prior_quad(x, sites) = sites == 1 ? sum(map(i ->
        logpdf(prior_dist_qcc, x[5*(i-1)+1]) +
        logpdf(prior_dist_σqcc, x[5*(i-1)+2]) +
        logpdf(prior_dist_η, x[5*(i-1)+3]) +
        logpdf(prior_dist_ση, x[5*(i-1)+4]), 1:sites)) +
        logpdf(prior_dist_σ, x[end]) :
        sum(map(i ->
        logpdf(prior_dist_qcc, x[5*(i-1)+1]) +
        logpdf(prior_dist_σqcc, x[5*(i-1)+2]) +
        logpdf(prior_dist_η, x[5*(i-1)+3]) +
        logpdf(prior_dist_ση, x[5*(i-1)+4]), 1:sites)) +
        sum(map(i -> logpdf(prior_dist_w, x[5*(i-1)+5]), 1:sites-1)) +
        logpdf(prior_dist_σ, x[end])

    a = vcat(quadrupolar_starting_value(sites), [rand(prior_dist_σ)])
    samples[1, :] = a

    @showprogress for i = 2:N
        b = a + tol .* (rand(5*sites) .- 0.5)  # Compute new state randomly
        # Calculate density
        prob_old = likelihood_quad(a, experimental_ecdf, experimental, I, ν0) +
                   prior_quad(a, sites)
        prob_new = likelihood_quad(b, experimental_ecdf, experimental, I, ν0) +
                   prior_quad(b, sites)
        r = prob_new - prob_old # Compute acceptance ratio
        if log(rand()) < r
            a = b  # Accept new state and update
        end
        samples[i, :] = a  # Update state
    end

    return samples
end

function mh_chemical_shift(experimental;
    N = 1_000_000,
    sites = 1,
    lb = CSA_lb(sites),
    ub = CSA_ub(sites),
    tol = CSA_tol(sites),
)
    experimental_ecdf = get_experimental_ecdf(experimental)
    ν0 = get_ν0(experimental, experimental_ecdf)

    samples = zeros(N, 7 * sites)  # initialize zero array for samples

    prior_dist_δᵢₛₒ, prior_dist_σδᵢₛₒ, prior_dist_Δδ, prior_dist_σΔδ,
        prior_dist_ηδ, prior_dist_σηδ =
        Uniform(lb[1], ub[1]), Uniform(lb[2], ub[2]),
        Uniform(lb[3], ub[3]), Uniform(lb[4], ub[4]),
        Uniform(lb[5], ub[5]), Uniform(lb[6], ub[6])
    if sites != 1
        prior_dist_w = Uniform(lb[7], ub[7])
    end
    prior_dist_σ = Uniform(0, 1)

    prior_quad(x, sites) = sites == 1 ?
        sum(map(i -> logpdf(prior_dist_δᵢₛₒ, x[7*(i-1)+1]) +
        logpdf(prior_dist_σδᵢₛₒ, x[7*(i-1)+2]) +
        logpdf(prior_dist_Δδ, x[7*(i-1)+3]) +
        logpdf(prior_dist_σΔδ, x[7*(i-1)+4]) +
        logpdf(prior_dist_ηδ, x[7*(i-1)+5]) +
        logpdf(prior_dist_ση, x[7*(i-1)+6]), 1:sites)) +
        logpdf(prior_dist_σηδ, x[end]) :
        sum(map(i -> logpdf(prior_dist_δᵢₛₒ, x[7*(i-1)+1]) +
        logpdf(prior_dist_σδᵢₛₒ, x[7*(i-1)+2]) +
        logpdf(prior_dist_Δδ, x[7*(i-1)+3]) +
        logpdf(prior_dist_σΔδ, x[7*(i-1)+4]) +
        logpdf(prior_dist_ηδ, x[7*(i-1)+5]) +
        logpdf(prior_dist_σηδ, x[7*(i-1)+6]), 1:sites)) +
        sum(map(i -> logpdf(prior_dist_w, x[7*(i-1)+7]), 1:sites-1)) +
        logpdf(prior_dist_σ, x[end])

    a = vcat(CSA_starting_value(sites), [rand(prior_dist_σ)])
    samples[1, :] = a

    @showprogress for i = 2:N
        b = a + tol .* (rand(7*sites) .- 0.5)  # Compute new state randomly
        # Calculate density
        prob_old = likelihood_CSA(a, experimental_ecdf, experimental) +
                   prior_quad(a, sites)
        prob_new = likelihood_CSA(b, experimental_ecdf, experimental) +
                   prior_quad(b, sites)
        r = prob_new - prob_old # Compute acceptance ratio
        if log(rand()) < r
            a = b  # Accept new state and update
        end
        samples[i, :] = a  # Update state
    end

    return samples
end
