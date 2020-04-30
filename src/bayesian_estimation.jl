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
function likelihood(yhat, experimental_ecdf, experimental, I, ν0)
    # We must choose a distribution for the error!
    likelihood_dist = Normal(0, max(0, yhat[3]))

    # Here we're generating the theoretical sample and converting it to a CDF
    powder_pattern = estimate_powder_pattern(yhat[1], yhat[2], 100_000, ν0, I)
    theoretical_ecdf = ecdf(powder_pattern).(experimental[:, 1])

    # Then we return the likelihood, based on two CDFs' differences
    return sum(logpdf.(likelihood_dist, experimental_ecdf .- theoretical_ecdf))
end

"""
    metropolis_hastings(experimental)

An implementation of the Metropolis-Hastings algorithm for Monte Carlo Markov
Chain (MCMC) Bayesian estimates, with optional keyword arguments for the number
of samples (default 1,000,000) and tolerances for each parameter (default 0.1
for qcc, 0.1 for η, 0.05 for σ)
    
"""
function metropolis_hastings(
    experimental;
    N = 1_000_000,
    tol = [0.1, 0.1, 0.05],
)
    I = 3
    experimental_ecdf = cumsum(experimental[:, 2]) ./ sum(experimental[:, 2])
    riemann_sum = 0
    for i = 2:length(experimental_ecdf)
        riemann_sum += (experimental_ecdf[i]) *
                       (experimental[i, 1] - experimental[i-1, 1])
    end
    ν0 = experimental[end, 1] - riemann_sum
    samples = zeros(N, 3)

    #We need to define prior distributions for each parameter
    prior_dist_qcc = Uniform(0, 9)
    prior_dist_η = Uniform(0, 1)
    prior_dist_σ = Uniform(0, 1)

    # Assume variables are independent
    prior(x) = logpdf(prior_dist_qcc, x[1]) + logpdf(prior_dist_η, x[2]) +
    logpdf(prior_dist_σ, x[3])

    a = [rand(prior_dist_qcc), rand(prior_dist_η), rand(prior_dist_σ)]
    samples[1, :] = a

    # Repeat N times
    @showprogress for i = 2:N

        # Compute new state randomly
        b = a + tol .* (rand(3) .- 0.5)

        # Calculate density
        prob_old = likelihood(a, experimental_ecdf, experimental, I, ν0) +
                   prior(a)
        prob_new = likelihood(b, experimental_ecdf, experimental, I, ν0) +
                   prior(b)

        # Compute acceptance ratio
        r = prob_new - prob_old

        if log(rand()) < r
        #Accept new state and update
            a = b
        end

        # Update state
        samples[i, :] = a
    end

    return samples
end;
