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
    yhat[5] < 0 && return 0
    likelihood_dist = Normal(0, yhat[5])

    # Here we're generating the theoretical sample and converting it to a CDF
    quad = Quadrupolar(yhat[1:4])
    ismissing(quad) && return 0
    powder_pattern = estimate_powder_pattern(quad, 100_000, ν0, I)
    theoretical_ecdf = ecdf(powder_pattern).(experimental[:, 1])
    theoretical_ecdf .-= theoretical_ecdf[1]
    theoretical_ecdf ./= theoretical_ecdf[end]
    # Then we return the likelihood, based on two CDFs' differences
    return sum(logpdf.(likelihood_dist, experimental_ecdf .- theoretical_ecdf))
end

function likelihood(yhat, experimental_ecdf, experimental)
    # We must choose a distribution for the error!
    yhat[7] < 0 && return 0
    likelihood_dist = Normal(0, yhat[7])

    # Here we're generating the theoretical sample and converting it to a CDF
    csa = ChemicalShift(vcat(yhat[1:6], [1.0]))
    ismissing(csa) && return 0
    powder_pattern = estimate_powder_pattern(csa, 100_000)
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
    interaction = "quadrupolar",
    N = 1_000_000,
    tol = interaction == "quadrupolar" ? [0.1, 0.2, 0.1, 0.1, 0.05] : [10.0, 5.0, 10.0, 5.0, 0.1, 0.05, 0.05],
    I = 3,
)
    experimental_ecdf = get_experimental_ecdf(experimental)
    if interaction == "quadrupolar"
        ν0 = get_ν0(experimental, experimental_ecdf)

        samples = zeros(N, 5)  # initialize zero array for samples

        #We need to define prior distributions for each parameter
        prior_dist_qcc = Uniform(0, 9)
        prior_dist_σqcc = Uniform(0, 3)
        prior_dist_η = Uniform(0, 1)
        prior_dist_ση = Uniform(0, 1)
        prior_dist_σ = Uniform(0, 1)

        prior_quad(x) = logpdf(prior_dist_qcc, x[1]) + logpdf(prior_dist_σqcc, x[2]) +
            logpdf(prior_dist_η, x[3]) + logpdf(prior_dist_ση, x[4]) +
            logpdf(prior_dist_σ, x[5]) # Assume variables are independent

        a = [rand(prior_dist_qcc), rand(prior_dist_σqcc), rand(prior_dist_η),
            rand(prior_dist_ση),rand(prior_dist_σ)]
        samples[1, :] = a

        # Repeat N times
        @showprogress for i = 2:N
            b = a + tol .* (rand(5) .- 0.5)  # Compute new state randomly
            # Calculate density
            prob_old = likelihood(a, experimental_ecdf, experimental, I, ν0) +
                       prior_quad(a)
            prob_new = likelihood(b, experimental_ecdf, experimental, I, ν0) +
                       prior_quad(b)
            r = prob_new - prob_old # Compute acceptance ratio
            if log(rand()) < r
                a = b  # Accept new state and update
            end
            samples[i, :] = a  # Update state
        end
    else
        samples = zeros(N, 7)  # initialize zero array for samples

        #We need to define prior distributions for each parameter
        prior_dist_σᵢₛₒ = Uniform(-4000, 4000)
        prior_dist_σσᵢₛₒ = Uniform(0.00001, 800)
        prior_dist_Δσ = Uniform(-4000, 4000)
        prior_dist_σΔσ = Uniform(0.00001, 400)
        prior_dist_ησ = Uniform(0, 1)
        prior_dist_σησ = truncated(Normal(0.00001, 0.2), 0.0, Inf)
        prior_dist_σ = truncated(Normal(0.00001, 0.5), 0.0, Inf)

        prior_csa(x) = logpdf(prior_dist_σᵢₛₒ, x[1]) + logpdf(prior_dist_σσᵢₛₒ, x[2]) +
            logpdf(prior_dist_Δσ, x[3]) + logpdf(prior_dist_σΔσ, x[4]) +
            logpdf(prior_dist_ησ, x[5]) + logpdf(prior_dist_σησ, x[6]) +
            logpdf(prior_dist_σ, x[7]) # Assume variables are independent

        a = [rand(prior_dist_σᵢₛₒ), rand(prior_dist_σσᵢₛₒ), rand(prior_dist_Δσ),
            rand(prior_dist_σΔσ), rand(prior_dist_ησ), rand(prior_dist_σησ),
            rand(prior_dist_σ)]
        samples[1, :] = a

        # Repeat N times
        @showprogress for i = 2:N
            b = a + tol .* (rand(7) .- 0.5)  # Compute new state randomly
            # Calculate density
            prob_old = likelihood(a, experimental_ecdf, experimental) + prior_csa(a)
            prob_new = likelihood(b, experimental_ecdf, experimental) + prior_csa(b)
            r = prob_new - prob_old # Compute acceptance ratio
            if log(rand()) < r
                a = b  # Accept new state and update
            end
            samples[i, :] = a  # Update state
        end
    end

    return samples
end;
