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

function likelihood(yhat, experimental_ecdf, experimental; sites = 1)
    if sites == 1
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
    elseif sites == 2
        # We must choose a distribution for the error!
        yhat[14] < 0 && return 0
        likelihood_dist = Normal(0, yhat[14])

        # Here we're generating the theoretical sample and converting it to a CDF
        csa = ChemicalShift(vcat(yhat[1:6], [yhat[13]], yhat[7:12], [1 - yhat[13]]))
        ismissing(csa) && return 0
        powder_pattern = estimate_powder_pattern(csa, 100_000)
        theoretical_ecdf = ecdf(powder_pattern).(experimental[:, 1])

        # Then we return the likelihood, based on two CDFs' differences
        return sum(logpdf.(likelihood_dist, experimental_ecdf .- theoretical_ecdf))
    end
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
    sites = 1,
    tol = interaction == "quadrupolar" ? [0.1, 0.2, 0.1, 0.1, 0.05] : sites == 1 ? [100.0, 20.0, 30.0, 5.0, 0.1, 0.05, 0.05] : [10.0, 5.0, 10.0, 5.0, 0.1, 0.05, 10.0, 5.0, 10.0, 5.0, 0.1, 0.05, 0.01, 0.05],
    I = 3
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
        if sites == 1
            samples = zeros(N, 7)  # initialize zero array for samples

            #We need to define prior distributions for each parameter
            prior_dist_σᵢₛₒ = Uniform(minimum(experimental[:, 1]), maximum(experimental[:, 1]))
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
        elseif sites == 2
            samples = zeros(N, 14)  # initialize zero array for samples

            #We need to define prior distributions for each parameter
            prior_dist_σᵢₛₒ1 = Uniform(minimum(experimental[:, 1]), maximum(experimental[:, 1]))
            prior_dist_σσᵢₛₒ1 = Uniform(0.00001, 800)
            prior_dist_Δσ1 = Uniform(-4000, 4000)
            prior_dist_σΔσ1 = Uniform(0.00001, 400)
            prior_dist_ησ1 = Uniform(0, 1)
            prior_dist_σησ1 = truncated(Normal(0.00001, 0.2), 0.0, Inf)
            prior_dist_σᵢₛₒ2 = Uniform(minimum(experimental[:, 1]), maximum(experimental[:, 1]))
            prior_dist_σσᵢₛₒ2 = Uniform(0.00001, 800)
            prior_dist_Δσ2 = Uniform(-4000, 4000)
            prior_dist_σΔσ2 = Uniform(0.00001, 400)
            prior_dist_ησ2 = Uniform(0, 1)
            prior_dist_σησ2 = truncated(Normal(0.00001, 0.2), 0.0, Inf)
            prior_dist_w = Uniform(0, 1)
            prior_dist_σ = truncated(Normal(0.00001, 0.5), 0.0, Inf)

            a = [rand(prior_dist_σᵢₛₒ1), rand(prior_dist_σσᵢₛₒ1), rand(prior_dist_Δσ1),
                rand(prior_dist_σΔσ1), rand(prior_dist_ησ1), rand(prior_dist_σησ1),
                rand(prior_dist_σᵢₛₒ2), rand(prior_dist_σσᵢₛₒ2), rand(prior_dist_Δσ2),
                rand(prior_dist_σΔσ2), rand(prior_dist_ησ2), rand(prior_dist_σησ2),
                rand(prior_dist_w),
                rand(prior_dist_σ)]
            samples[1, :] = a

            # Repeat N times
            @showprogress for i = 2:N
                b = a + tol .* (rand(14) .- 0.5)  # Compute new state randomly


                # Calculate density
                prob_old = likelihood(a, experimental_ecdf, experimental, sites = 2) +
                logpdf(prior_dist_σᵢₛₒ1, a[1]) + logpdf(prior_dist_σσᵢₛₒ1, a[2]) +
                   logpdf(prior_dist_Δσ1, a[3]) + logpdf(prior_dist_σΔσ1, a[4]) +
                   logpdf(prior_dist_ησ1, a[5]) + logpdf(prior_dist_σησ1, a[6]) +
                   logpdf(prior_dist_σᵢₛₒ2, a[7]) + logpdf(prior_dist_σσᵢₛₒ2, a[8]) +
                   logpdf(prior_dist_Δσ2, a[9]) + logpdf(prior_dist_σΔσ2, a[10]) +
                   logpdf(prior_dist_ησ2, a[11]) + logpdf(prior_dist_σησ2, a[12]) +
                   logpdf(prior_dist_w, a[13]) +
                   logpdf(prior_dist_σ, a[14])
                prob_new = likelihood(b, experimental_ecdf, experimental, sites = 2) +
                logpdf(prior_dist_σᵢₛₒ1, b[1]) + logpdf(prior_dist_σσᵢₛₒ1, b[2]) +
                   logpdf(prior_dist_Δσ1, b[3]) + logpdf(prior_dist_σΔσ1, b[4]) +
                   logpdf(prior_dist_ησ1, b[5]) + logpdf(prior_dist_σησ1, b[6]) +
                   logpdf(prior_dist_σᵢₛₒ2, b[7]) + logpdf(prior_dist_σσᵢₛₒ2, b[8]) +
                   logpdf(prior_dist_Δσ2, b[9]) + logpdf(prior_dist_σΔσ2, b[10]) +
                   logpdf(prior_dist_ησ2, b[11]) + logpdf(prior_dist_σησ2, b[12]) +
                   logpdf(prior_dist_w, b[13]) +
                   logpdf(prior_dist_σ, b[14])
                r = prob_new - prob_old # Compute acceptance ratio
                if log(rand()) < r
                    a = b  # Accept new state and update
                end
                samples[i, :] = a  # Update state
            end
        end
    end

    return samples
end;
