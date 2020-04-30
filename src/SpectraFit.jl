module SpectraFit

include("utility_functions.jl")
include("forward_model.jl")
include("optim_estimation.jl")
include("bayesian_estimation.jl")

export get_experimental,
       generate_theoretical_spectrum,
       nmr_params,
       fit_nmr,
       metropolis_hastings

end # module
