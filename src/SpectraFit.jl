module SpectraFit

include("nmr_params.jl")
include("utility_functions.jl")
include("forward_model.jl")
include("optim_estimation.jl")
include("bayesian_estimation.jl")

export get_experimental,
       nmr_params,
       fit_nmr,
       metropolis_hastings,
       compare_theoreticals

end # module
