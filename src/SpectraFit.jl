module SpectraFit

ENV["GKS_ENCODING"] = "utf-8"

include("Quadrupolar.jl")
include("ChemicalShift.jl")
include("utility_functions.jl")
include("forward_model.jl")
include("optim_estimation.jl")
include("bayesian_estimation.jl")

export get_experimental,
       Quadrupolar,
       fit_nmr,
       metropolis_hastings,
       compare_theoreticals,
       get_data

end # module
