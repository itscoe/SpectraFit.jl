module SpectraFit

ENV["GKS_ENCODING"] = "utf-8"

include("Quadrupolar.jl")
include("ChemicalShift.jl")
include("utility_functions.jl")
include("optim_estimation.jl")
include("bayesian_estimation.jl")

export get_experimental,
       Quadrupolar,
       ChemicalShift,
       fit_quadrupolar,
       fit_chemical_shift,
       metropolis_hastings,
       compare_theoreticals,
       get_data,
       plot_chemical_shift

end # module
