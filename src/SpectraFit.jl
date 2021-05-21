module SpectraFit

ENV["GKS_ENCODING"] = "utf-8"

include("Quadrupolar.jl")
include("ChemicalShift.jl")
include("utility_functions.jl")
include("optimization.jl")
include("bayesian_estimation.jl")

export get_experimental,
       Quadrupolar,
       ChemicalShift,
       quadrupolar_opt,
       metropolis_hastings,
       compare_theoreticals,
       get_data,
       plot_chemical_shift

end # module
