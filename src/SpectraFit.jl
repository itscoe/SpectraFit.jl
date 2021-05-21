module SpectraFit

ENV["GKS_ENCODING"] = "utf-8"

include("Quadrupolar.jl")
include("ChemicalShift.jl")
include("utility_functions.jl")
include("optimization.jl")
#include("bayesian_estimation.jl")
#include("genetic_algorithm.jl")
#include("bboptim_estimation.jl")

export get_experimental,
       Quadrupolar,
       ChemicalShift,
       fit_quadrupolar,
       fit_chemical_shift,
       metropolis_hastings,
       compare_theoreticals,
       get_data,
       plot_chemical_shift,
       get_output_table,
       genetic_algorithm,
       genetic_algorithm_csa,
       fit_quadrupolar_bb,
       fit_chemicalshift_bb

end # module
