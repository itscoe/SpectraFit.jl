module SpectraFit

ENV["GKS_ENCODING"] = "utf-8"

include("interactions/NMRInteraction.jl")
include("interactions/Quadrupolar.jl")
include("interactions/ChemicalShift.jl")
include("data.jl")
include("optimization/bounds.jl")
include("optimization/objective_function.jl")
include("optimization/optimizers.jl")
include("optimization/bayesian_estimation.jl")
include("plot.jl")

export get_experimental,
       Quadrupolar,
       ChemicalShift,
       quadrupolar_opt,
       metropolis_hastings,
       compare_theoreticals,
       get_data,
       plot_chemical_shift

end # module
