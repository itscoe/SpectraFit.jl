module SpectraFit

include("units/constants.jl")
include("units/ppm.jl")

include("experimental/isotopes.jl")
include("experimental/data.jl")

include("theoretical/interactions/NMRInteraction.jl")
include("theoretical/interactions/Zeeman.jl")
include("theoretical/interactions/Dipolar.jl")
include("theoretical/interactions/Quadrupolar.jl")
include("theoretical/interactions/ChemicalShift.jl")

include("theoretical/Spectra.jl")

include("ABC/smc.jl")

include("visualization/plot.jl")

export get_experimental,
       get_data,
       Zeeman,
       Dipolar,
       Quadrupolar,
       ChemicalShift, 
       Spectra

end # module
