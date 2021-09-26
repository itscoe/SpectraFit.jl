module SpectraFit

include("units/constants.jl")
include("units/ppm.jl")

include("experimental/isotopes.jl")
include("experimental/data.jl")

include("theoretical/euler_angles.jl")
include("theoretical/interactions/NMRInteraction.jl")
include("theoretical/interactions/Zeeman.jl")
include("theoretical/interactions/Dipolar.jl")
include("theoretical/interactions/Quadrupolar.jl")
include("theoretical/interactions/ChemicalShift.jl")
include("theoretical/Spectra.jl")

include("ABC/smc.jl")

include("visualization/plot.jl")

export ExperimentalSpectra,
       get_example_data,
       get_data,
       Zeeman,
       Dipolar,
       Quadrupolar,
       ChemicalShiftI, 
       Spectra,
       abc_smc

end # module
