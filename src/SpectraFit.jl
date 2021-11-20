module SpectraFit

# install NMRGlue
ENV["PYTHON"] = "";
using PyCall, Conda
Conda.add("nmrglue", channel = "spectrocat")
nmrglue = pyimport("nmrglue")

include("units/constants.jl")
include("units/ppm.jl")

include("experimental/isotopes.jl")
include("experimental/ExperimentalSpectrum.jl")
include("experimental/ExperimentalSeries.jl")

include("theoretical/euler_angles.jl")
include("theoretical/interactions/NMRInteraction.jl")
include("theoretical/interactions/Zeeman.jl")
include("theoretical/interactions/Dipolar.jl")
include("theoretical/interactions/Quadrupolar.jl")
include("theoretical/interactions/ChemicalShiftI.jl")
include("theoretical/interactions/ChemicalShiftA.jl")
include("theoretical/Spectrum.jl")

include("ABC/smc.jl")

include("visualization/plot.jl")

export ExperimentalSpectrum,
       ExperimentalSeries,
       get_example_data,
       get_data,
       Zeeman,
       Dipolar,
       Quadrupolar,
       ChemicalShiftI, 
       ChemicalShiftA,
       Spectrum,
       abc_smc,
       plot_parameters,
       plot_fits, 
       Isotope,
       elements

end # module
