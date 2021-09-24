struct Dipolar <: NMRInteraction
    broadening::Float64
end

Base.size(_::Dipolar) = (1,)

Base.getindex(d::Dipolar, _::Int) = d.broadening