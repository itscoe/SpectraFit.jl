using Unitful

struct Zeeman <: NMRInteraction end

Base.length(_::Zeeman) = 1
Base.size(_::Zeeman) = (0,)

estimate_powder_pattern(_::Zeeman, N::Int, ν₀::typeof(1.0u"MHz")) = 
    ν₀ .* ones(N)

estimate_powder_pattern(_::Zeeman, N::Int, ν₀) = 
    (ν₀ |> u"mHz") .* ones(N)