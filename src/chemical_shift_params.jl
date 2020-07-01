struct chemical_shift_params
    σᵢₛₒ::Normal
    Δσ::Normal
    ησ::Truncated{Normal{Float64},Continuous,Float64}
end

function chemical_shift_params(p::Array{Float64})
    return chemical_shift_params(Normal(p[1], max(0.0, p[2])), 
        Normal(p[3], max(0.0, p[4])),
        truncated(Normal(clamp(p[5], 0.001, 0.999), max(0.0, p[6])), 0.0, 1.0))
end
