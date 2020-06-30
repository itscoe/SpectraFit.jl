struct chemical_shift_params
    σᵢₛₒ::Normal
    Δσ::Normal
    ησ::Truncated{Normal{Float64},Continuous,Float64}
end
