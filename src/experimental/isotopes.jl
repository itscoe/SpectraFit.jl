using PeriodicTable, Unitful

struct Isotope
    Z::Element
    A::Int64
end

I(isotope::Isotope) = 
    isotope.Z.number == 5 && isotope.A == 10 ? 3//1 : 
    isotope.Z.number == 82 && isotope.A == 207 ? 1//2 : 
    0//1

γ(isotope::Isotope) = 
    isotope.Z.number == 5 && isotope.A == 10 ? (28.746786/2π)u"MHz/T" : 
    isotope.Z.number == 82 && isotope.A == 207 ? (55.8046/2π)u"MHz/T" : 
    0u"MHz/T"

# Stone, N. J. "Table of nuclear electric quadrupole moments." 
#     Atomic Data and Nuclear Data Tables 111 (2016): 1-28.
Q(isotope::Isotope) = 
    isotope.Z.number == 5 && isotope.A == 10 ? 0.0845e-28u"m^2" : 
    0.0u"m^-2"
