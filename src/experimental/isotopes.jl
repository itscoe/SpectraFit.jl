using PeriodicTable, Unitful

"""
    Isotope

An isotope is composed of the element (Z), and the total amount of protons and 
neutrons (A)

# Fields
- `Z`
- `A`
"""
struct Isotope
    Z::Element
    A::Int64
end

"""
    I(isotope)

Return the spin of the isotope (currently hard-coded and returns a rational)

"""
I(isotope::Isotope) = 
    isotope.Z.number == 5  && isotope.A == 10  ? 3//1 : 
    isotope.Z.number == 5  && isotope.A == 11  ? 3//2 : 
    isotope.Z.number == 82 && isotope.A == 207 ? 1//2 : 
                                                 0//1

"""
    γ(isotope)

Return the gyromagnetic ratio of the isotope from: 

>Harris, Robin K., Edwin D. Becker, Sonia M. Cabral De Menezes, 
>    Robin Goodfellow, and Pierre Granger. "NMR nomenclature. 
>    Nuclear spin properties and conventions for chemical shifts 
>    (IUPAC Recommendations 2001)." Pure and Applied Chemistry 73, 
>    no. 11 (2001): 1795-1818.

"""
γ(isotope::Isotope) = 
    isotope.Z.number == 5  && isotope.A == 10  ? (28.746786/2π)u"MHz/T" :
    isotope.Z.number == 5  && isotope.A == 11  ? (85.847044/2π)u"MHz/T" : 
    isotope.Z.number == 82 && isotope.A == 207 ? (55.8046/2π)u"MHz/T" : 
                                                 0u"MHz/T"

"""
    Q(isotope)

Return the quadrupole moment of the isotope from: 

>Stone, N. J. "Table of nuclear electric quadrupole moments."
>    Atomic Data and Nuclear Data Tables 111 (2016): 1-28.

"""
Q(isotope::Isotope) = 
    isotope.Z.number == 5 && isotope.A == 10 ? 0.0845e-28u"m^2" : 
    0.0u"m^-2"
