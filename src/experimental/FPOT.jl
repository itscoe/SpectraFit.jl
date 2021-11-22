# FPOT are "Fractional (Multiples of) Powers of Two"

struct FPOT <: Real
    n::Int16   # numerator
    d::UInt16  # denominator power of 2
end

FPOT(n::Int, d::Int) = FPOT(Int16(n), UInt16(d))
FPOT(n::Int) = FPOT(Int16(n), 0x0000)

# Addition
@inline Base.:+(a::FPOT, b::FPOT) = a.d == b.d ? FPOT(a.n + b.n, a.d) : 
    FPOT(a.n * (Int16(1) << b.d) + b.n * (Int16(1) << a.d), a.d + b.d)
@inline Base.:+(a::FPOT, b::Int64) = FPOT(a.n + b * (Int16(1) << a.d), a.d)
@inline Base.:+(a::FPOT, ::Val{1}) = FPOT(a.n + Int16(1) << a.d, a.d)
@inline Base.:+(a::FPOT, ::Val{FPOT(3, 1)}) = a.d == 1 ? FPOT(a.n + 3, a.d) : 
    FPOT(a.n * Int16(2) + 3 * (Int16(1) << a.d), a.d + 1)
@inline Base.:+(b::Int64, a::FPOT) = a + b

# Subtraction
@inline Base.:-(a::FPOT) = FPOT(-a.n, a.d)
@inline Base.:-(a::FPOT, b::FPOT) = a.d == b.d ? FPOT(a.n - b.n, a.d) : 
    FPOT(a.n * (Int16(1) << b.d) - b.n * (Int16(1) << a.d), a.d + b.d)
@inline Base.:-(a::FPOT, b::Int64) = FPOT(a.n - b * (Int16(1) << a.d), a.d)
@inline Base.:-(b::Int64, a::FPOT) = FPOT(b * (Int16(1) << a.d) - a.n, a.d)
@inline Base.:-(a::FPOT, ::Val{1}) = FPOT(a.n - (Int16(1) << a.d), a.d)
@inline Base.:-(a::FPOT, ::Val{FPOT(1, 1)}) = a.d == 0x0001 ? 
    FPOT(a.n + Int16(1), a.d) : 
    FPOT(a.n * 0x0002 - (Int16(1) << a.d), a.d + 1)

# Multiplication
@inline Base.:*(a::FPOT, b::FPOT) = FPOT(a.n * b.n, a.d + b.d)
@inline Base.:*(a::FPOT, b::Int64) = FPOT(a.n * b, a.d)
@inline Base.:*(a::FPOT, ::Val{2}) = FPOT(a.n + a.n, a.d)
@inline Base.:*(a::FPOT, ::Val{3}) = FPOT(a.n + a.n + a.n, a.d)
@inline Base.:*(a::FPOT, ::Val{4}) = FPOT(a.n + a.n + a.n + a.n, a.d)
@inline Base.:*(a::FPOT, ::Val{5}) = FPOT(a.n + a.n + a.n + a.n + a.n, a.d)
@inline Base.:*(a::FPOT, ::Val{6}) = FPOT(a.n + a.n + a.n + a.n + a.n + a.n, 
    a.d)
@inline Base.:*(b::Int64, a::FPOT) = a * b

# Exponentiation
@inline Base.:^(a::FPOT, ::Val{2}) = FPOT(a.n * a.n, a.d + a.d)

# Conversion
Base.Float64(a::FPOT) = a.n / (0x0001 << a.d)
Base.Int64(a::FPOT) = a.d == 0x0000 ? a.n : throw(InexactError)
Base.Integer(a::FPOT) = a.d == 0x0000 ? a.n : throw(InexactError)

# Comparison
Base.:<(a::FPOT, b::FPOT) = a.d == b.d ? a.n < b.n : Float64(a) < Float64(b)
Base.:<=(a::FPOT, b::FPOT) = a.d == b.d ? a.n <= b.n : Float64(a) <= Float64(b)
Base.:<(a::FPOT, b::Int64) = a.d == 0x0000 ? a.n < b : Float64(a) < b
Base.:<(b::Int64, a::FPOT) = a.d == 0x0000 ? b < a.n : b < Float64(a)
Base.:<=(a::FPOT, b::Int64) = a.d == 0x0000 ? a.n <= b : Float64(a) <= b
Base.:<=(b::Int64, a::FPOT) = a.d == 0x0000 ? b <= a.n : b <= Float64(a)

# Rounding
Base.round(a::SpectraFit.FPOT, _::RoundingMode{:Down}) = 
    FPOT(a.n ÷ (Int16(1) << a.d), 0x0000)