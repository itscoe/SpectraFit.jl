# FPOT are "Fractional (Multiples of) Powers of Two"

struct FPOT <: Real
    n::Int8   # numerator
    d::UInt8  # denominator power of 2
end

FPOT(n::Int, d::Int) = FPOT(Int8(n), UInt8(d))
FPOT(n::Int) = FPOT(Int8(n), 0x00)

# Addition
@inline Base.:+(a::FPOT, b::FPOT) = 
    a.d == b.d ? FPOT(a.n + b.n, a.d) : 
                 FPOT(a.n * (0x01 << b.d) + b.n * (0x01 << a.d), a.d + b.d)
@inline Base.:+(a::FPOT, b::Int64) = FPOT(a.n + b * (0x01 << a.d), a.d)
@inline Base.:+(a::FPOT, ::Val{1}) = FPOT(a.n + 0x01 << a.d, a.d)
@inline Base.:+(a::FPOT, ::Val{FPOT(3, 1)}) = a.d == 1 ? FPOT(a.n + 3, a.d) : 
    FPOT(a.n * 0x02 + 3 * (0x01 << a.d), a.d + 1)
@inline Base.:+(b::Int64, a::FPOT) = a + b

# Subtraction
@inline Base.:-(a::FPOT) = FPOT(-a.n, a.d)
@inline Base.:-(a::FPOT, b::FPOT) = 
    a.d == b.d ? FPOT(a.n + b.n, a.d) : 
                 FPOT(a.n * (0x01 << b.d) - b.n * (0x01 << a.d), a.d + b.d)
@inline Base.:-(a::FPOT, b::Int64) = FPOT(a.n - b * (0x01 << a.d), a.d)
@inline Base.:-(a::FPOT, b::Int64) = FPOT(b * (0x01 << a.d) - a.n, a.d)
@inline Base.:-(a::FPOT, ::Val{1}) = FPOT(a.n - (0x01 << a.d), a.d)
@inline Base.:-(a::FPOT, ::Val{FPOT(1, 1)}) = a.d == 1 ? FPOT(a.n + 1, a.d) : 
    FPOT(a.n * 0x02 - (0x01 << a.d), a.d + 1)

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
Base.Float64(a::FPOT) = a.n / (0x01 << a.d)
Base.Int64(a::FPOT) = a.d == 0x00 ? a.n : throw(InexactError)
Base.Integer(a::FPOT) = a.d == 0x00 ? a.n : throw(InexactError)

# Comparison
Base.:<(a::FPOT, b::FPOT) = a.d == b.d ? a.n < b.n : Float64(a) < Float64(b)
Base.:<=(a::FPOT, b::FPOT) = a.d == b.d ? a.n <= b.n : Float64(a) <= Float64(b)
Base.:<(a::FPOT, b::Int64) = a.d == 0x01 ? a.n < b : Float64(a) < b
Base.:<(b::Int64, a::FPOT) = a.d == 0x01 ? b < a.n : b < Float64(a)
Base.:<=(a::FPOT, b::Int64) = a.d == 0x01 ? a.n <= b : Float64(a) <= b
Base.:<=(b::Int64, a::FPOT) = a.d == 0x01 ? b <= a.n : b <= Float64(a)

# Rounding
Base.round(a::SpectraFit.FPOT, _::RoundingMode{:Down}) = 
    FPOT(a.n ÷ (0x01 << a.d), 0x00)