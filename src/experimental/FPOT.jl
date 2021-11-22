# FPOT are "Fractional (Multiples of) Powers of Two

struct FPOT
    n::Int8
    d::UInt8
end

HalfInteger(n::Int, d::Int) = HalfInteger(Int8(n), UInt8(d))
HalfInteger(n::Int) = HalfInteger(Int8(n), 0x00)

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

# Conversion to Float64
Base.Float64(a::FPOT) = a.n / (0x01 << a.d)