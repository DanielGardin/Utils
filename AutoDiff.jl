struct Dual
    primal
    dual
end

const ϵ = Dual(0, 1)

Base.length(::Dual) = 1
Base.eltype(::Dual) = Dual

Base.one(::Type{Dual}) = Dual(1, 0)
Base.zero(::Type{Dual}) = Dual(0, 0)
Base.one(::Dual) = Dual(1, 0)
Base.zero(::Dual) = Dual(0, 0)

primal(x::Dual) = x.primal
primal(x::Real) = x

dual(x::Dual) = x.dual
dual(x::Real) = x

Dual(x::Real) = Dual(x, 0)

Base.:+(d::Dual) = d
Base.:-(d::Dual) = Dual(-d.primal, -d.dual)


Base.:+(a::Dual, b::Dual) = Dual(a.primal + b.primal, a.dual + b.dual)
Base.:-(a::Dual, b::Dual) = Dual(a.primal - b.primal, a.dual - b.dual)


Base.:+(x, d::Dual) = Dual(x + d.primal, d.dual)
Base.:-(x, d::Dual) = Dual(x - d.primal, -d.dual)
Base.:+(d::Dual, x) = Dual(x + d.primal, d.dual)
Base.:-(d::Dual, x) = Dual(d.primal - x, d.dual)

Base.:*(a::Dual, b::Dual) = Dual(a.primal * b.primal, a.primal * b.dual + a.dual * b.primal)
Base.:*(x::T, d::Dual) where T<:Real = Dual(x * d.primal, x * d.dual)
Base.:*(d::Dual, x::T) where T<:Real = Dual(x * d.primal, x * d.dual)

Base.:/(a::Dual, b::Dual) = Dual(a.primal/b.primal, (a.dual * b.primal- a.primal * b.dual)/b.primal^2)
Base.:/(x, d::Dual) = Dual(x/d.primal, -x*d.dual/d.primal^2)
Base.:/(d::Dual, x) = Dual(d.primal/x, d.dual/x)

Base.sqrt(d::Dual) = Dual(sqrt(d.primal), sqrt(d.dual^2/d.primal)/2)
Base.exp(d::Dual) = Dual(exp(d.primal), d.dual * exp(d.primal))
Base.log(d::Dual) = Dual(log(d.primal), d.dual/d.primal)
Base.cos(d::Dual) = Dual(cos(d.primal), -d.dual * sin(d.primal))
Base.sin(d::Dual) = Dual(sin(d.primal), d.dual * cos(d.primal))
Base.tan(d::Dual) = sin(d)/cos(d)
Base.:^(d::Dual, n::Integer) = Dual(d.primal^n, d.dual * n * d.primal^(n-1))

Base.isless(a::Dual, b::Dual) = a.primal < b.primal
Base.isless(x, d::Dual) = x < d.primal
Base.isless(d::Dual, x) = d.primal < x
Base.isequal(a::Dual, b::Dual) = a.primal == b.primal && a.dual == b.dual

derivative(f::Function) = x -> f(x + ϵ).dual
derivative(f::Function, x::Real) = f(x + ϵ).dual

function gradient(f::Function, x)
    ∇f = zeros(eltype(x), length(x))

    for pos in eachindex(x)
        dx = Dual.(x)
        dx[pos] += ϵ

        ∇f[pos] = f(dx...).dual
    end

    ∇f
end

∇(f::Function, x) = gradient(f, x)

function Base.show(io::IO, d::Dual)
    print(io, "$(d.primal) + $(d.dual)ϵ")
end
