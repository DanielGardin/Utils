include("AutoDiff.jl")

const VecOrElement{T} = Union{Vector{T}, T}

const ReLU(x) = max(0, x)
const relu = ReLU
const σ(x) = 1/(1 + exp(-x))
const sigmoid = σ
const logistic = σ

"""

    AbstractLayer

Supertype for layer structure. A layer has the property of, given
an input and some parameters, return a output. It is possible to get
parameters from a layer with `parameters` method
"""
abstract type AbstractLayer end

mutable struct Dense{I, O} <: AbstractLayer
    weight::Matrix{Float64}
    bias::Vector{Float64}
    activation::Function
    id

    Dense((in, out)::Pair{Int, Int}, activation=identity; id="") = new{in, out}(rand(out, in), rand(out), activation, id)
end

parameters(D::Dense) = D.weight, D.bias

(D::Dense)(x) = D.activation.(D.weight * x .+ D.bias)