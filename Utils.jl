include("AutoDiff.jl")
#--------------------------------------------------------------------------------------------------
## Neural Networks

const VecOrElement{T} = Union{Vector{T}, T}

# Common activation Functions
const ReLU(x) = max(0, x)
const relu = ReLU
const σ(x) = 1/(1 + exp(-x))
const sigmoid = σ
const logistic = σ
# tanh(x) already defined
const step(x) = x < 0 ? 0 : 1

abstract type AbstractLayer end
mutable struct Dense{I, O} <: AbstractLayer
    weight::VecOrMat{Float64}
    bias::VecOrElement{Float64}
    activation::Function
    id

    Dense((in, out)::Pair{Int, Int}, activation=ReLU; id="", w_init=rand(out, in), b_init = rand(out)) = new{in, out}(w_init, b_init, activation, id)
    Dense(in, out, activation=ReLU; id="", w_init=rand(out, in), b_init = rand(out)) = new{in, out}(w_init, b_init, activation, id)
end

(L::Dense)(x) = L.weight * x .+ L.bias

function Base.size(L::Dense{I, O}) where {I, O}
    I, O
end

function Base.show(io::IO, L::Dense)
    if L.id == ""
        return print(io, "Dense($(size(L.weight, 2)) => $(length(L.bias)), $(L.activation))")
    end
    return print(io, "$(L.id) Dense Layer : $(size(L.weight, 2)) => $(length(L.bias)), $(L.activation)")
end

struct Softmax <: AbstractLayer
    τ::Float64
    activation

    Softmax(τ = 1) = new(τ, identity)
end

function (S::Softmax)(x)
    r = exp.(x ./ S.τ)

    return r ./ sum(r)
end

struct OneHot{N}
    _index::Int
end

struct Loss
    lossFunction
end

(L::Loss)(h, y) = sum(L.lossFunction.(h, y))

const SQLoss = Loss((h, y) -> (h - y)^2)
const CrossEntropy = Loss((h, y) -> -y*log(h)-(1 - y)*log(1-h))

abstract type AbstractNetwork end

mutable struct NeuralNetwork <: AbstractNetwork
    layers::Vector{AbstractLayer}
    loss::Loss
    learningRate::Float64
end

NeuralNetwork(η, loss::Loss=SQLoss) = NeuralNetwork(AbstractLayer[], loss, η)

function (NN::NeuralNetwork)(x)
    output = x

    for layer in NN.layers
        output = layer.activation.(layer(output))
    end

    output
end

function Base.show(io::IO, NN::NeuralNetwork)
    if isempty(NN.layers)
        return println(io, "Empty Neural Network")
    end

    for layer in NN.layers[1:end-1]
        println(io, "├─ $layer")
    end
    println(io, "└─ $(NN.layers[end])")
end

getOutput(L::Dense{I,O}) where {I, O} = O

function addLayer!(NN::NeuralNetwork, L::Dense{i, o}) where {i, o}
    if isempty(NN.layers)
    elseif getOutput(NN.layers[end]) != i
        throw(DimensionMismatch(""))
    end

    push!(NN.layers, L)

    NN
end

function addLayer!(NN::NeuralNetwork, L::T) where T <: AbstractLayer
    push!(NN.layers, L)

    NN
end

addLayer!(NN::NeuralNetwork, in::Int, out::Int, activation=ReLU; id="") = addLayer!(NN, Dense(in, out, activation, id=id))
addLayer!(NN::NeuralNetwork, (in, out)::Pair, activation=ReLU; id="") = addLayer!(NN, Dense(in, out, activation, id=id))

function matrixTensor(A::Matrix{T}, H::Array{T, 3}; mult = 1) where T <: Number
    R = zeros(size(H)[1:2]..., size(A, 1))

    @simd for i = 1:size(R, 1)
        @simd for j = 1:size(R, 2)
            R[i, j, :] .= mult .* A * H[i, j, :]
        end
    end

    R
end

function vectorTensor(V, H::Array{T, 3}) where T <: Number
    R = zeros(size(H)[1:2])

    @simd for i in 1:size(H, 1)
        @simd for j in 1:size(H, 2)
            R[i, j] = (V' * H[i, j, :])[1]
        end
    end

    R
end

function train!(NN::NeuralNetwork, x, y)
    h = x
    tensors = Array{Float64, 3}[]
    biases = Array{Float64, 2}[]
    for layer in NN.layers
        z = (layer(h))[:, 1]

        ∂W = zeros(length(z), length(h), length(z))
        ∂b = [Int(i == j) for i = eachindex(z), j = eachindex(z)]

        for i = 1:size(layer.weight, 1)
            ∂W[i, :, i] .= h
        end

        for i in eachindex(tensors)
            tensors[i] = matrixTensor(layer.weight, tensors[i], mult=derivative(layer.activation, z))
            biases[i] = derivative(layer.activation, z) .* layer.weight * biases[i]
        end

        push!(tensors, ∂W)
        push!(biases, ∂b)

        h = layer.activation.(z)
    end

    for i in eachindex(tensors)
        NN.layers[i].weight -= NN.learningRate * vectorTensor(h.-y, tensors[i])
        NN.layers[i].bias -= NN.learningRate * biases[i]' * (h.-y)
    end
end


function batch!(NN::NeuralNetwork, batch)
    for (x, y) ∈ batch
        train!(NN, x, y)
    end
end

batch!(NN::NeuralNetwork, x, y) = batch!(NN, collect(zip(x,y)))

function testing()
    N = NeuralNetwork(0.01)
    addLayer!(N, 1, 5, identity)
    addLayer!(N, 5, 1, identity)

    batch = [i => 1 + 2*i + rand() for i = 1:100]

    batch!(N, batch)

    N
end
