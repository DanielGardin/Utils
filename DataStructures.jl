
"""

Stack{T}

A "FIFO" (First-in-first-out) array struct. It has 2 main methods:

`insert!(Stack, element)` append an element to the Stack

`remove!(Stack)` removes the last element inserted to the Stack
"""
mutable struct Stack{T}
content::Array{T}
n::Int

Stack(T = Any) = new{T}(T[], 0)
end


"""

Queue{T}

A "FILO" (First-in-last-out) array struct. It has 2 main methods:

`insert!(Stack, element)` append an element to the Stack

`remove!(Stack)` removes the oldest element inserted to the Stack
"""
mutable struct Queue{T}
content::Array{T}
n::Int

Queue(T = Any) = new{T}(T[], 0)
end

"""

insert!(datastructure, item)

Insert an element to a Structured Data collection as Stack, Queue or Tree
"""
function Base.insert!(arraylike::Union{Stack{T}, Queue{T}}, elem::T) where T
append!(arraylike.content, elem)
arraylike.n += 1

arraylike
end

"""

remove!(datastructure [, item])

Remove and return an item based on the logic of the Data Structure.

[`Stack`](@ref)s remove the most recent item added to it\\
[`Queue`](@ref)s remove the oldest item added to it\\
[`SearchTree`](@ref)s remove a given item in O(log n) efficiency
"""
function remove!(arraylike::Stack)
arraylike.n -= 1

pop!(arraylike.content)
end

function remove!(arraylike::Queue)
arraylike.n -= 1

popfirst!(arraylike.content)
end

abstract type AbstractNode end

mutable struct Node{T} <: AbstractNode
value::T
key
left
right
end

Node(value::T, key) where T = Node(value, key, missing, missing)
Node(value::T) where T <: Real = Node(value, value, missing, missing)
Node(value) = Node(value, 0, missing, missing)

isleaf(Node) = ismissing(Node.left) && ismissing(Node.right)

function _indent(depth)
for v in depth 
    v ? print("│  ") : print("   ")
end
end

function _recursiveWalk(nd, depth)
_indent(depth)
if !ismissing(nd.left)
    kind = isleaf(nd.left) ? "Leaf" : "Node"

    if !ismissing(nd.right)
        println("├─ $kind: $(nd.left.value)")
        _recursiveWalk(nd.left, [depth; true])

        _indent(depth)
        kind = isleaf(nd.right) ? "Leaf" : "Node"
        println("└─ $kind: $(nd.right.value)")
        _recursiveWalk(nd.right, [depth; false])

    else
        println("└─ $kind: $(nd.left.value)")
        _recursiveWalk(nd.left, [depth; false])
    end

else
    if !ismissing(nd.right)
        kind = isleaf(nd.right) ? "Leaf" : "Node"
        println("└─ $kind: $(nd.right.value)")
        _recursiveWalk(nd.right, [depth; false])

    else
        println()
    end
end

end

function Base.show(io::IO, nd::Node)
println("Root: $(nd.value)")

_recursiveWalk(nd, Bool[])
end

mutable struct NNode{N, T} <: AbstractNode
value::T
key
children::Vector{Union{Missing, NNode}}
end

NNode(value::T, key, N::Int) where T = N == 2 ? Node(value, key) : NNode{N, T}(value, key, [missing for _ in 1:N])
NNode(value::T, N::Int) where T <: Real = N == 2 ? Node(value) : NNode{N, T}(value, value, [missing for _ in 1:N])
NNode(value, N::Int) = N == 2 ? Node(value) : NNode{N, T}(value, 0, [missing for _ in 1:N])

function Base.convert(::Type{Node}, nnode::NNode{2, T}) where T
node = Node(nnode.value, nnode.key)
node.left = nnode.children[1]
node.right = nnode.children[2]

node
end

function Base.show(io::IO, node::NNode{N, T}) where {N, T}
println("Root: $(node.value)")
_recursiveNWalk(node, Bool[])
end

isleaf(nd::NNode) = all(ismissing.(nd.children))

function _recursiveNWalk(node, depth)
for (index, child) in enumerate(node.children)   
    if !ismissing(child)
        _indent(depth) 
        kind = isleaf(child) ? "Leaf" : "Node"

        if !all(ismissing.(node.children[index+1:end]))
            println("├─ $kind: $(child.value)")
            _recursiveNWalk(child, [depth; true])
        
        else
            println("└─ $kind: $(child.value)")
            _recursiveNWalk(child, [depth; false])
            break
        end
    
    else
        if all(ismissing.(node.children[index+1:end]))
            _indent(depth)
            println()
            break
        end
    end
end
end

@inline function Base.getindex(nd::NNode, index) 
nd.children[index]
end

function Base.setindex!(nd::NNode{N, T}, node::NNode, index) where {N, T}
nd.children[index] = node
end

abstract type AbstractTree end

mutable struct Tree <: AbstractTree
root
end

mutable struct SearchTree <: AbstractTree
root
end

Tree() = Tree(missing)
SearchTree() = SearchTree(missing)

function Base.show(io::IO, tree::AbstractTree)
if ismissing(tree.root) return print(io, "Empty Tree") end
show(io, tree.root)
end

function Base.insert!(tree::SearchTree, value, key)
if ismissing(tree.root)
    tree.root = Node(value, key)

    return tree
end

parent = tree.root
current = tree.root
while !ismissing(current)
    parent = current
    current = if key <= current.key
        current.left
    else
        current.right
    end
end

if key <= parent.key
    parent.left = Node(value, key)
else
    parent.right = Node(value, key)
end

tree
end

function find(tree::SearchTree, key)
current = tree.root

while !isleaf(current)
    if key <= current.key
        current = current.left
    else
        current = current.right
    end
end

current.value
end

function popmin!(tree::SearchTree)
current = tree.root

if ismissing(current.left)
    tree.root = current.right
    current.right = nothing
    return current.value
end

parent = tree.root

while !ismissing(current.left)
    parent = current
    current = current.left
end

parent.left = current.right
current.right = nothing

return current.value
end

Base.insert!(tree::SearchTree, value) = insert!(tree, value, value)

mutable struct ExploratoryTree <: AbstractTree
root
explorationSet
end

ExploratoryTree(node::Node, exploration = Stack) = ExploratoryTree(node, exploration[node])

function resolve!(tree::ExploratoryTree, fun)
evaluating = remove!(tree.explorationSet)

return fun(evaluating)
end

mutable struct NTree{N} <: AbstractTree
root::NNode{N}
end