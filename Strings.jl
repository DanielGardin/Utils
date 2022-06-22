include("DataStructures.jl")

function frequencies(xs)
    fs = Dict{eltype(xs),Int}()
    for x in xs
        fs[x] = get(fs, x, 0) + 1
    end
  
    fs
end

function huffman_encoder(freq::Dict{Char, T}) where T <: Real
    tempTree = SearchTree()

    for (value, key) in freq
        insert!(tempTree, Node(value, key), key)
    end

    while !isleaf(tempTree.root)
        f1 = popmin!(tempTree)
        f2 = popmin!(tempTree)
        insert!(tempTree, Node(f1.value*f2.value, f1.key + f2.key, f1, f2), f1.key+f2.key)
    end

    Tree(tempTree.root.value)
end

function encode(str::String, encoder::AbstractTree)
        encoded = ""
    for char in str
        encoded *= huffman_search(encoder, char)
    end

    encoded
end

huffman_encode(str::String) = encode(str, huffman_encoder(frequencies(str)))

function bin2str(bin::Vector{Bool})
    result = ""
    for bit in bin
        result *= bit ? '1' : '0'
    end

    result
end

function huffman_search(encoder::AbstractTree, char::Char)
    current = encoder.root
    representation = Bool[]

    while !isleaf(current)
        if char in current.left.value
            current = current.left
            representation = [representation; false]
        else
            current = current.right
            representation = [representation; true]
        end
    end

    bin2str(representation)
end

function decode(coded::String, decoder::AbstractTree)
    current = decoder.root
    decoded = ""

    for bit in coded
        if bit == '1' current = current.right
        else          current = current.left end

        if isleaf(current)
            decoded *= current.value
            current = decoder.root
        end
    end

    decoded
end



const PTBR_Encoder = let PTBR = Dict('a' => 14.63, 'b' => 1.04, 'c' => 3.88, 'd' => 4.99, 'e' => 12.57,
                                     'f' => 1.02 , 'g' => 1.3 , 'h' => 1.28, 'i' => 6.18, 'j' => 0.4  ,
                                     'k' => 0.02 , 'l' => 2.78, 'm' => 4.74, 'n' => 5.05, 'o' => 10.73,
                                     'p' => 2.52 , 'q' => 1.2 , 'r' => 6.53, 's' => 7.81, 't' => 4.34 ,
                                     'u' => 4.63 , 'v' => 1.67, 'w' => 0.01, 'x' => 0.21, 'y' => 0.01 ,
                                     'z' => 0.47 , ' ' => 15.32)
                        huffman_encoder(PTBR)
                    end

"""

    entropy(string)

Returns the entropy of a string. The entropy of a string is related to the number of bits a
given string can be represented when optimally compressed.
"""
function entropy(s::String)
    prob = frequencies(s)

    H = 0
    for value in values(prob)
        H -= value/length(s) * (log2(value) - log2(length(s)))
    end
    H
end
