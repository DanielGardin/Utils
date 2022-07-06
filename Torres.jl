function allocateTowers(points::AbstractArray, r::Real)
    x = sort(points)
    i = 1
    Towers = Float64[]

    while i <= length(points)
        xFirst = x[i]
        i += 1
        while i <= length(points) && x[i] <= xFirst + 2r
            i += 1
        end

        push!(Towers, (x[i-1] + xFirst)/2)
    end

    Towers
end