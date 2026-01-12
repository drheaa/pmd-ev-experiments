# src/core/topology.jl
#
# Purpose
# -------
# This file builds simple topology utilities that help with plotting and bus selection:
# - extract line connectivity from the engineering model
# - build a graph of the feeder (bus names as nodes)
# - compute electrical distance from the source bus
# - compute a stable 2D layout for topology plots (not geographic, but consistent)
# - identify "mid-feeder" and "far-end" buses using distance
#
# Notes
# -----
# This is intentionally simple and student-friendly.
# The layout is tree-like using the shortest-path tree, so it is stable across runs.

using DataFrames
using Graphs
using Statistics

"""
base_bus_name(bus_string)

OpenDSS sometimes stores buses like "12.1.2.3" or "sourcebus.1.2.3".
For topology (graph), the base bus name is needed (everything before the first dot).
"""
function base_bus_name(bus_string)
    return split(String(bus_string), ".")[1]
end


"""
build_lines_df(eng)

Extracts a simple table of lines with:
- Bus1 (base bus name)
- Bus2 (base bus name)
- Length (numeric)

If length is missing, it uses 1.0 so the graph still works.
"""
function build_lines_df(eng)
    line_dict = get(eng, "line", Dict())
    rows = NamedTuple[]

    for (id, ln) in line_dict
        bus1 = get(ln, "bus1", get(ln, "f_bus", nothing))
        bus2 = get(ln, "bus2", get(ln, "t_bus", nothing))
        if bus1 === nothing || bus2 === nothing
            continue
        end

        b1 = base_bus_name(bus1)
        b2 = base_bus_name(bus2)
        L  = float(get(ln, "length", 1.0))

        push!(rows, (Bus1=b1, Bus2=b2, Length=L))
    end

    return DataFrame(rows)
end


"""
build_graph_and_weights(lines_df)

Builds:
- g: an undirected graph
- w: edge weights dictionary (i,j) -> length
- b2i: bus name -> node index
- i2b: node index -> bus name
"""
function build_graph_and_weights(lines_df::DataFrame)
    buses = unique(vcat(lines_df.Bus1, lines_df.Bus2))
    sort!(buses)

    b2i = Dict(b => i for (i, b) in enumerate(buses))
    i2b = Dict(i => b for (b, i) in b2i)

    g = Graph(length(buses))
    w = Dict{Tuple{Int,Int}, Float64}()

    for r in eachrow(lines_df)
        i = b2i[r.Bus1]
        j = b2i[r.Bus2]
        add_edge!(g, i, j)
        w[(i, j)] = r.Length
        w[(j, i)] = r.Length
    end

    return g, w, b2i, i2b
end


"""
dijkstra_distances(g, w, source_idx)

Computes weighted shortest-path distance from source_idx to all nodes.
Returns:
- dist: vector of distances
- prev: predecessor vector (defines a shortest-path tree)
"""
function dijkstra_distances(g::Graph, w::Dict{Tuple{Int,Int},Float64}, source_idx::Int)
    n = nv(g)
    dist = fill(Inf, n)
    prev = fill(0, n)
    visited = falses(n)

    dist[source_idx] = 0.0
    pq = [(0.0, source_idx)]  # (distance, node)

    while !isempty(pq)
        sort!(pq, by=x -> x[1])
        (d, u) = popfirst!(pq)

        if visited[u]
            continue
        end
        visited[u] = true

        for v in neighbors(g, u)
            nd = d + get(w, (u, v), 1.0)
            if nd < dist[v]
                dist[v] = nd
                prev[v] = u
                push!(pq, (nd, v))
            end
        end
    end

    return dist, prev
end


"""
tree_children(prev)

Turns a predecessor vector into children lists.
This is useful for making a clean tree-like plot layout.
"""
function tree_children(prev::Vector{Int})
    n = length(prev)
    children = [Int[] for _ in 1:n]
    for v in 1:n
        p = prev[v]
        if p != 0
            push!(children[p], v)
        end
    end
    return children
end


"""
tree_layout(prev, dist_km, root)

Computes a stable 2D layout:
- x coordinate = electrical distance from source (km)
- y coordinate = tree order (so branches do not overlap as much)

This is not a geographic layout. It is for consistent comparisons.
"""
function tree_layout(prev::Vector{Int}, dist_km::Vector{Float64}, root::Int)
    children = tree_children(prev)
    y = fill(0.0, length(prev))
    next_y = Ref(0.0)

    function dfs(u::Int)
        if isempty(children[u])
            y[u] = next_y[]
            next_y[] += 1.0
        else
            for c in sort(children[u])
                dfs(c)
            end
            y[u] = mean(y[children[u]])
        end
    end

    dfs(root)
    x = copy(dist_km)
    return x, y
end


"""
dist_by_busname(i2b, dist_km)

Builds a dictionary: bus_name -> distance (km).
Only finite distances are kept.
"""
function dist_by_busname(i2b::Dict{Int,String}, dist_km::Vector{Float64})
    d = Dict{String,Float64}()
    for i in 1:length(dist_km)
        if isfinite(dist_km[i])
            d[i2b[i]] = dist_km[i]
        end
    end
    return d
end


"""
find_far_and_mid_bus(i2b, dist_km)

Finds:
- far_end bus: maximum distance from source
- mid_feeder bus: closest to half of max distance
"""
function find_far_and_mid_bus(i2b::Dict{Int,String}, dist_km::Vector{Float64})
    finite_mask = isfinite.(dist_km)
    maxdist = maximum(dist_km[finite_mask])

    far_idx = argmax(dist_km .* finite_mask)
    far_name = i2b[far_idx]

    target = 0.5 * maxdist
    mid_idx = argmin(abs.(dist_km .- target) .+ (.!finite_mask) .* 1e9)
    mid_name = i2b[mid_idx]

    return far_name, mid_name
end
