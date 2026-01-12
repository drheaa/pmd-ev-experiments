# src/core/plotting.jl
#
# Purpose
# -------
# This file contains plotting helpers used in notebooks.
#
# Plots included:
# 1) Topology plot (graph layout) with EV locations marked.
# 2) Voltage-along-feeder plot (voltage magnitude vs distance).
#
# Notes
# -----
# This is written for snapshot experiments.
# Voltages are expected in per-unit from OPF.
# The feeder-distance plot uses the line table and the per-bus distance dictionary.

using DataFrames
using Plots
using CairoMakie

"""
plot_topology_with_markers(g, xpos, ypos, node_color; source_idx, ev_indices, ...)

- g: Graph
- xpos, ypos: vectors from tree_layout
- node_color: usually dist_km (electrical distance)
- source_idx: node index of source bus
- ev_indices: node indices where EVs are connected

This saves a supporting figure that makes it easy to point to EV locations.
"""
function plot_topology_with_markers(
    g, xpos::Vector{Float64}, ypos::Vector{Float64}, node_color::Vector{Float64};
    source_idx::Int,
    ev_indices::Vector{Int}=Int[],
    title_str::String="Topology",
    save_path::Union{Nothing,String}=nothing
)
    fig = Figure(size=(1000, 700))
    ax = Axis(fig[1,1], title=title_str)

    # Draw edges first
    for e in edges(g)
        u = src(e); v = dst(e)
        lines!(ax, [xpos[u], xpos[v]], [ypos[u], ypos[v]], color=:gray50, linewidth=2)
    end

    # Draw nodes (colored by distance or other metric)
    scatter!(ax, xpos, ypos, color=node_color, markersize=10)

    # Source marker
    scatter!(ax, [xpos[source_idx]], [ypos[source_idx]], markersize=22, color=:black)
    text!(ax, "source", position=(xpos[source_idx], ypos[source_idx] + 0.6), align=(:center, :bottom))

    # EV markers
    for (k, ev) in enumerate(ev_indices)
        scatter!(ax, [xpos[ev]], [ypos[ev]], markersize=22, color=:red)
        text!(ax, "EV$(k)", position=(xpos[ev], ypos[ev] + 0.6), align=(:center, :bottom))
    end

    # Clean look similar to the screenshot style
    hidedecorations!(ax)
    hidespines!(ax)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end


"""
build_buses_dict_for_voltage_plot(id2name, dist_km_byname, vm_byid)

Creates a buses_dict that stores:
- distance
- vma, vmb, vmc as 1-element arrays (snapshot format)

This matches the style used in simple feeder plots.
"""
function build_buses_dict_for_voltage_plot(id2name::Dict{String,String},
                                          dist_km_byname::Dict{String,Float64},
                                          vm_byid::Dict{String,Vector{Float64}})
    buses_dict = Dict{String, Dict{String,Any}}()

    for (id, vm) in vm_byid
        name = id2name[id]

        # Three-wire: first 3 entries correspond to phases A/B/C
        vma = length(vm) >= 1 ? vm[1] : NaN
        vmb = length(vm) >= 2 ? vm[2] : NaN
        vmc = length(vm) >= 3 ? vm[3] : NaN

        buses_dict[name] = Dict(
            "distance" => get(dist_km_byname, name, NaN),
            "vma" => [float(vma)],
            "vmb" => [float(vmb)],
            "vmc" => [float(vmc)]
        )
    end

    return buses_dict
end


"""
plot_voltage_along_feeder_snap(buses_dict, lines_df; vmin=0.9, vmax=1.1)

Plots per-phase voltage magnitude along the feeder using:
- x axis: distance from source (km)
- y axis: voltage magnitude (p.u.)

This is a supporting plot for voltage drop intuition.
"""
function plot_voltage_along_feeder_snap(buses_dict, lines_df; t=1, vmin=0.9, vmax=1.1)
    p = plot(legend=false)
    xlabel!("Electrical distance from source (km)")
    ylabel!("Voltage magnitude (p.u.)")
    title!("Voltage drop along feeder (snapshot)")

    # Phase colors: A, B, C
    colors = [:blue, :red, :black]

    for r in eachrow(lines_df)
        b1 = r.Bus1
        b2 = r.Bus2

        # Skip if buses are not present in buses_dict
        if !haskey(buses_dict, b1) || !haskey(buses_dict, b2)
            continue
        end

        d1 = buses_dict[b1]["distance"]
        d2 = buses_dict[b2]["distance"]
        if !isfinite(d1) || !isfinite(d2)
            continue
        end

        for phase in 1:3
            v1 = phase==1 ? buses_dict[b1]["vma"][t] : phase==2 ? buses_dict[b1]["vmb"][t] : buses_dict[b1]["vmc"][t]
            v2 = phase==1 ? buses_dict[b2]["vma"][t] : phase==2 ? buses_dict[b2]["vmb"][t] : buses_dict[b2]["vmc"][t]

            if isfinite(v1) && isfinite(v2)
                plot!([d1, d2], [v1, v2], color=colors[phase], marker=:circle, markersize=1)
            end
        end
    end

    # Plot voltage limits as dashed red lines
    maxdist = maximum([b["distance"] for (k,b) in buses_dict if isfinite(b["distance"])])
    plot!([0, maxdist], [vmin, vmin], color=:red, linestyle=:dash)
    plot!([0, maxdist], [vmax, vmax], color=:red, linestyle=:dash)

    return p
end


"""
min_phase_voltage(vm_byid)

Computes the minimum voltage magnitude across all buses and phases (A/B/C).
"""
function min_phase_voltage(vm_byid::Dict{String,Vector{Float64}})
    mins = Float64[]
    for (id, vm) in vm_byid
        nph = min(3, length(vm))
        push!(mins, minimum(vm[1:nph]))
    end
    return minimum(mins)
end


"""
count_voltage_violations(vm_byid; vmin=0.9, vmax=1.1)

Counts buses where at least one phase violates voltage bounds.
"""
function count_voltage_violations(vm_byid::Dict{String,Vector{Float64}}; vmin=0.9, vmax=1.1)
    viol = 0
    for (id, vm) in vm_byid
        nph = min(3, length(vm))
        bad = false
        for ph in 1:nph
            if vm[ph] < vmin || vm[ph] > vmax
                bad = true
                break
            end
        end
        if bad
            viol += 1
        end
    end
    return viol
end
