import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using PowerModelsDistribution
using Ipopt
using JuMP
using LinearAlgebra
using DataFrames
using Plots


const PMD = PowerModelsDistribution

file = "/mnt/c/Users/auc009/OneDrive - CSIRO/Documents/power-models-distribution/pmd_ev_experiments/data/Three-wire-Kron-reduced/network_1/Feeder_1/Master.dss"
#file = "/mnt/c/Users/auc009/OneDrive - CSIRO/Documents/power-models-distribution/pmd_ev_experiments/data/Four-wire/network_1/Feeder_1/Master.dss"


ipopt_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>1, "sb"=>"yes","warm_start_init_point"=>"yes")

eng4w = parse_file(file, transformations=[transform_loops!, reduce_lines!])

eng4w["settings"]["sbase_default"] = 1
eng4w["voltage_source"]["source"]["rs"] .= 0
eng4w["voltage_source"]["source"]["xs"] .= 0
#eng4w["is_kron_reduced"] = true    #Activate only for 3 wire networks

math4w = transform_data_model(eng4w, multinetwork=false, kron_reduce=true, phase_project=true)

# for (i, bus) in math4w["bus"] 
#   bus["vmin"] = [0.9 * ones(3) ; 0 ] 
#   bus["vmax"] = [1.1 * ones(3) ; Inf] 
# end

# for (i, load) in math4w["load"] 
#     load["pd"] *= 4 
#     load["qd"] *= 4 
# end

println("Running unbalanced AC Optimal Power Flow with Ipopt...") 
#result_opf = solve_mc_opf( math4w, ACPUPowerModel, ipopt_solver) 
result_opf = solve_mc_opf( math4w, IVRUPowerModel, ipopt_solver)
println("OPF solve status: ", result_opf["termination_status"]) 
println("Objective value (if any): ", get(result_opf, "objective", missing))


#########Visualisation of the network#########
function make_lines_df_from_eng(eng)
    rows = NamedTuple[]

    for (id, ln) in eng["line"]
        f_bus = ln["f_bus"]
        t_bus = ln["t_bus"]

        f_phases = ln["f_connections"]
        t_phases = ln["t_connections"]

        @assert length(f_phases) == length(t_phases)

        # PMD lengths are in meters unless otherwise stated
        length_km = ln["length"] / 1000.0

        push!(rows, (
            Bus1 = f_bus,
            Bus2 = t_bus,
            phases = f_phases,
            length_km = length_km
        ))
    end

    return DataFrame(rows)
end


function compute_bus_distances(lines_df; source_bus="sourcebus")
    adj = Dict{String, Vector{Tuple{String, Float64}}}()

    for r in eachrow(lines_df)
        b1 = r.Bus1
        b2 = r.Bus2
        len = r.length_km

        push!(get!(adj, b1, Tuple{String,Float64}[]), (b2, len))
        push!(get!(adj, b2, Tuple{String,Float64}[]), (b1, len))
    end

    dist = Dict{String,Float64}(source_bus => 0.0)
    queue = [source_bus]

    while !isempty(queue)
        u = popfirst!(queue)
        for (v, w) in get(adj, u, [])
            if !haskey(dist, v)
                dist[v] = dist[u] + w
                push!(queue, v)
            end
        end
    end

    return dist
end

# println(keys(eng4w["bus"]))
# println(keys(eng4w["voltage_source"]))

function solved_bus_vm_volts(result_opf, math; vbase_ln=230.0)
    sol_bus = result_opf["solution"]["bus"]
    buses_dict = Dict{String, Dict{String,Any}}()

    for (bus_id, bus_data) in math["bus"]
        name = bus_data["name"]

        if !haskey(sol_bus, bus_id)
            continue
        end

        sb = sol_bus[bus_id]

        vm_pu =
            haskey(sb, "vm") ? sb["vm"] :
            haskey(sb, "vr") ? sqrt.(sb["vr"].^2 .+ sb["vi"].^2) :
            nothing

        vm_pu === nothing && continue

        vmV = vm_pu .* vbase_ln

        buses_dict[name] = Dict(
            "vma" => [vmV[1]],
            "vmb" => [vmV[2]],
            "vmc" => [vmV[3]]
        )
    end

    return buses_dict
end

# println("Top-level keys: ", keys(eng4w))
# println("Has line? ", haskey(eng4w, "line"))
# println("Has branch? ", haskey(eng4w, "branch"))

# # inspect one element
# if haskey(eng4w, "line")
#     k, ln = first(collect(eng4w["line"]))
#     println("\nSample line id = ", k)
#     println("Line keys: ", keys(ln))
#     println("Line dict: ", ln)
# elseif haskey(eng4w, "branch")
#     k, br = first(collect(eng4w["branch"]))
#     println("\nSample branch id = ", k)
#     println("Branch keys: ", keys(br))
#     println("Branch dict: ", br)
# end

function plot_voltage_along_feeder_snap(buses_dict, lines_df;
        t=1, Vthreshold=1000, vmin=0.94*230, vmax=1.1*230)

    p = plot(legend=false)
    ylabel!("Voltage magnitude P–N (V)")
    xlabel!("Distance from reference bus (km)")
    title!("Voltage drop along feeder")

    colors = Dict(1=>:blue, 2=>:red, 3=>:black)

    for r in eachrow(lines_df)
        b1 = r.Bus1
        b2 = r.Bus2
        phases = r.phases

        for ph in phases
            vm_f = ph == 1 ? buses_dict[b1]["vma"][t] :
                   ph == 2 ? buses_dict[b1]["vmb"][t] :
                             buses_dict[b1]["vmc"][t]

            vm_t = ph == 1 ? buses_dict[b2]["vma"][t] :
                   ph == 2 ? buses_dict[b2]["vmb"][t] :
                             buses_dict[b2]["vmc"][t]

            if vm_f < Vthreshold && vm_t < Vthreshold
                plot!(
                    [buses_dict[b1]["distance"], buses_dict[b2]["distance"]],
                    [vm_f, vm_t],
                    color=colors[ph],
                    marker=:circle,
                    markersize=1
                )
            end
        end
    end

    maxdist = maximum(bus["distance"] for bus in values(buses_dict))
    plot!([0, maxdist], [vmin, vmin], linestyle=:dash, color=:red)
    plot!([0, maxdist], [vmax, vmax], linestyle=:dash, color=:red)

    display(p)
    return p
end

lines_df = make_lines_df_from_eng(eng4w)
dist = compute_bus_distances(lines_df; source_bus="sourcebus")

buses_dict = solved_bus_vm_volts(result_opf, math4w)

# add distance into each bus entry (your plot needs buses_dict[bus]["distance"])
for (bus, d) in dist
    haskey(buses_dict, bus) && (buses_dict[bus]["distance"] = d)
end

p = plot_voltage_along_feeder_snap(
    buses_dict,
    lines_df;
    t=1,
    Vthreshold=1000,
    vmin=0.94*230,
    vmax=1.1*230
)

figdir = joinpath(@__DIR__, "..", "results", "figures")
mkpath(figdir)
savefig(p, joinpath(figdir, "baseline_voltage_profile.png"))

function plot_voltage_histogram_snap(buses_dict; t=1, Vthreshold=1000, vmin = 0.94*230, vmax = 1.1*230)
    colors = [:blue, :red, :black]
    phase_a = []
    phase_b = []
    phase_c = []
    for (bus_name, bus_data) in buses_dict
        if haskey(bus_data, "vma") && bus_data["vma"][t] < Vthreshold
            push!(phase_a, bus_data["vma"][t])
        end
        if haskey(bus_data, "vmb") && bus_data["vmb"][t] < Vthreshold
            push!(phase_b, bus_data["vmb"][t])
        end
        if haskey(bus_data, "vmc") && bus_data["vmc"][t] < Vthreshold
            push!(phase_c, bus_data["vmc"][t])
        end
    end

    bins = (vmin-1):0.5:(vmax+1)
    p = histogram(phase_a; bins, color=colors[1], label="phase a")
    histogram!(phase_b; bins, color=colors[2], label="phase b")
    histogram!(phase_c; bins, color=colors[3], label="phase c")
    ylabel!("Counts (-)")
    title!("Voltage histogram")
    xlabel!("Voltage magnitude (V)")

    # plot!([0; maxdist], [vmin; vmin], color=:red, linestyle=:dash)
    # plot!([0; maxdist], [vmax; vmax], color=:red, linestyle=:dash)
    display(p)
    return p
end

p_hist = plot_voltage_histogram_snap(
    buses_dict;
    t=1,
    Vthreshold=1000,
    vmin=0.94*230,
    vmax=1.1*230
)

savefig(p_hist, joinpath(figdir, "baseline_voltage_histogram.png"))
# optional vector version (nice for reports)
savefig(p_hist, joinpath(figdir, "baseline_voltage_histogram.pdf"))

function plot_voltage_snap(buses_dict, lines_df; t=1, Vthreshold=1000, vmin = 0.94*230, vmax = 1.1*230)
    p1 = plot_voltage_along_feeder_snap(buses_dict, lines_df; t=t, Vthreshold=Vthreshold, vmin = vmin, vmax = vmax)
    p2 = plot_voltage_histogram_snap(buses_dict; t=t, Vthreshold=Vthreshold, vmin = vmin, vmax = vmax)
    p = plot(p1, p2, layout=(1,2))
    return p
end

p_combined = plot_voltage_snap(
    buses_dict,
    lines_df;
    t=1,
    Vthreshold=1000,
    vmin=0.94*230,
    vmax=1.1*230
)

savefig(p_combined, joinpath(figdir, "baseline_voltage_combined.png"))

