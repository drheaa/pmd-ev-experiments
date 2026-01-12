import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using PowerModelsDistribution
using Ipopt
using JuMP
using LinearAlgebra
using DataFrames
using Plots
using CSV

const PMD = PowerModelsDistribution

file = "/mnt/c/Users/auc009/OneDrive - CSIRO/Documents/power-models-distribution/pmd_ev_experiments/data/Three-wire-Kron-reduced/network_1/Feeder_1/Master.dss"

ipopt_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer,
    "print_level"=>1,
    "sb"=>"yes",
    "warm_start_init_point"=>"yes"
)

# Voltage limits in volts (AU LV)
const VBASE_LN = 230.0
const VMIN = 0.94 * VBASE_LN
const VMAX = 1.10 * VBASE_LN

# EV settings
const EV_P_KW = 7.0
const EV_PF = 0.95              # lagging PF for reactive power (set to 1.0 if you want Q=0)
const EV_PHASES_LABEL = Dict(1=>"a", 2=>"b", 3=>"c")

# Convert kW -> MW
ev_p_mw = EV_P_KW / 1000.0
ev_q_mvar = (EV_PF >= 0.999) ? 0.0 : ev_p_mw * tan(acos(EV_PF))


eng4w = parse_file(file, transformations=[transform_loops!, reduce_lines!])

eng4w["settings"]["sbase_default"] = 1
eng4w["voltage_source"]["source"]["rs"] .= 0
eng4w["voltage_source"]["source"]["xs"] .= 0

math4w = transform_data_model(eng4w, multinetwork=false, kron_reduce=true, phase_project=true)

println("Running BASELINE unbalanced AC OPF with Ipopt...")
result_base = solve_mc_opf(math4w, IVRUPowerModel, ipopt_solver)
println("BASELINE solve status: ", result_base["termination_status"])
println("BASELINE objective: ", get(result_base, "objective", missing))

# -----------------------
# Helper functions (network + voltages)
# -----------------------
function make_lines_df_from_eng(eng)
    rows = NamedTuple[]
    for (id, ln) in eng["line"]
        f_bus = ln["f_bus"]
        t_bus = ln["t_bus"]
        f_phases = ln["f_connections"]
        t_phases = ln["t_connections"]
        @assert length(f_phases) == length(t_phases)
        length_km = ln["length"] / 1000.0  # PMD length is meters here
        push!(rows, (Bus1=f_bus, Bus2=t_bus, phases=f_phases, length_km=length_km))
    end
    return DataFrame(rows)
end

function compute_bus_distances(lines_df; source_bus="sourcebus")
    adj = Dict{String, Vector{Tuple{String, Float64}}}()
    for r in eachrow(lines_df)
        b1, b2, len = r.Bus1, r.Bus2, r.length_km
        push!(get!(adj, b1, Tuple{String,Float64}[]), (b2, len))
        push!(get!(adj, b2, Tuple{String,Float64}[]), (b1, len))
    end

    dist = Dict{String,Float64}(source_bus => 0.0)
    queue = [source_bus]

    while !isempty(queue)
        u = popfirst!(queue)
        for (v, w) in get(adj, u, Tuple{String,Float64}[])
            if !haskey(dist, v)
                dist[v] = dist[u] + w
                push!(queue, v)
            end
        end
    end
    return dist
end

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
            (haskey(sb, "vr") && haskey(sb, "vi")) ? sqrt.(sb["vr"].^2 .+ sb["vi"].^2) :
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

function plot_voltage_along_feeder_snap(buses_dict, lines_df;
        t=1, Vthreshold=1000, vmin=VMIN, vmax=VMAX, title_str="Voltage drop along feeder")

    p = plot(legend=false)
    ylabel!("Voltage magnitude P-N (V)")
    xlabel!("Distance from reference bus (km)")
    title!(title_str)

    colors = Dict(1=>:blue, 2=>:red, 3=>:black)

    for r in eachrow(lines_df)
        b1, b2, phases = r.Bus1, r.Bus2, r.phases
        if !(haskey(buses_dict, b1) && haskey(buses_dict, b2))
            continue
        end

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

    maxdist = maximum(bus["distance"] for bus in values(buses_dict) if haskey(bus, "distance"))
    plot!([0, maxdist], [vmin, vmin], linestyle=:dash, color=:red)
    plot!([0, maxdist], [vmax, vmax], linestyle=:dash, color=:red)

    display(p)
    return p
end

function plot_voltage_histogram_snap(buses_dict; t=1, Vthreshold=1000, vmin=VMIN, vmax=VMAX, title_str="Voltage histogram")
    phase_a = Float64[]
    phase_b = Float64[]
    phase_c = Float64[]

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
    p = histogram(phase_a; bins, color=:blue, label="phase a")
    histogram!(phase_b; bins, color=:red, label="phase b")
    histogram!(phase_c; bins, color=:black, label="phase c")
    ylabel!("Counts (-)")
    xlabel!("Voltage magnitude (V)")
    title!(title_str)

    # show limits on histogram
    vline!([vmin], color=:red, linestyle=:dash, label=false)
    vline!([vmax], color=:red, linestyle=:dash, label=false)

    display(p)
    return p
end

function voltage_summary(buses_dict; t=1, vmin=VMIN)
    mins = Dict{Int, Float64}(1=>Inf, 2=>Inf, 3=>Inf)
    minbus = Dict{Int, Union{Missing,String}}(1=>missing, 2=>missing, 3=>missing)
    viols = Dict{Int, Int}(1=>0, 2=>0, 3=>0)

    for (bus, d) in buses_dict
        for ph in 1:3
            key = ph == 1 ? "vma" : ph == 2 ? "vmb" : "vmc"
            if haskey(d, key)
                v = d[key][t]
                if v < mins[ph]
                    mins[ph] = v
                    minbus[ph] = bus
                end
                if v < vmin
                    viols[ph] += 1
                end
            end
        end
    end

    return mins, minbus, viols
end

function find_weakest_phase_and_end_bus(buses_dict; t=1)
    mins, minbus, _ = voltage_summary(buses_dict; t=t, vmin=VMIN)

    # weakest phase = lowest minimum voltage
    weakest_phase = argmin([mins[1], mins[2], mins[3]])

    # pick a bus near end: max distance among buses with that phase voltage defined
    phase_key = weakest_phase == 1 ? "vma" : weakest_phase == 2 ? "vmb" : "vmc"
    best_bus = nothing
    best_dist = -Inf

    for (bus, d) in buses_dict
        if haskey(d, phase_key) && haskey(d, "distance")
            if d["distance"] > best_dist
                best_dist = d["distance"]
                best_bus = bus
            end
        end
    end

    # fallback: if distances missing, just use min-voltage bus
    if best_bus === nothing
        best_bus = minbus[weakest_phase]
        best_dist = missing
    end

    return weakest_phase, best_bus, best_dist, mins
end

# -----------------------
# Build baseline buses_dict + distances
# -----------------------
lines_df = make_lines_df_from_eng(eng4w)
dist = compute_bus_distances(lines_df; source_bus="sourcebus")

buses_base = solved_bus_vm_volts(result_base, math4w; vbase_ln=VBASE_LN)

for (bus, d) in dist
    haskey(buses_base, bus) && (buses_base[bus]["distance"] = d)
end

# Save baseline plots
figdir = joinpath(@__DIR__, "..", "results", "figures")
mkpath(figdir)

p_base_profile = plot_voltage_along_feeder_snap(buses_base, lines_df; t=1, title_str="Baseline: voltage drop along feeder")
savefig(p_base_profile, joinpath(figdir, "baseline_voltage_profile.png"))

p_base_hist = plot_voltage_histogram_snap(buses_base; t=1, title_str="Baseline: voltage histogram")
savefig(p_base_hist, joinpath(figdir, "baseline_voltage_histogram.png"))
savefig(p_base_hist, joinpath(figdir, "baseline_voltage_histogram.pdf"))

p_base_combined = plot(p_base_profile, p_base_hist, layout=(1,2))
savefig(p_base_combined, joinpath(figdir, "baseline_voltage_combined.png"))

# -----------------------
# Decide EV location automatically
# -----------------------
weakest_phase, ev_bus, ev_bus_dist, mins_base = find_weakest_phase_and_end_bus(buses_base; t=1)

println("\n--- EV placement (auto) ---")
println("Weakest phase (baseline) = phase ", weakest_phase, " (", EV_PHASES_LABEL[weakest_phase], ")")
println("Chosen EV bus (near end) = ", ev_bus, "  | distance(km) = ", ev_bus_dist)
println("Baseline min voltages (V): A=", round(mins_base[1], digits=2),
        " B=", round(mins_base[2], digits=2),
        " C=", round(mins_base[3], digits=2))

# -----------------------
# Add EV load to math model
# -----------------------
function next_numeric_id(tbl::Dict{String,Any})
    ids = Int[]
    for k in keys(tbl)
        x = tryparse(Int, k)
        x === nothing && continue
        push!(ids, x)
    end
    return isempty(ids) ? 1 : maximum(ids) + 1
end

function add_single_phase_ev!(math, ev_bus::String, phase::Int; p_mw::Float64, q_mvar::Float64)
    pd = zeros(3); qd = zeros(3)
    pd[phase] = p_mw
    qd[phase] = q_mvar

    isempty(math["load"]) && error("math model has no loads to copy schema from.")

    template_id, template = first(collect(math["load"]))
    new_load = deepcopy(template)

    new_id = string(next_numeric_id(math["load"]))   # numeric string key

    new_load["bus"] = ev_bus
    new_load["pd"] = pd
    new_load["qd"] = qd
    new_load["status"] = 1

    # optional: keep readable label inside
    new_load["name"] = "ev_$(ev_bus)_ph$(phase)"

    if haskey(new_load, "connections")
        new_load["connections"] = [1,2,3]
    end

    math["load"][new_id] = new_load
    return new_id
end

math_ev = deepcopy(math4w)
ev_load_id = add_single_phase_ev!(math_ev, ev_bus, weakest_phase; p_mw=ev_p_mw, q_mvar=ev_q_mvar)

println("\nAdded EV load:")
println("  id = ", ev_load_id)
println("  bus = ", ev_bus)
println("  phase = ", weakest_phase, " (", EV_PHASES_LABEL[weakest_phase], ")")
println("  P = ", EV_P_KW, " kW  (", ev_p_mw, " MW)")
println("  PF = ", EV_PF, "  => Q ≈ ", round(ev_q_mvar*1000, digits=2), " kvar")

# -----------------------
# Solve EV OPF
# -----------------------
println("\nRunning EV (7kW single-phase) unbalanced AC OPF with Ipopt...")
result_ev = solve_mc_opf(math_ev, IVRUPowerModel, ipopt_solver)
println("EV solve status: ", result_ev["termination_status"])
println("EV objective: ", get(result_ev, "objective", missing))

# Build EV buses_dict + distances
buses_ev = solved_bus_vm_volts(result_ev, math_ev; vbase_ln=VBASE_LN)

for (bus, d) in dist
    haskey(buses_ev, bus) && (buses_ev[bus]["distance"] = d)
end

# -----------------------
# Plot EV results (and overlay)
# -----------------------
p_ev_profile = plot_voltage_along_feeder_snap(buses_ev, lines_df; t=1, title_str="EV: voltage drop along feeder (7 kW, 1φ)")
savefig(p_ev_profile, joinpath(figdir, "ev7kw_voltage_profile.png"))

p_ev_hist = plot_voltage_histogram_snap(buses_ev; t=1, title_str="EV: voltage histogram (7 kW, 1φ)")
savefig(p_ev_hist, joinpath(figdir, "ev7kw_voltage_histogram.png"))
savefig(p_ev_hist, joinpath(figdir, "ev7kw_voltage_histogram.pdf"))

p_ev_combined = plot(p_ev_profile, p_ev_hist, layout=(1,2))
savefig(p_ev_combined, joinpath(figdir, "ev7kw_voltage_combined.png"))

# Overlay plot (baseline vs EV on same axes)
p_overlay = plot(legend=:topright)
ylabel!("Voltage magnitude P-N (V)")
xlabel!("Distance from reference bus (km)")
title!("Overlay: Baseline vs EV (7 kW, 1φ)")

# plot baseline as faint
colors = Dict(1=>:blue, 2=>:red, 3=>:black)
for ph in 1:3
    # collect (distance, voltage) points from buses_dict (snapshot)
    xs = Float64[]
    ys_base = Float64[]
    ys_ev = Float64[]
    key = ph == 1 ? "vma" : ph == 2 ? "vmb" : "vmc"

    for (bus, d) in buses_base
        if haskey(d, "distance") && haskey(d, key) && haskey(buses_ev, bus) && haskey(buses_ev[bus], key)
            push!(xs, d["distance"])
            push!(ys_base, d[key][1])
            push!(ys_ev, buses_ev[bus][key][1])
        end
    end

    # sort by distance for a cleaner line
    perm = sortperm(xs)
    xs = xs[perm]; ys_base = ys_base[perm]; ys_ev = ys_ev[perm]

    plot!(xs, ys_base, color=colors[ph], linestyle=:dash, label="baseline phase $(EV_PHASES_LABEL[ph])")
    plot!(xs, ys_ev, color=colors[ph], linestyle=:solid, label="ev phase $(EV_PHASES_LABEL[ph])")
end

hline!([VMIN], color=:red, linestyle=:dot, label=false)
hline!([VMAX], color=:red, linestyle=:dot, label=false)

display(p_overlay)
savefig(p_overlay, joinpath(figdir, "overlay_baseline_vs_ev7kw.png"))

# --- side-by-side comparison (baseline vs EV) ---

p_base_profile = plot_voltage_along_feeder_snap(
    buses_base, lines_df;
    t=1, Vthreshold=1000, vmin=0.94*230, vmax=1.1*230
)
title!(p_base_profile, "Baseline: voltage drop")

p_ev_profile = plot_voltage_along_feeder_snap(
    buses_ev, lines_df;
    t=1, Vthreshold=1000, vmin=0.94*230, vmax=1.1*230
)
title!(p_ev_profile, "EV (7 kW, 1φ): voltage drop")

p_base_hist = plot_voltage_histogram_snap(
    buses_base;
    t=1, Vthreshold=1000, vmin=0.94*230, vmax=1.1*230
)
title!(p_base_hist, "Baseline: voltage histogram")

p_ev_hist = plot_voltage_histogram_snap(
    buses_ev;
    t=1, Vthreshold=1000, vmin=0.94*230, vmax=1.1*230
)
title!(p_ev_hist, "EV (7 kW, 1φ): voltage histogram")

p_compare = plot(
    p_base_profile, p_ev_profile,
    p_base_hist,    p_ev_hist,
    layout=(2,2),
    size=(1200,800)
)

display(p_compare)

savefig(p_compare, joinpath(figdir, "compare_baseline_vs_ev7kw.png"))
savefig(p_compare, joinpath(figdir, "compare_baseline_vs_ev7kw.pdf"))


# -----------------------
# Summary numbers + save table
# -----------------------
mins_b, minbus_b, viols_b = voltage_summary(buses_base; t=1, vmin=VMIN)
mins_e, minbus_e, viols_e = voltage_summary(buses_ev; t=1, vmin=VMIN)

summary_df = DataFrame(
    case = ["baseline", "ev7kw_1ph"],
    minV_phaseA = [mins_b[1], mins_e[1]],
    minV_phaseB = [mins_b[2], mins_e[2]],
    minV_phaseC = [mins_b[3], mins_e[3]],
    minBus_phaseA = [string(minbus_b[1]), string(minbus_e[1])],
    minBus_phaseB = [string(minbus_b[2]), string(minbus_e[2])],
    minBus_phaseC = [string(minbus_b[3]), string(minbus_e[3])],
    countBelowVmin_phaseA = [viols_b[1], viols_e[1]],
    countBelowVmin_phaseB = [viols_b[2], viols_e[2]],
    countBelowVmin_phaseC = [viols_b[3], viols_e[3]]
)

println("\n--- Summary ---")
show(summary_df, allrows=true, allcols=true)

tabledir = joinpath(@__DIR__, "..", "results", "tables")
mkpath(tabledir)
CSV.write(joinpath(tabledir, "baseline_vs_ev7kw_summary.csv"), summary_df)

println("\nSaved figures to: ", figdir)
println("Saved summary table to: ", joinpath(tabledir, "baseline_vs_ev7kw_summary.csv"))
