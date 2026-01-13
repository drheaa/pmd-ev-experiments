# Goal:
# - Solve baseline once
# - Add a fixed EV (e.g., 7 kW single-phase) at 4 locations:
#   1) source bus
#   2) mid-feeder bus
#   3) far-end bus
#   4) weakest bus (lowest baseline voltage)
# - For each location: generate the same plots + comparisons
# - Incorporate symmetrical components (+, -, 0) via VUF = |V2|/|V1|

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
using Statistics

const PMD = PowerModelsDistribution

file = "/mnt/c/Users/auc009/OneDrive - CSIRO/Documents/power-models-distribution/pmd_ev_experiments/data/Three-wire-Kron-reduced/network_1/Feeder_1/Master.dss"

ipopt_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer,
    "print_level"=>1,
    "sb"=>"yes",
    "warm_start_init_point"=>"yes"
)

# Voltage base and limits (AU LV)
const VBASE_LN = 230.0
const VMIN = 0.94 * VBASE_LN
const VMAX = 1.10 * VBASE_LN

# EV settings
const EV_P_KW = 7.0
const EV_PF = 0.95             # set to 1.0 if you want Q=0
const EV_PHASES_LABEL = Dict(1=>"a", 2=>"b", 3=>"c")

# Convert kW -> MW
ev_p_mw = EV_P_KW / 1000.0
ev_q_mvar = (EV_PF >= 0.999) ? 0.0 : ev_p_mw * tan(acos(EV_PF))

# Output folders
figdir = joinpath(@__DIR__, "..", "results", "figures")
tabledir = joinpath(@__DIR__, "..", "results", "tables")
mkpath(figdir)
mkpath(tabledir)

eng = parse_file(file, transformations=[transform_loops!, reduce_lines!])

eng["settings"]["sbase_default"] = 1
eng["voltage_source"]["source"]["rs"] .= 0
eng["voltage_source"]["source"]["xs"] .= 0

math = transform_data_model(eng, multinetwork=false, kron_reduce=true, phase_project=true)

# -----------------------
# Helper: lines + distances
# -----------------------
function make_lines_df_from_eng(eng)
    rows = NamedTuple[]
    for (id, ln) in eng["line"]
        f_bus = ln["f_bus"]
        t_bus = ln["t_bus"]
        f_phases = ln["f_connections"]
        t_phases = ln["t_connections"]
        @assert length(f_phases) == length(t_phases)
        length_km = ln["length"] / 1000.0
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

# -----------------------
# Helper: extract solved voltages (magnitudes + phasors if available)
# -----------------------
"""
Returns buses_dict keyed by bus name, with:
- vma/vmb/vmc (in volts, snapshot stored as length-1 vector)
- vra/via/vrb/vib/vrc/vic (if available, in pu, snapshot stored as length-1 vector)
We keep pu phasors because sequence components work on phasors; scaling cancels in ratio anyway.
"""
function solved_bus_voltages(result_opf, math; vbase_ln=230.0)
    sol_bus = result_opf["solution"]["bus"]
    buses_dict = Dict{String, Dict{String,Any}}()

    for (bus_id, bus_data) in math["bus"]
        name = bus_data["name"]
        if !haskey(sol_bus, bus_id)
            continue
        end
        sb = sol_bus[bus_id]

        # magnitudes in pu if available
        vm_pu =
            haskey(sb, "vm") ? sb["vm"] :
            (haskey(sb, "vr") && haskey(sb, "vi")) ? sqrt.(sb["vr"].^2 .+ sb["vi"].^2) :
            nothing

        d = Dict{String,Any}()

        if vm_pu !== nothing && length(vm_pu) >= 3
            vmV = vm_pu .* vbase_ln
            d["vma"] = [vmV[1]]
            d["vmb"] = [vmV[2]]
            d["vmc"] = [vmV[3]]
        end

        # phasors in pu, if present
        if haskey(sb, "vr") && haskey(sb, "vi") && length(sb["vr"]) >= 3 && length(sb["vi"]) >= 3
            d["vra"] = [sb["vr"][1]]; d["via"] = [sb["vi"][1]]
            d["vrb"] = [sb["vr"][2]]; d["vib"] = [sb["vi"][2]]
            d["vrc"] = [sb["vr"][3]]; d["vic"] = [sb["vi"][3]]
        end

        # only store if we have at least magnitudes or phasors
        if !isempty(d)
            buses_dict[name] = d
        end
    end

    return buses_dict
end

# -----------------------
# Symmetrical components: V0, V1, V2 and VUF
# -----------------------
"""
Compute sequence components from phase phasors Va,Vb,Vc (complex).
Returns (V0, V1, V2).
"""
function seq_components(Va::ComplexF64, Vb::ComplexF64, Vc::ComplexF64)
    a = cis(2π/3)  # e^(j 120°)
    # Fortescue transform:
    V0 = (Va + Vb + Vc) / 3
    V1 = (Va + a*Vb + a^2*Vc) / 3
    V2 = (Va + a^2*Vb + a*Vc) / 3
    return V0, V1, V2
end

"""
Adds:
- vuf (|V2|/|V1|) at snapshot t into buses_dict[bus]["vuf"]
- v0mag, v1mag, v2mag (magnitudes) if you want to inspect later

If phasors are missing, leaves bus untouched.
"""
function add_vuf!(buses_dict; t=1)
    for (bus, d) in buses_dict
        if all(haskey(d, k) for k in ["vra","via","vrb","vib","vrc","vic"])
            Va = ComplexF64(d["vra"][t], d["via"][t])
            Vb = ComplexF64(d["vrb"][t], d["vib"][t])
            Vc = ComplexF64(d["vrc"][t], d["vic"][t])

            V0, V1, V2 = seq_components(Va, Vb, Vc)
            vuf = (abs(V1) > 1e-9) ? abs(V2)/abs(V1) : NaN

            d["vuf"] = [vuf]
            d["v0mag"] = [abs(V0)]
            d["v1mag"] = [abs(V1)]
            d["v2mag"] = [abs(V2)]
        end
    end
    return buses_dict
end

# -----------------------
# Plotting helpers
# -----------------------
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
        if !(haskey(buses_dict[b1], "distance") && haskey(buses_dict[b2], "distance"))
            continue
        end

        for ph in phases
            key = ph == 1 ? "vma" : ph == 2 ? "vmb" : "vmc"
            if !(haskey(buses_dict[b1], key) && haskey(buses_dict[b2], key))
                continue
            end
            vm_f = buses_dict[b1][key][t]
            vm_t = buses_dict[b2][key][t]

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

    return p
end

function plot_voltage_histogram_snap(buses_dict; t=1, Vthreshold=1000, vmin=VMIN, vmax=VMAX, title_str="Voltage histogram")
    phase_a = Float64[]
    phase_b = Float64[]
    phase_c = Float64[]

    for (bus_name, d) in buses_dict
        if haskey(d, "vma") && d["vma"][t] < Vthreshold; push!(phase_a, d["vma"][t]); end
        if haskey(d, "vmb") && d["vmb"][t] < Vthreshold; push!(phase_b, d["vmb"][t]); end
        if haskey(d, "vmc") && d["vmc"][t] < Vthreshold; push!(phase_c, d["vmc"][t]); end
    end

    bins = (vmin-1):0.5:(vmax+1)
    p = histogram(phase_a; bins, color=:blue, label="phase a")
    histogram!(phase_b; bins, color=:red, label="phase b")
    histogram!(phase_c; bins, color=:black, label="phase c")
    ylabel!("Counts (-)")
    xlabel!("Voltage magnitude (V)")
    title!(title_str)

    vline!([vmin], color=:red, linestyle=:dash, label=false)
    vline!([vmax], color=:red, linestyle=:dash, label=false)

    return p
end

function plot_vuf_along_feeder(buses_dict; t=1, title_str="VUF along feeder (|V2|/|V1|)")
    xs = Float64[]
    ys = Float64[]

    for (bus, d) in buses_dict
        if haskey(d, "distance") && haskey(d, "vuf")
            v = d["vuf"][t]
            if isfinite(v)
                push!(xs, d["distance"])
                push!(ys, v)
            end
        end
    end

    perm = sortperm(xs)
    xs = xs[perm]; ys = ys[perm]

    p = plot(xs, ys, marker=:circle, markersize=2, legend=false)
    xlabel!("Distance from reference bus (km)")
    ylabel!("VUF = |V2|/|V1| (-)")
    title!(title_str)
    return p
end

function plot_vuf_histogram(buses_dict; t=1, title_str="VUF histogram (|V2|/|V1|)")
    vals = Float64[]
    for (bus, d) in buses_dict
        if haskey(d, "vuf")
            v = d["vuf"][t]
            if isfinite(v)
                push!(vals, v)
            end
        end
    end
    p = histogram(vals, bins=0:0.002:0.08, legend=false)
    xlabel!("VUF (-)")
    ylabel!("Counts (-)")
    title!(title_str)
    return p
end

# -----------------------
# Summary helpers
# -----------------------
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

function vuf_summary(buses_dict; t=1)
    vals = Float64[]
    worst = -Inf
    worst_bus = missing

    for (bus, d) in buses_dict
        if haskey(d, "vuf")
            v = d["vuf"][t]
            if isfinite(v)
                push!(vals, v)
                if v > worst
                    worst = v
                    worst_bus = bus
                end
            end
        end
    end

    if isempty(vals)
        return (mean=missing, p95=missing, max=missing, max_bus=missing)
    end

    sort!(vals)
    p95 = vals[clamp(Int(ceil(0.95*length(vals))), 1, length(vals))]
    return (mean=mean(vals), p95=p95, max=maximum(vals), max_bus=worst_bus)
end

# -----------------------
# EV load injection
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
    new_load["name"] = "ev_$(ev_bus)_ph$(phase)"

    if haskey(new_load, "connections")
        new_load["connections"] = [1,2,3]
    end

    math["load"][new_id] = new_load
    return new_id
end

# -----------------------
# Decide location buses based on baseline
# -----------------------
function pick_mid_bus(dist::Dict{String,Float64}; must_exist_in=nothing)
    maxd = maximum(values(dist))
    target = 0.5 * maxd

    best_bus = nothing
    best_err = Inf

    for (bus, d) in dist
        if must_exist_in !== nothing && !haskey(must_exist_in, bus)
            continue
        end
        err = abs(d - target)
        if err < best_err
            best_err = err
            best_bus = bus
        end
    end
    return best_bus
end

function pick_far_end_bus(dist::Dict{String,Float64}; must_exist_in=nothing)
    best_bus = nothing
    best_d = -Inf

    for (bus, d) in dist
        if must_exist_in !== nothing && !haskey(must_exist_in, bus)
            continue
        end
        if d > best_d
            best_d = d
            best_bus = bus
        end
    end
    return best_bus
end

function find_weakest_bus_and_phase(buses_base; t=1)
    # find overall minimum voltage across phases
    best_v = Inf
    best_bus = missing
    best_phase = missing

    for (bus, d) in buses_base
        for ph in 1:3
            key = ph == 1 ? "vma" : ph == 2 ? "vmb" : "vmc"
            if haskey(d, key)
                v = d[key][t]
                if v < best_v
                    best_v = v
                    best_bus = bus
                    best_phase = ph
                end
            end
        end
    end

    return best_bus, best_phase, best_v
end

# -----------------------
# BASELINE solve
# -----------------------
println("Running BASELINE unbalanced AC OPF with Ipopt...")
result_base = solve_mc_opf(math, IVRUPowerModel, ipopt_solver)
println("BASELINE status: ", result_base["termination_status"])
println("BASELINE objective: ", get(result_base, "objective", missing))

# Build baseline dicts
lines_df = make_lines_df_from_eng(eng)
dist = compute_bus_distances(lines_df; source_bus="sourcebus")

buses_base = solved_bus_voltages(result_base, math; vbase_ln=VBASE_LN)

for (bus, d) in dist
    haskey(buses_base, bus) && (buses_base[bus]["distance"] = d)
end

add_vuf!(buses_base; t=1)

# Baseline plots
p_base_profile = plot_voltage_along_feeder_snap(buses_base, lines_df; t=1, title_str="Baseline: voltage drop along feeder")
p_base_hist    = plot_voltage_histogram_snap(buses_base; t=1, title_str="Baseline: voltage histogram")
p_base_vuf     = plot_vuf_along_feeder(buses_base; t=1, title_str="Baseline: VUF along feeder")
p_base_vufhist = plot_vuf_histogram(buses_base; t=1, title_str="Baseline: VUF histogram")

savefig(p_base_profile, joinpath(figdir, "baseline_voltage_profile.png"))
savefig(p_base_hist,    joinpath(figdir, "baseline_voltage_histogram.png"))
savefig(p_base_vuf,     joinpath(figdir, "baseline_vuf_profile.png"))
savefig(p_base_vufhist, joinpath(figdir, "baseline_vuf_histogram.png"))

savefig(plot(p_base_profile, p_base_hist, layout=(1,2)), joinpath(figdir, "baseline_voltage_combined.png"))
savefig(plot(p_base_vuf, p_base_vufhist, layout=(1,2)), joinpath(figdir, "baseline_vuf_combined.png"))

# Choose weakest phase (fixed across cases)
weak_bus, weak_phase, weak_v = find_weakest_bus_and_phase(buses_base; t=1)
println("\nBaseline weakest bus-phase:")
println("  bus = ", weak_bus, " | phase = ", weak_phase, " (", EV_PHASES_LABEL[weak_phase], ") | Vmin = ", round(weak_v, digits=2), " V")

# Decide location buses
source_bus = "sourcebus"
mid_bus = pick_mid_bus(dist; must_exist_in=buses_base)
far_bus = pick_far_end_bus(dist; must_exist_in=buses_base)
weakest_bus = weak_bus

println("\nEV location buses:")
println("  source   = ", source_bus)
println("  mid      = ", mid_bus)
println("  far end  = ", far_bus)
println("  weakest  = ", weakest_bus)

# -----------------------
# Run sensitivity cases
# -----------------------
cases = [
    (name="ev_source",  bus=source_bus,  phase=weak_phase),
    (name="ev_mid",     bus=mid_bus,     phase=weak_phase),
    (name="ev_far",     bus=far_bus,     phase=weak_phase),
    (name="ev_weakest", bus=weakest_bus, phase=weak_phase)
]

summary_rows = NamedTuple[]

for c in cases
    println("\n==============================")
    println("Case: ", c.name, "  | bus=", c.bus, " | phase=", c.phase, " (", EV_PHASES_LABEL[c.phase], ")")
    println("==============================")

    math_case = deepcopy(math)

    # Add EV
    ev_id = add_single_phase_ev!(math_case, c.bus, c.phase; p_mw=ev_p_mw, q_mvar=ev_q_mvar)
    println("Added EV load id = ", ev_id, " | P=", EV_P_KW, " kW | PF=", EV_PF, " | Q≈", round(ev_q_mvar*1000, digits=2), " kvar")

    # Solve
    result = solve_mc_opf(math_case, IVRUPowerModel, ipopt_solver)
    status = string(result["termination_status"])
    println("Solve status: ", status)

    if status != "LOCALLY_SOLVED" && status != "OPTIMAL" && status != "ALMOST_LOCALLY_SOLVED"
        println("Skipping plots for this case because solver did not converge cleanly.")
        push!(summary_rows, (
            case=c.name, ev_bus=c.bus, ev_phase=EV_PHASES_LABEL[c.phase],
            status=status,
            minV_A=missing, minV_B=missing, minV_C=missing,
            minBus_A=missing, minBus_B=missing, minBus_C=missing,
            belowVMIN_A=missing, belowVMIN_B=missing, belowVMIN_C=missing,
            vuf_mean=missing, vuf_p95=missing, vuf_max=missing, vuf_max_bus=missing
        ))
        continue
    end

    # Extract voltages
    buses_ev = solved_bus_voltages(result, math_case; vbase_ln=VBASE_LN)
    for (bus, d) in dist
        haskey(buses_ev, bus) && (buses_ev[bus]["distance"] = d)
    end
    add_vuf!(buses_ev; t=1)

    # Voltage plots (EV only)
    p_ev_profile = plot_voltage_along_feeder_snap(buses_ev, lines_df; t=1, title_str="$(c.name): voltage drop (EV 7 kW, 1φ)")
    p_ev_hist    = plot_voltage_histogram_snap(buses_ev; t=1, title_str="$(c.name): voltage histogram")
    savefig(p_ev_profile, joinpath(figdir, "$(c.name)_voltage_profile.png"))
    savefig(p_ev_hist,    joinpath(figdir, "$(c.name)_voltage_histogram.png"))
    savefig(plot(p_ev_profile, p_ev_hist, layout=(1,2)), joinpath(figdir, "$(c.name)_voltage_combined.png"))

    # VUF plots (EV only)
    p_ev_vuf     = plot_vuf_along_feeder(buses_ev; t=1, title_str="$(c.name): VUF along feeder")
    p_ev_vufhist = plot_vuf_histogram(buses_ev; t=1, title_str="$(c.name): VUF histogram")
    savefig(p_ev_vuf,     joinpath(figdir, "$(c.name)_vuf_profile.png"))
    savefig(p_ev_vufhist, joinpath(figdir, "$(c.name)_vuf_histogram.png"))
    savefig(plot(p_ev_vuf, p_ev_vufhist, layout=(1,2)), joinpath(figdir, "$(c.name)_vuf_combined.png"))

    # Overlay: baseline vs EV (voltage profile)
    p_overlay = plot(legend=:topright)
    ylabel!("Voltage magnitude P-N (V)")
    xlabel!("Distance from reference bus (km)")
    title!("Overlay: baseline vs $(c.name) (EV 7 kW, 1φ @ $(c.bus))")

    colors = Dict(1=>:blue, 2=>:red, 3=>:black)
    for ph in 1:3
        key = ph == 1 ? "vma" : ph == 2 ? "vmb" : "vmc"

        xs = Float64[]
        ys_base = Float64[]
        ys_ev = Float64[]

        for (bus, d) in buses_base
            if haskey(d, "distance") && haskey(d, key) && haskey(buses_ev, bus) && haskey(buses_ev[bus], key)
                push!(xs, d["distance"])
                push!(ys_base, d[key][1])
                push!(ys_ev, buses_ev[bus][key][1])
            end
        end

        perm = sortperm(xs)
        xs = xs[perm]; ys_base = ys_base[perm]; ys_ev = ys_ev[perm]

        plot!(xs, ys_base, color=colors[ph], linestyle=:dash, label="baseline phase $(EV_PHASES_LABEL[ph])")
        plot!(xs, ys_ev,   color=colors[ph], linestyle=:solid, label="$(c.name) phase $(EV_PHASES_LABEL[ph])")
    end
    hline!([VMIN], color=:red, linestyle=:dot, label=false)
    hline!([VMAX], color=:red, linestyle=:dot, label=false)

    savefig(p_overlay, joinpath(figdir, "$(c.name)_overlay_voltage.png"))

    # Side-by-side 2x2 compare (baseline vs EV)
    p_compare = plot(
        p_base_profile, p_ev_profile,
        p_base_hist,    p_ev_hist,
        layout=(2,2),
        size=(1200,800)
    )
    plot!(p_compare, title="Compare: baseline vs $(c.name)")
    savefig(p_compare, joinpath(figdir, "$(c.name)_compare_voltage.png"))
    savefig(p_compare, joinpath(figdir, "$(c.name)_compare_voltage.pdf"))

    # VUF compare (baseline vs EV)
    p_compare_vuf = plot(
        p_base_vuf, p_ev_vuf,
        p_base_vufhist, p_ev_vufhist,
        layout=(2,2),
        size=(1200,800)
    )
    plot!(p_compare_vuf, title="Compare VUF: baseline vs $(c.name)")
    savefig(p_compare_vuf, joinpath(figdir, "$(c.name)_compare_vuf.png"))
    savefig(p_compare_vuf, joinpath(figdir, "$(c.name)_compare_vuf.pdf"))

    # Summary numbers
    mins_e, minbus_e, viols_e = voltage_summary(buses_ev; t=1, vmin=VMIN)
    vuf_e = vuf_summary(buses_ev; t=1)

    push!(summary_rows, (
        case=c.name,
        ev_bus=c.bus,
        ev_phase=EV_PHASES_LABEL[c.phase],
        status=status,
        minV_A=mins_e[1], minV_B=mins_e[2], minV_C=mins_e[3],
        minBus_A=string(minbus_e[1]), minBus_B=string(minbus_e[2]), minBus_C=string(minbus_e[3]),
        belowVMIN_A=viols_e[1], belowVMIN_B=viols_e[2], belowVMIN_C=viols_e[3],
        vuf_mean=vuf_e.mean, vuf_p95=vuf_e.p95, vuf_max=vuf_e.max, vuf_max_bus=string(vuf_e.max_bus)
    ))
end

# Also add baseline row to the same table
mins_b, minbus_b, viols_b = voltage_summary(buses_base; t=1, vmin=VMIN)
vuf_b = vuf_summary(buses_base; t=1)

baseline_row = (
    case="baseline",
    ev_bus="",
    ev_phase="",
    status=string(result_base["termination_status"]),
    minV_A=mins_b[1], minV_B=mins_b[2], minV_C=mins_b[3],
    minBus_A=string(minbus_b[1]), minBus_B=string(minbus_b[2]), minBus_C=string(minbus_b[3]),
    belowVMIN_A=viols_b[1], belowVMIN_B=viols_b[2], belowVMIN_C=viols_b[3],
    vuf_mean=vuf_b.mean, vuf_p95=vuf_b.p95, vuf_max=vuf_b.max, vuf_max_bus=string(vuf_b.max_bus)
)

summary_df = DataFrame([baseline_row; summary_rows...])

println("\n--- Sensitivity summary table ---")
@show(summary_df, allrows=true, allcols=true)

out_csv = joinpath(tabledir, "ev7kw_sensitivity_summary.csv")
CSV.write(out_csv, summary_df)

println("\nSaved figures to: ", figdir)
println("Saved summary table to: ", out_csv)
