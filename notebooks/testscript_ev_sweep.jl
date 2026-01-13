# 1) Solve BASELINE once (OPF)
# 2) Pick a critical location from baseline: weakest bus-phase (lowest voltage)
# 3) Severity sweep (single EV at fixed critical location):
#    Increase EV size step by step and re-solve OPF each time.
#    Save plots and a summary CSV. Also track VUF (sequence unbalance).
# 4) Interaction effects (two EVs):
#    Compare clustered vs distributed placements with same total EV power.
#    Save plots and a summary CSV.

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
    "print_level" => 1,
    "sb" => "yes",
    "warm_start_init_point" => "yes"
)

# Voltage base and limits (AU LV)
const VBASE_LN = 230.0
const VMIN = 0.94 * VBASE_LN
const VMAX = 1.10 * VBASE_LN

# Power factor setting for EV
# If you want Q=0 always, set EV_PF = 1.0
const EV_PF = 0.95

# Phase label helper
const PH_LABEL = Dict(1 => "a", 2 => "b", 3 => "c")

# Severity sweep sizes (kW)
# You can adjust this later.
# Tip: do coarse first, then refine around the threshold.
ev_kw_levels = [0.0, 3.5, 7.0, 11.0, 14.0, 18.0, 22.0]

# Two EV interaction settings (kW per EV)
# Keep this fixed so comparisons are fair.
const EV2_KW = 7.0

# Output folders
root_figdir = joinpath(@__DIR__, "..", "results", "figures")
root_tabledir = joinpath(@__DIR__, "..", "results", "tables")
mkpath(root_figdir)
mkpath(root_tabledir)

severity_figdir = joinpath(root_figdir, "severity")
interaction_figdir = joinpath(root_figdir, "interaction")
mkpath(severity_figdir)
mkpath(interaction_figdir)

# -----------------------
# Helper: build lines DataFrame + distances
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
        push!(rows, (Bus1 = f_bus, Bus2 = t_bus, phases = f_phases, length_km = length_km))
    end
    return DataFrame(rows)
end

function compute_bus_distances(lines_df; source_bus = "sourcebus")
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
# Helper: extract voltages (magnitudes in volts, and phasors in pu if present)
# -----------------------
function solved_bus_voltages(result_opf, math; vbase_ln = VBASE_LN)
    sol_bus = result_opf["solution"]["bus"]
    buses_dict = Dict{String, Dict{String,Any}}()

    for (bus_id, bus_data) in math["bus"]
        name = bus_data["name"]

        if !haskey(sol_bus, bus_id)
            continue
        end
        sb = sol_bus[bus_id]

        d = Dict{String,Any}()

        # magnitude
        vm_pu =
            haskey(sb, "vm") ? sb["vm"] :
            (haskey(sb, "vr") && haskey(sb, "vi")) ? sqrt.(sb["vr"].^2 .+ sb["vi"].^2) :
            nothing

        if vm_pu !== nothing && length(vm_pu) >= 3
            vmV = vm_pu .* vbase_ln
            d["vma"] = [vmV[1]]
            d["vmb"] = [vmV[2]]
            d["vmc"] = [vmV[3]]
        end

        # phasors in pu for sequence components
        if haskey(sb, "vr") && haskey(sb, "vi") && length(sb["vr"]) >= 3 && length(sb["vi"]) >= 3
            d["vra"] = [sb["vr"][1]]; d["via"] = [sb["vi"][1]]
            d["vrb"] = [sb["vr"][2]]; d["vib"] = [sb["vi"][2]]
            d["vrc"] = [sb["vr"][3]]; d["vic"] = [sb["vi"][3]]
        end

        if !isempty(d)
            buses_dict[name] = d
        end
    end

    return buses_dict
end

# -----------------------
# Symmetrical components and VUF
# -----------------------
function seq_components(Va::ComplexF64, Vb::ComplexF64, Vc::ComplexF64)
    a = cis(2π/3)
    V0 = (Va + Vb + Vc) / 3
    V1 = (Va + a*Vb + a^2*Vc) / 3
    V2 = (Va + a^2*Vb + a*Vc) / 3
    return V0, V1, V2
end

function add_vuf!(buses_dict; t = 1)
    for (bus, d) in buses_dict
        if all(haskey(d, k) for k in ["vra","via","vrb","vib","vrc","vic"])
            Va = ComplexF64(d["vra"][t], d["via"][t])
            Vb = ComplexF64(d["vrb"][t], d["vib"][t])
            Vc = ComplexF64(d["vrc"][t], d["vic"][t])

            V0, V1, V2 = seq_components(Va, Vb, Vc)
            vuf = (abs(V1) > 1e-9) ? abs(V2) / abs(V1) : NaN

            d["vuf"] = [vuf]
            d["v0mag"] = [abs(V0)]
            d["v1mag"] = [abs(V1)]
            d["v2mag"] = [abs(V2)]
        end
    end
    return buses_dict
end

function vuf_summary(buses_dict; t = 1)
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
        return (mean = missing, p95 = missing, max = missing, max_bus = missing)
    end

    sort!(vals)
    p95 = vals[clamp(Int(ceil(0.95*length(vals))), 1, length(vals))]
    return (mean = mean(vals), p95 = p95, max = maximum(vals), max_bus = worst_bus)
end

# -----------------------
# Plotting helpers (same spirit as your earlier scripts)
# -----------------------
function plot_voltage_along_feeder_snap(buses_dict, lines_df;
        t = 1, Vthreshold = 1000, vmin = VMIN, vmax = VMAX, title_str = "Voltage drop along feeder")

    p = plot(legend = false)
    ylabel!("Voltage magnitude P-N (V)")
    xlabel!("Distance from reference bus (km)")
    title!(title_str)

    colors = Dict(1 => :blue, 2 => :red, 3 => :black)

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
                    color = colors[ph],
                    marker = :circle,
                    markersize = 1
                )
            end
        end
    end

    maxdist = maximum(bus["distance"] for bus in values(buses_dict) if haskey(bus, "distance"))
    plot!([0, maxdist], [vmin, vmin], linestyle = :dash, color = :red)
    plot!([0, maxdist], [vmax, vmax], linestyle = :dash, color = :red)

    return p
end

function plot_voltage_histogram_snap(buses_dict; t = 1, Vthreshold = 1000, vmin = VMIN, vmax = VMAX, title_str = "Voltage histogram")
    phase_a = Float64[]
    phase_b = Float64[]
    phase_c = Float64[]

    for (bus, d) in buses_dict
        if haskey(d, "vma") && d["vma"][t] < Vthreshold; push!(phase_a, d["vma"][t]); end
        if haskey(d, "vmb") && d["vmb"][t] < Vthreshold; push!(phase_b, d["vmb"][t]); end
        if haskey(d, "vmc") && d["vmc"][t] < Vthreshold; push!(phase_c, d["vmc"][t]); end
    end

    bins = (vmin - 1):0.5:(vmax + 1)
    p = histogram(phase_a; bins, color = :blue, label = "phase a")
    histogram!(phase_b; bins, color = :red, label = "phase b")
    histogram!(phase_c; bins, color = :black, label = "phase c")
    ylabel!("Counts (-)")
    xlabel!("Voltage magnitude (V)")
    title!(title_str)

    vline!([vmin], color = :red, linestyle = :dash, label = false)
    vline!([vmax], color = :red, linestyle = :dash, label = false)

    return p
end

function plot_vuf_along_feeder(buses_dict; t = 1, title_str = "VUF along feeder (|V2|/|V1|)")
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

    p = plot(xs, ys, marker = :circle, markersize = 2, legend = false)
    xlabel!("Distance from reference bus (km)")
    ylabel!("VUF = |V2|/|V1| (-)")
    title!(title_str)
    return p
end

function plot_vuf_histogram(buses_dict; t = 1, title_str = "VUF histogram (|V2|/|V1|)")
    vals = Float64[]
    for (bus, d) in buses_dict
        if haskey(d, "vuf")
            v = d["vuf"][t]
            if isfinite(v)
                push!(vals, v)
            end
        end
    end
    p = histogram(vals, bins = 0:0.002:0.08, legend = false)
    xlabel!("VUF (-)")
    ylabel!("Counts (-)")
    title!(title_str)
    return p
end

# Overlay plot (baseline vs scenario)
function overlay_voltage_profile(buses_base, buses_case; title_str = "Overlay voltage profile")
    p = plot(legend = :topright)
    ylabel!("Voltage magnitude P-N (V)")
    xlabel!("Distance from reference bus (km)")
    title!(title_str)

    colors = Dict(1 => :blue, 2 => :red, 3 => :black)

    for ph in 1:3
        key = ph == 1 ? "vma" : ph == 2 ? "vmb" : "vmc"

        xs = Float64[]
        ys_base = Float64[]
        ys_case = Float64[]

        for (bus, d) in buses_base
            if haskey(d, "distance") && haskey(d, key) && haskey(buses_case, bus) && haskey(buses_case[bus], key)
                push!(xs, d["distance"])
                push!(ys_base, d[key][1])
                push!(ys_case, buses_case[bus][key][1])
            end
        end

        perm = sortperm(xs)
        xs = xs[perm]; ys_base = ys_base[perm]; ys_case = ys_case[perm]

        plot!(xs, ys_base, color = colors[ph], linestyle = :dash, label = "baseline phase $(PH_LABEL[ph])")
        plot!(xs, ys_case, color = colors[ph], linestyle = :solid, label = "case phase $(PH_LABEL[ph])")
    end

    hline!([VMIN], color = :red, linestyle = :dot, label = false)
    hline!([VMAX], color = :red, linestyle = :dot, label = false)

    return p
end

function overlay_vuf_profile(buses_base, buses_case; title_str = "Overlay VUF profile")
    # Build series of (distance, vuf)
    function get_series(buses)
        xs = Float64[]
        ys = Float64[]
        for (bus, d) in buses
            if haskey(d, "distance") && haskey(d, "vuf")
                v = d["vuf"][1]
                if isfinite(v)
                    push!(xs, d["distance"])
                    push!(ys, v)
                end
            end
        end
        perm = sortperm(xs)
        return xs[perm], ys[perm]
    end

    xb, yb = get_series(buses_base)
    xc, yc = get_series(buses_case)

    p = plot(xb, yb, linestyle = :dash, label = "baseline")
    plot!(p, xc, yc, linestyle = :solid, label = "case")
    xlabel!("Distance from reference bus (km)")
    ylabel!("VUF = |V2|/|V1| (-)")
    title!(title_str)
    return p
end

# -----------------------
# Summary helpers
# -----------------------
function voltage_summary(buses_dict; t = 1, vmin = VMIN)
    mins = Dict{Int, Float64}(1 => Inf, 2 => Inf, 3 => Inf)
    minbus = Dict{Int, Union{Missing,String}}(1 => missing, 2 => missing, 3 => missing)
    viols = Dict{Int, Int}(1 => 0, 2 => 0, 3 => 0)

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

function find_weakest_bus_and_phase(buses_base; t = 1)
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

function pick_mid_bus(dist::Dict{String,Float64}; must_exist_in = nothing)
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

# -----------------------
# EV load injection helpers
# -----------------------
function kw_to_mw(kW::Float64)
    return kW / 1000.0
end

function q_from_pf(p_mw::Float64, pf::Float64)
    if pf >= 0.999
        return 0.0
    end
    return p_mw * tan(acos(pf))
end

function next_numeric_id(tbl::Dict{String,Any})
    ids = Int[]
    for k in keys(tbl)
        x = tryparse(Int, k)
        x === nothing && continue
        push!(ids, x)
    end
    return isempty(ids) ? 1 : maximum(ids) + 1
end

function add_single_phase_ev!(math, ev_bus::String, phase::Int; p_mw::Float64, q_mvar::Float64, name::String = "ev")
    pd = zeros(3)
    qd = zeros(3)
    pd[phase] = p_mw
    qd[phase] = q_mvar

    isempty(math["load"]) && error("math model has no loads to copy schema from.")

    template_id, template = first(collect(math["load"]))
    new_load = deepcopy(template)

    new_id = string(next_numeric_id(math["load"]))

    new_load["bus"] = ev_bus
    new_load["pd"] = pd
    new_load["qd"] = qd
    new_load["status"] = 1
    new_load["name"] = "$(name)_$(ev_bus)_ph$(phase)"

    if haskey(new_load, "connections")
        new_load["connections"] = [1,2,3]
    end

    math["load"][new_id] = new_load
    return new_id
end

# -----------------------
# Run OPF and build buses_dict
# -----------------------
function run_opf_case(math_case, lines_df, dist; label = "case")
    result = solve_mc_opf(math_case, IVRUPowerModel, ipopt_solver)
    status = string(result["termination_status"])

    if !(haskey(result, "solution") && haskey(result["solution"], "bus"))
        return (status = status, result = result, buses = Dict{String, Dict{String,Any}}())
    end

    buses = solved_bus_voltages(result, math_case; vbase_ln = VBASE_LN)
    for (bus, d) in dist
        haskey(buses, bus) && (buses[bus]["distance"] = d)
    end
    add_vuf!(buses; t = 1)

    return (status = status, result = result, buses = buses)
end

# -----------------------
# Parse + transform data
# -----------------------
eng = parse_file(file, transformations = [transform_loops!, reduce_lines!])

eng["settings"]["sbase_default"] = 1
eng["voltage_source"]["source"]["rs"] .= 0
eng["voltage_source"]["source"]["xs"] .= 0

math = transform_data_model(eng, multinetwork = false, kron_reduce = true, phase_project = true)

lines_df = make_lines_df_from_eng(eng)
dist = compute_bus_distances(lines_df; source_bus = "sourcebus")

# -----------------------
# BASELINE solve
# -----------------------
println("Running BASELINE OPF...")
base = run_opf_case(math, lines_df, dist; label = "baseline")
println("BASELINE status: ", base.status)

buses_base = base.buses
mins_b, minbus_b, viols_b = voltage_summary(buses_base; t = 1, vmin = VMIN)
vuf_b = vuf_summary(buses_base; t = 1)

# Save baseline plots
p_base_profile = plot_voltage_along_feeder_snap(buses_base, lines_df; t = 1, title_str = "Baseline: voltage profile")
p_base_hist = plot_voltage_histogram_snap(buses_base; t = 1, title_str = "Baseline: voltage histogram")
savefig(p_base_profile, joinpath(root_figdir, "baseline_voltage_profile.png"))
savefig(p_base_hist, joinpath(root_figdir, "baseline_voltage_histogram.png"))
savefig(plot(p_base_profile, p_base_hist, layout = (1,2)), joinpath(root_figdir, "baseline_voltage_combined.png"))

p_base_vuf = plot_vuf_along_feeder(buses_base; t = 1, title_str = "Baseline: VUF profile")
p_base_vufhist = plot_vuf_histogram(buses_base; t = 1, title_str = "Baseline: VUF histogram")
savefig(p_base_vuf, joinpath(root_figdir, "baseline_vuf_profile.png"))
savefig(p_base_vufhist, joinpath(root_figdir, "baseline_vuf_histogram.png"))
savefig(plot(p_base_vuf, p_base_vufhist, layout = (1,2)), joinpath(root_figdir, "baseline_vuf_combined.png"))

# Find critical location from baseline
weak_bus, weak_phase, weak_v = find_weakest_bus_and_phase(buses_base; t = 1)
println("\nBaseline weakest point:")
println("  weakest bus = ", weak_bus)
println("  weakest phase = ", weak_phase, " (", PH_LABEL[weak_phase], ")")
println("  min voltage = ", round(weak_v, digits = 2), " V")

# Some helpful buses for interaction study
mid_bus = pick_mid_bus(dist; must_exist_in = buses_base)
println("  mid feeder bus = ", mid_bus)

# -----------------------
# Part 1: Severity sweep (single EV at weakest bus-phase)
# -----------------------
println("\n==============================")
println("Severity sweep: single EV at weakest bus-phase")
println("==============================")

severity_rows = NamedTuple[]

for ev_kw in ev_kw_levels
    case_name = "sev_ev_$(replace(string(ev_kw), "." => "p"))kw"
    case_dir = joinpath(severity_figdir, case_name)
    mkpath(case_dir)

    # Build a fresh math copy each time
    math_case = deepcopy(math)

    # Add EV only if ev_kw > 0
    if ev_kw > 1e-9
        p_mw = kw_to_mw(ev_kw)
        q_mvar = q_from_pf(p_mw, EV_PF)
        add_single_phase_ev!(math_case, string(weak_bus), weak_phase; p_mw = p_mw, q_mvar = q_mvar, name = "ev")
    end

    out = run_opf_case(math_case, lines_df, dist; label = case_name)
    println("  ", case_name, " | status = ", out.status)

    buses_case = out.buses

    # Summaries
    mins, minbus, viols = voltage_summary(buses_case; t = 1, vmin = VMIN)
    vuf_s = vuf_summary(buses_case; t = 1)

    # Save main plots (voltage + VUF)
    if !isempty(buses_case)
        p_prof = plot_voltage_along_feeder_snap(buses_case, lines_df; t = 1, title_str = "$(case_name): voltage profile")
        p_hist = plot_voltage_histogram_snap(buses_case; t = 1, title_str = "$(case_name): voltage histogram")
        savefig(p_prof, joinpath(case_dir, "voltage_profile.png"))
        savefig(p_hist, joinpath(case_dir, "voltage_histogram.png"))
        savefig(plot(p_prof, p_hist, layout = (1,2)), joinpath(case_dir, "voltage_combined.png"))

        p_ov = overlay_voltage_profile(buses_base, buses_case; title_str = "Overlay: baseline vs $(case_name)")
        savefig(p_ov, joinpath(case_dir, "overlay_voltage.png"))

        # VUF plots (if phasors exist)
        p_vuf = plot_vuf_along_feeder(buses_case; t = 1, title_str = "$(case_name): VUF profile")
        p_vufh = plot_vuf_histogram(buses_case; t = 1, title_str = "$(case_name): VUF histogram")
        savefig(p_vuf, joinpath(case_dir, "vuf_profile.png"))
        savefig(p_vufh, joinpath(case_dir, "vuf_histogram.png"))
        savefig(plot(p_vuf, p_vufh, layout = (1,2)), joinpath(case_dir, "vuf_combined.png"))

        p_ov_vuf = overlay_vuf_profile(buses_base, buses_case; title_str = "Overlay VUF: baseline vs $(case_name)")
        savefig(p_ov_vuf, joinpath(case_dir, "overlay_vuf.png"))
    end

    push!(severity_rows, (
        case = case_name,
        ev_kw = ev_kw,
        ev_bus = string(weak_bus),
        ev_phase = PH_LABEL[weak_phase],
        status = out.status,
        minV_A = mins[1], minV_B = mins[2], minV_C = mins[3],
        minBus_A = string(minbus[1]), minBus_B = string(minbus[2]), minBus_C = string(minbus[3]),
        belowVMIN_A = viols[1], belowVMIN_B = viols[2], belowVMIN_C = viols[3],
        vuf_mean = vuf_s.mean, vuf_p95 = vuf_s.p95, vuf_max = vuf_s.max, vuf_max_bus = string(vuf_s.max_bus)
    ))
end

severity_df = DataFrame(severity_rows)

# Add baseline row for reference
baseline_row = (
    case = "baseline",
    ev_kw = 0.0,
    ev_bus = "",
    ev_phase = "",
    status = base.status,
    minV_A = mins_b[1], minV_B = mins_b[2], minV_C = mins_b[3],
    minBus_A = string(minbus_b[1]), minBus_B = string(minbus_b[2]), minBus_C = string(minbus_b[3]),
    belowVMIN_A = viols_b[1], belowVMIN_B = viols_b[2], belowVMIN_C = viols_b[3],
    vuf_mean = vuf_b.mean, vuf_p95 = vuf_b.p95, vuf_max = vuf_b.max, vuf_max_bus = string(vuf_b.max_bus)
)
severity_df = vcat(DataFrame([baseline_row]), severity_df)

CSV.write(joinpath(root_tabledir, "severity_single_ev_sweep.csv"), severity_df)

# Quick severity summary plots (threshold style)
# 1) min voltage vs EV kW (overall minimum across phases)
function overall_min_voltage_row(row)
    vals = Float64[]
    for k in [:minV_A, :minV_B, :minV_C]
        v = row[k]
        if v isa Missing
            continue
        end
        push!(vals, Float64(v))
    end
    return isempty(vals) ? NaN : minimum(vals)
end

sev_nonbase = severity_df[severity_df.case .!= "baseline", :]

ev_kw = Float64.(sev_nonbase.ev_kw)
minV_overall = [overall_min_voltage_row(sev_nonbase[i, :]) for i in 1:nrow(sev_nonbase)]
vuf_max_series = [sev_nonbase.vuf_max[i] isa Missing ? NaN : Float64(sev_nonbase.vuf_max[i]) for i in 1:nrow(sev_nonbase)]
below_total = [
    (sev_nonbase.belowVMIN_A[i] isa Missing ? 0 : Int(sev_nonbase.belowVMIN_A[i])) +
    (sev_nonbase.belowVMIN_B[i] isa Missing ? 0 : Int(sev_nonbase.belowVMIN_B[i])) +
    (sev_nonbase.belowVMIN_C[i] isa Missing ? 0 : Int(sev_nonbase.belowVMIN_C[i]))
    for i in 1:nrow(sev_nonbase)
]

p_minV = plot(ev_kw, minV_overall, marker = :circle, legend = false)
xlabel!("EV size (kW)")
ylabel!("Overall minimum voltage (V)")
title!("Severity sweep: overall minimum voltage vs EV size")
hline!([VMIN], linestyle = :dash, label = false)
savefig(p_minV, joinpath(severity_figdir, "severity_min_voltage_vs_kw.png"))

p_vufmax = plot(ev_kw, vuf_max_series, marker = :circle, legend = false)
xlabel!("EV size (kW)")
ylabel!("Max VUF (-)")
title!("Severity sweep: max VUF vs EV size")
savefig(p_vufmax, joinpath(severity_figdir, "severity_max_vuf_vs_kw.png"))

p_viol = plot(ev_kw, below_total, marker = :circle, legend = false)
xlabel!("EV size (kW)")
ylabel!("Total count below VMIN (A+B+C)")
title!("Severity sweep: total voltage violations vs EV size")
savefig(p_viol, joinpath(severity_figdir, "severity_total_violations_vs_kw.png"))

# -----------------------
# Part 2: Interaction effects (two EVs)
# -----------------------
println("\n==============================")
println("Interaction effects: two EVs")
println("==============================")

interaction_cases = [
    # Clustered: two EVs at weakest bus on same phase (strong unbalance case)
    (name = "twoev_clustered_samephase",
     evs = [(bus = string(weak_bus), phase = weak_phase, kw = EV2_KW),
            (bus = string(weak_bus), phase = weak_phase, kw = EV2_KW)]),

    # Distributed: one EV at source, one at weakest, same phase as weak_phase
    (name = "twoev_distributed_source_plus_weakest",
     evs = [(bus = "sourcebus", phase = weak_phase, kw = EV2_KW),
            (bus = string(weak_bus), phase = weak_phase, kw = EV2_KW)]),

    # Spread across mid + weakest (often more realistic than source)
    (name = "twoev_distributed_mid_plus_weakest",
     evs = [(bus = string(mid_bus), phase = weak_phase, kw = EV2_KW),
            (bus = string(weak_bus), phase = weak_phase, kw = EV2_KW)])
]

interaction_rows = NamedTuple[]

for ic in interaction_cases
    case_dir = joinpath(interaction_figdir, ic.name)
    mkpath(case_dir)

    math_case = deepcopy(math)

    # Add the two EVs
    for (j, ev) in enumerate(ic.evs)
        p_mw = kw_to_mw(Float64(ev.kw))
        q_mvar = q_from_pf(p_mw, EV_PF)
        add_single_phase_ev!(math_case, string(ev.bus), Int(ev.phase); p_mw = p_mw, q_mvar = q_mvar, name = "ev$(j)")
    end

    out = run_opf_case(math_case, lines_df, dist; label = ic.name)
    println("  ", ic.name, " | status = ", out.status)

    buses_case = out.buses

    mins, minbus, viols = voltage_summary(buses_case; t = 1, vmin = VMIN)
    vuf_s = vuf_summary(buses_case; t = 1)

    if !isempty(buses_case)
        p_prof = plot_voltage_along_feeder_snap(buses_case, lines_df; t = 1, title_str = "$(ic.name): voltage profile")
        p_hist = plot_voltage_histogram_snap(buses_case; t = 1, title_str = "$(ic.name): voltage histogram")
        savefig(p_prof, joinpath(case_dir, "voltage_profile.png"))
        savefig(p_hist, joinpath(case_dir, "voltage_histogram.png"))
        savefig(plot(p_prof, p_hist, layout = (1,2)), joinpath(case_dir, "voltage_combined.png"))

        p_ov = overlay_voltage_profile(buses_base, buses_case; title_str = "Overlay: baseline vs $(ic.name)")
        savefig(p_ov, joinpath(case_dir, "overlay_voltage.png"))

        # 2x2 compare (baseline vs case)
        p_compare = plot(
            p_base_profile, p_prof,
            p_base_hist, p_hist,
            layout = (2,2),
            size = (1200, 800)
        )
        savefig(p_compare, joinpath(case_dir, "compare_voltage.png"))
        savefig(p_compare, joinpath(case_dir, "compare_voltage.pdf"))

        # VUF compare
        p_vuf = plot_vuf_along_feeder(buses_case; t = 1, title_str = "$(ic.name): VUF profile")
        p_vufh = plot_vuf_histogram(buses_case; t = 1, title_str = "$(ic.name): VUF histogram")
        savefig(p_vuf, joinpath(case_dir, "vuf_profile.png"))
        savefig(p_vufh, joinpath(case_dir, "vuf_histogram.png"))
        savefig(plot(p_vuf, p_vufh, layout = (1,2)), joinpath(case_dir, "vuf_combined.png"))

        p_ov_vuf = overlay_vuf_profile(buses_base, buses_case; title_str = "Overlay VUF: baseline vs $(ic.name)")
        savefig(p_ov_vuf, joinpath(case_dir, "overlay_vuf.png"))
    end

    # Record EV details in a readable string
    ev_desc = join(["(bus=$(e.bus), ph=$(PH_LABEL[Int(e.phase)]), kw=$(e.kw))" for e in ic.evs], " + ")

    push!(interaction_rows, (
        case = ic.name,
        evs = ev_desc,
        status = out.status,
        minV_A = mins[1], minV_B = mins[2], minV_C = mins[3],
        minBus_A = string(minbus[1]), minBus_B = string(minbus[2]), minBus_C = string(minbus[3]),
        belowVMIN_A = viols[1], belowVMIN_B = viols[2], belowVMIN_C = viols[3],
        vuf_mean = vuf_s.mean, vuf_p95 = vuf_s.p95, vuf_max = vuf_s.max, vuf_max_bus = string(vuf_s.max_bus)
    ))
end

interaction_df = DataFrame(interaction_rows)

# Include baseline row in interaction table too
interaction_df = vcat(DataFrame([(
    case = "baseline",
    evs = "",
    status = base.status,
    minV_A = mins_b[1], minV_B = mins_b[2], minV_C = mins_b[3],
    minBus_A = string(minbus_b[1]), minBus_B = string(minbus_b[2]), minBus_C = string(minbus_b[3]),
    belowVMIN_A = viols_b[1], belowVMIN_B = viols_b[2], belowVMIN_C = viols_b[3],
    vuf_mean = vuf_b.mean, vuf_p95 = vuf_b.p95, vuf_max = vuf_b.max, vuf_max_bus = string(vuf_b.max_bus)
)]), interaction_df)

CSV.write(joinpath(root_tabledir, "interaction_two_ev_cases.csv"), interaction_df)

println("\nDone.")
println("Saved severity figures to: ", severity_figdir)
println("Saved interaction figures to: ", interaction_figdir)
println("Saved severity table to: ", joinpath(root_tabledir, "severity_single_ev_sweep.csv"))
println("Saved interaction table to: ", joinpath(root_tabledir, "interaction_two_ev_cases.csv"))
