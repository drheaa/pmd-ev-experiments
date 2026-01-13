# Congestion identification that later STATCOM script can consume cleanly.

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

# -----------------------
# User settings
# -----------------------
file = "/mnt/c/Users/auc009/OneDrive - CSIRO/Documents/power-models-distribution/pmd_ev_experiments/data/Three-wire-Kron-reduced/network_1/Feeder_1/Master.dss"

ipopt_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 1,
    "sb" => "yes",
    "warm_start_init_point" => "yes"
)

# Voltage base and limits (per-unit)
const VBASE_LN = 230.0
const VMIN_PU = 0.94
const VMAX_PU = 1.10

# VUF limit (typical planning thresholds are often around 2% to 3%)
const VUF_LIMIT = 0.02

# EV settings
const EV_PF = 0.95
const PH_LABEL = Dict(1 => "a", 2 => "b", 3 => "c")

# Severity sweep EV sizes (kW), single EV at critical location
ev_kw_levels = [0.0, 3.5, 7.0, 11.0, 14.0, 18.0, 22.0]

# Two-EV interaction settings (kW per EV)
const EV2_KW = 7.0

# Output dirs
root_dir = joinpath(@__DIR__, "..", "results", "congestion_identification")
figdir = joinpath(root_dir, "figures")
tabledir = joinpath(root_dir, "tables")
mkpath(figdir)
mkpath(tabledir)

# -----------------------
# Helpers: network lines + distances
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
# Helpers: EV conversions and injection
# -----------------------
kw_to_mw(kW::Float64) = kW / 1000.0

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

    _, template = first(collect(math["load"]))
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
# Helpers: voltages (pu + volts for plotting) and VUF
# -----------------------
"""
Returns buses_dict keyed by bus name:
- vma_pu/vmb_pu/vmc_pu (pu, snapshot in length-1 vector)
- vma_V/vmb_V/vmc_V (volts)
- vr*/vi* (pu phasors if available)
- distance (km) added later
- vuf (if computed)
"""
function solved_bus_voltages(result_opf, math; vbase_ln = VBASE_LN)
    sol_bus = result_opf["solution"]["bus"]
    buses = Dict{String, Dict{String,Any}}()

    for (bus_id, bus_data) in math["bus"]
        name = bus_data["name"]
        if !haskey(sol_bus, bus_id)
            continue
        end

        sb = sol_bus[bus_id]
        d = Dict{String,Any}()

        # magnitude in pu
        vm_pu =
            haskey(sb, "vm") ? sb["vm"] :
            (haskey(sb, "vr") && haskey(sb, "vi")) ? sqrt.(sb["vr"].^2 .+ sb["vi"].^2) :
            nothing

        if vm_pu !== nothing && length(vm_pu) >= 3
            d["vma_pu"] = [vm_pu[1]]
            d["vmb_pu"] = [vm_pu[2]]
            d["vmc_pu"] = [vm_pu[3]]

            vmV = vm_pu .* vbase_ln
            d["vma_V"] = [vmV[1]]
            d["vmb_V"] = [vmV[2]]
            d["vmc_V"] = [vmV[3]]
        end

        # phasors in pu
        if haskey(sb, "vr") && haskey(sb, "vi") && length(sb["vr"]) >= 3 && length(sb["vi"]) >= 3
            d["vra"] = [sb["vr"][1]]; d["via"] = [sb["vi"][1]]
            d["vrb"] = [sb["vr"][2]]; d["vib"] = [sb["vi"][2]]
            d["vrc"] = [sb["vr"][3]]; d["vic"] = [sb["vi"][3]]
        end

        if !isempty(d)
            buses[name] = d
        end
    end

    return buses
end

function seq_components(Va::ComplexF64, Vb::ComplexF64, Vc::ComplexF64)
    a = cis(2π/3)
    V0 = (Va + Vb + Vc) / 3
    V1 = (Va + a*Vb + a^2*Vc) / 3
    V2 = (Va + a^2*Vb + a*Vc) / 3
    return V0, V1, V2
end

function add_vuf!(buses; t = 1)
    for (bus, d) in buses
        if all(haskey(d, k) for k in ["vra","via","vrb","vib","vrc","vic"])
            Va = ComplexF64(d["vra"][t], d["via"][t])
            Vb = ComplexF64(d["vrb"][t], d["vib"][t])
            Vc = ComplexF64(d["vrc"][t], d["vic"][t])

            _, V1, V2 = seq_components(Va, Vb, Vc)
            vuf = (abs(V1) > 1e-9) ? abs(V2)/abs(V1) : NaN
            d["vuf"] = [vuf]
        end
    end
    return buses
end

# -----------------------
# Helpers: congestion checks, margins, regions
# -----------------------
function phase_keys_pu(ph::Int)
    return ph == 1 ? "vma_pu" : ph == 2 ? "vmb_pu" : "vmc_pu"
end

function compute_bus_metrics(buses; t = 1, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU, vuf_limit = VUF_LIMIT)
    rows = NamedTuple[]
    for (bus, d) in buses
        dist_km = haskey(d, "distance") ? d["distance"] : missing

        # per-phase fields
        va = haskey(d, "vma_pu") ? d["vma_pu"][t] : missing
        vb = haskey(d, "vmb_pu") ? d["vmb_pu"][t] : missing
        vc = haskey(d, "vmc_pu") ? d["vmc_pu"][t] : missing

        function margins(v)
            if v isa Missing
                return (missing, missing, missing, missing, missing)
            end
            margin_low = v - vmin_pu
            margin_high = vmax_pu - v
            risk = min(margin_low, margin_high)
            viol_low = v < vmin_pu
            viol_high = v > vmax_pu
            return (margin_low, margin_high, risk, viol_low, viol_high)
        end

        mla, mha, ra, vla, vha = margins(va)
        mlb, mhb, rb, vlb, vhb = margins(vb)
        mlc, mhc, rc, vlc, vhc = margins(vc)

        # bus risk = min risk across available phases
        risks = Float64[]
        for r in (ra, rb, rc)
            if !(r isa Missing)
                push!(risks, Float64(r))
            end
        end
        bus_risk = isempty(risks) ? missing : minimum(risks)

        vuf = haskey(d, "vuf") ? d["vuf"][t] : missing
        vuf_viol = (vuf isa Missing) ? missing : (isfinite(vuf) && vuf > vuf_limit)

        push!(rows, (
            bus = bus,
            distance_km = dist_km,
            va_pu = va, vb_pu = vb, vc_pu = vc,

            margin_low_a = mla, margin_high_a = mha, risk_a = ra, viol_a_low = vla, viol_a_high = vha,
            margin_low_b = mlb, margin_high_b = mhb, risk_b = rb, viol_b_low = vlb, viol_b_high = vhb,
            margin_low_c = mlc, margin_high_c = mhc, risk_c = rc, viol_c_low = vlc, viol_c_high = vhc,

            bus_risk = bus_risk,
            vuf = vuf,
            vuf_viol = vuf_viol
        ))
    end
    return DataFrame(rows)
end

function region_from_flags(df::DataFrame, flag_col::Symbol; dist_col::Symbol = :distance_km)
    # Returns (start_km, end_km, length_km, reach_km, count_buses)
    sub = df[.!ismissing.(df[!, flag_col]) .&& (df[!, flag_col] .== true) .&& .!ismissing.(df[!, dist_col]), :]
    if nrow(sub) == 0
        return (missing, missing, 0.0, missing, 0)
    end
    dists = Float64.(sub[!, dist_col])
    start_km = minimum(dists)
    end_km = maximum(dists)
    return (start_km, end_km, end_km - start_km, start_km, nrow(sub))
end

function extreme_voltage(df::DataFrame)
    # Finds global min/max among va/vb/vc pu, returns:
    # (vmin_pu, bus, phase), (vmax_pu, bus, phase)
    vmin = Inf
    vmin_bus = missing
    vmin_phase = missing

    vmax = -Inf
    vmax_bus = missing
    vmax_phase = missing

    for i in 1:nrow(df)
        b = df.bus[i]
        vals = [
            ("a", df.va_pu[i]),
            ("b", df.vb_pu[i]),
            ("c", df.vc_pu[i])
        ]
        for (ph, v) in vals
            if v isa Missing
                continue
            end
            vv = Float64(v)
            if vv < vmin
                vmin = vv
                vmin_bus = b
                vmin_phase = ph
            end
            if vv > vmax
                vmax = vv
                vmax_bus = b
                vmax_phase = ph
            end
        end
    end

    if vmin == Inf
        vmin = missing
    end
    if vmax == -Inf
        vmax = missing
    end

    return (vmin, vmin_bus, vmin_phase), (vmax, vmax_bus, vmax_phase)
end

function vuf_extreme(df::DataFrame)
    # Returns (vuf_max, bus, dist_km, n_vuf_viol)
    vuf_max = -Inf
    vuf_bus = missing
    vuf_dist = missing
    nviol = 0

    for i in 1:nrow(df)
        v = df.vuf[i]
        if !(v isa Missing) && isfinite(v)
            vv = Float64(v)
            if vv > vuf_max
                vuf_max = vv
                vuf_bus = df.bus[i]
                vuf_dist = df.distance_km[i]
            end
            if df.vuf_viol[i] == true
                nviol += 1
            end
        end
    end

    if vuf_max == -Inf
        vuf_max = missing
    end

    return (vuf_max, vuf_bus, vuf_dist, nviol)
end

function count_voltage_violations(df::DataFrame)
    # n_phase_viol = count of (bus,phase) entries violating
    # n_bus_viol = count of buses with any phase violating
    nphase = 0
    nbus = 0

    for i in 1:nrow(df)
        v_any = false
        for (low, high) in ((df.viol_a_low[i], df.viol_a_high[i]),
                            (df.viol_b_low[i], df.viol_b_high[i]),
                            (df.viol_c_low[i], df.viol_c_high[i]))
            if low == true || high == true
                nphase += 1
                v_any = true
            end
        end
        if v_any
            nbus += 1
        end
    end
    return nphase, nbus
end

# -----------------------
# Reactive limit checks (gen qmin/qmax binding)
# -----------------------
function gen_q_metrics(result_opf, math; eps = 1e-6)
    rows = NamedTuple[]

    if !(haskey(result_opf, "solution") && haskey(result_opf["solution"], "gen") && haskey(math, "gen"))
        return DataFrame(rows), 0, 0, 0
    end

    sol_gen = result_opf["solution"]["gen"]
    math_gen = math["gen"]

    total = 0
    atmin = 0
    atmax = 0

    for (gid, gdata) in math_gen
        if !haskey(sol_gen, gid)
            continue
        end

        sg = sol_gen[gid]
        # PMD stores qg as a vector by phase; for limits too
        qg = get(sg, "qg", missing)
        qmin = get(gdata, "qmin", missing)
        qmax = get(gdata, "qmax", missing)

        # reduce vector -> scalar check (max absolute phase-wise)
        function as_scalar(x)
            if x isa Missing
                return missing
            end
            if x isa AbstractVector
                # take sum as a simple proxy (or maximum). Here: sum.
                return sum(Float64.(x))
            end
            return Float64(x)
        end

        qg_s = as_scalar(qg)
        qmin_s = as_scalar(qmin)
        qmax_s = as_scalar(qmax)

        at_qmin = false
        at_qmax = false

        if !(qg_s isa Missing) && !(qmin_s isa Missing)
            at_qmin = (qg_s <= qmin_s + eps)
        end
        if !(qg_s isa Missing) && !(qmax_s isa Missing)
            at_qmax = (qg_s >= qmax_s - eps)
        end

        total += 1
        atmin += at_qmin ? 1 : 0
        atmax += at_qmax ? 1 : 0

        push!(rows, (
            gen_id = gid,
            bus = get(gdata, "bus", missing),
            qg = qg_s,
            qmin = qmin_s,
            qmax = qmax_s,
            at_qmin = at_qmin,
            at_qmax = at_qmax
        ))
    end

    return DataFrame(rows), total, atmin, atmax
end

# -----------------------
# Congestion plots
# -----------------------
function plot_congestion_scatter_voltage(df::DataFrame; scenario_id::String, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU)
    p = plot(legend = :topright)
    xlabel!("Distance from source (km)")
    ylabel!("Voltage magnitude (pu)")
    title!("Congestion view (voltage): $(scenario_id)")

    # plot each phase as scatter, mark violators larger
    for (ph, colv, lowc, highc) in (
        ("a", :va_pu, :viol_a_low, :viol_a_high),
        ("b", :vb_pu, :viol_b_low, :viol_b_high),
        ("c", :vc_pu, :viol_c_low, :viol_c_high)
    )
        xs = Float64[]
        ys = Float64[]
        xsv = Float64[]
        ysv = Float64[]

        for i in 1:nrow(df)
            v = df[i, colv]
            d = df[i, :distance_km]
            if v isa Missing || d isa Missing
                continue
            end
            if df[i, lowc] == true || df[i, highc] == true
                push!(xsv, Float64(d)); push!(ysv, Float64(v))
            else
                push!(xs, Float64(d)); push!(ys, Float64(v))
            end
        end

        scatter!(xs, ys, markersize = 2, label = "phase $(ph)")
        scatter!(xsv, ysv, markersize = 4, label = "phase $(ph) viol")
    end

    hline!([vmin_pu], linestyle = :dash, label = false)
    hline!([vmax_pu], linestyle = :dash, label = false)

    return p
end

function plot_congestion_scatter_margin(df::DataFrame; scenario_id::String)
    # bus_risk vs distance: lower is worse (closer to violating)
    xs = Float64[]
    ys = Float64[]
    for i in 1:nrow(df)
        d = df.distance_km[i]
        r = df.bus_risk[i]
        if d isa Missing || r isa Missing
            continue
        end
        push!(xs, Float64(d))
        push!(ys, Float64(r))
    end

    p = scatter(xs, ys, markersize = 3, legend = false)
    xlabel!("Distance from source (km)")
    ylabel!("Bus risk = min(V - VMIN, VMAX - V) (pu)")
    title!("Congestion view (margin): $(scenario_id)")
    hline!([0.0], linestyle = :dash, label = false)
    return p
end

function plot_congestion_scatter_vuf(df::DataFrame; scenario_id::String, vuf_limit = VUF_LIMIT)
    xs = Float64[]
    ys = Float64[]
    xsv = Float64[]
    ysv = Float64[]

    for i in 1:nrow(df)
        d = df.distance_km[i]
        v = df.vuf[i]
        if d isa Missing || v isa Missing || !isfinite(v)
            continue
        end
        if df.vuf_viol[i] == true
            push!(xsv, Float64(d)); push!(ysv, Float64(v))
        else
            push!(xs, Float64(d)); push!(ys, Float64(v))
        end
    end

    p = scatter(xs, ys, markersize = 3, label = "VUF")
    scatter!(xsv, ysv, markersize = 5, label = "VUF viol")
    xlabel!("Distance from source (km)")
    ylabel!("VUF = |V2|/|V1| (-)")
    title!("Congestion view (VUF): $(scenario_id)")
    hline!([vuf_limit], linestyle = :dash, label = false)
    return p
end

# -----------------------
# Run OPF case and package outputs
# -----------------------
function run_case(scenario_id::String, scenario_type::String, ev_desc::String, math_case, lines_df, dist)
    result = solve_mc_opf(math_case, IVRUPowerModel, ipopt_solver)
    status = string(result["termination_status"])

    buses = Dict{String, Dict{String,Any}}()
    if haskey(result, "solution") && haskey(result["solution"], "bus")
        buses = solved_bus_voltages(result, math_case; vbase_ln = VBASE_LN)
        for (bus, d) in dist
            haskey(buses, bus) && (buses[bus]["distance"] = d)
        end
        add_vuf!(buses; t = 1)
    end

    # Bus metrics
    df_bus = compute_bus_metrics(buses; t = 1, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU, vuf_limit = VUF_LIMIT)

    # Voltage extremes, counts
    (vmin_pu, vmin_bus, vmin_phase), (vmax_pu, vmax_bus, vmax_phase) = extreme_voltage(df_bus)
    n_phase_viol, n_bus_viol = count_voltage_violations(df_bus)

    worst_uv = (vmin_pu isa Missing) ? missing : max(0.0, VMIN_PU - Float64(vmin_pu))
    worst_ov = (vmax_pu isa Missing) ? missing : max(0.0, Float64(vmax_pu) - VMAX_PU)

    # Voltage congestion region (any phase violation)
    any_viol = Vector{Union{Missing,Bool}}(undef, nrow(df_bus))
    for i in 1:nrow(df_bus)
        any_viol[i] =
            (df_bus.viol_a_low[i] == true || df_bus.viol_a_high[i] == true ||
             df_bus.viol_b_low[i] == true || df_bus.viol_b_high[i] == true ||
             df_bus.viol_c_low[i] == true || df_bus.viol_c_high[i] == true)
    end
    df_bus[!, :any_voltage_viol] = any_viol
    region_start_km, region_end_km, region_len_km, reach_km, _ = region_from_flags(df_bus, :any_voltage_viol)

    # VUF region
    vuf_start, vuf_end, vuf_len, vuf_reach, _ = region_from_flags(df_bus, :vuf_viol)
    vuf_max, vuf_bus, vuf_dist, n_vuf_viol = vuf_extreme(df_bus)

    # Gen Q limit metrics
    df_gen, gen_total, gen_atmin, gen_atmax = gen_q_metrics(result, math_case)

    return (
        scenario_id = scenario_id,
        scenario_type = scenario_type,
        ev_desc = ev_desc,
        status = status,
        result = result,
        buses = buses,
        df_bus = df_bus,
        df_gen = df_gen,
        summary = (
            scenario_id = scenario_id,
            scenario_type = scenario_type,
            ev_desc = ev_desc,
            status = status,

            vmin_pu = vmin_pu,
            vmin_bus = vmin_bus,
            vmin_phase = vmin_phase,
            vmax_pu = vmax_pu,
            vmax_bus = vmax_bus,
            vmax_phase = vmax_phase,

            n_phase_viol = n_phase_viol,
            n_bus_viol = n_bus_viol,
            worst_undervoltage_pu = worst_uv,
            worst_overvoltage_pu = worst_ov,

            region_start_km = region_start_km,
            region_end_km = region_end_km,
            region_length_km = region_len_km,
            reach_km = reach_km,

            vuf_limit = VUF_LIMIT,
            vuf_max = vuf_max,
            vuf_max_bus = vuf_bus,
            vuf_max_dist_km = vuf_dist,
            n_vuf_viol = n_vuf_viol,

            vuf_region_start_km = vuf_start,
            vuf_region_end_km = vuf_end,
            vuf_region_length_km = vuf_len,
            vuf_reach_km = vuf_reach,

            gen_q_total = gen_total,
            gen_q_at_qmin = gen_atmin,
            gen_q_at_qmax = gen_atmax
        )
    )
end

# -----------------------
# Build base network (parse + transform)
# -----------------------
eng = parse_file(file, transformations = [transform_loops!, reduce_lines!])
eng["settings"]["sbase_default"] = 1
eng["voltage_source"]["source"]["rs"] .= 0
eng["voltage_source"]["source"]["xs"] .= 0

math = transform_data_model(eng, multinetwork = false, kron_reduce = true, phase_project = true)
lines_df = make_lines_df_from_eng(eng)
dist = compute_bus_distances(lines_df; source_bus = "sourcebus")

# -----------------------
# Baseline
# -----------------------
baseline = run_case("baseline", "baseline", "", deepcopy(math), lines_df, dist)

# Save baseline “standard” plots (reusing the usual style, but through bus metrics)
# Voltage profile by feeder lines still needs line connection plotting; for congestion ID we focus on scatter views.
# We'll save congestion scatter plots for baseline and a simple vuf plot.

# Baseline congestion plots
p_b_v = plot_congestion_scatter_voltage(baseline.df_bus; scenario_id = "baseline", vmin_pu = VMIN_PU, vmax_pu = VMAX_PU)
savefig(p_b_v, joinpath(figdir, "congestion_scatter_voltage__baseline.png"))

p_b_m = plot_congestion_scatter_margin(baseline.df_bus; scenario_id = "baseline")
savefig(p_b_m, joinpath(figdir, "congestion_scatter_margin__baseline.png"))

p_b_u = plot_congestion_scatter_vuf(baseline.df_bus; scenario_id = "baseline", vuf_limit = VUF_LIMIT)
savefig(p_b_u, joinpath(figdir, "congestion_scatter_vuf__baseline.png"))

# Find critical location from baseline: weakest bus-phase by voltage magnitude
function find_weakest_bus_and_phase_from_df(df::DataFrame)
    (vmin_pu, bus, phase) = extreme_voltage(df)[1]
    if vmin_pu isa Missing
        return (missing, missing, missing)
    end
    # phase is "a"/"b"/"c" from extreme_voltage
    ph = phase == "a" ? 1 : phase == "b" ? 2 : 3
    return (bus, ph, vmin_pu)
end

weak_bus, weak_phase, weak_vmin = find_weakest_bus_and_phase_from_df(baseline.df_bus)

# choose a mid-feeder bus for distributed case
function pick_mid_bus(dist::Dict{String,Float64}; must_exist = nothing)
    maxd = maximum(values(dist))
    target = 0.5 * maxd
    best_bus = nothing
    best_err = Inf
    for (bus, d) in dist
        if must_exist !== nothing && !(bus in must_exist)
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

mid_bus = pick_mid_bus(dist; must_exist = Set(baseline.df_bus.bus))

println("Baseline weakest: bus=$(weak_bus), phase=$(PH_LABEL[weak_phase]), vmin=$(weak_vmin) pu")
println("Mid-feeder bus: $(mid_bus)")

# -----------------------
# Scenario set (stress scenarios)
# - severity sweep at weakest bus
# - two EV clustered at weakest
# - two EV distributed (source + weakest)
# -----------------------
scenario_runs = Any[]
summary_rows = NamedTuple[]

# 1) Severity sweep scenarios
severity_curve_rows = NamedTuple[]
for ev_kw in ev_kw_levels
    scenario_id = "sev_$(replace(string(ev_kw), "." => "p"))kw"
    ev_desc = ev_kw <= 1e-9 ? "no_ev" : "1xEV@$(weak_bus) ph$(PH_LABEL[weak_phase]) size=$(ev_kw)kW pf=$(EV_PF)"
    math_case = deepcopy(math)

    if ev_kw > 1e-9
        p_mw = kw_to_mw(Float64(ev_kw))
        q_mvar = q_from_pf(p_mw, EV_PF)
        add_single_phase_ev!(math_case, string(weak_bus), Int(weak_phase); p_mw = p_mw, q_mvar = q_mvar, name = "ev")
    end

    out = run_case(scenario_id, "severity_sweep", ev_desc, math_case, lines_df, dist)

    # Save per-scenario bus/gen tables
    CSV.write(joinpath(tabledir, "scenario_bus_metrics__$(scenario_id).csv"), out.df_bus)
    CSV.write(joinpath(tabledir, "scenario_gen_q_metrics__$(scenario_id).csv"), out.df_gen)

    # Save congestion plots
    savefig(plot_congestion_scatter_voltage(out.df_bus; scenario_id = scenario_id, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU),
            joinpath(figdir, "congestion_scatter_voltage__$(scenario_id).png"))
    savefig(plot_congestion_scatter_margin(out.df_bus; scenario_id = scenario_id),
            joinpath(figdir, "congestion_scatter_margin__$(scenario_id).png"))
    savefig(plot_congestion_scatter_vuf(out.df_bus; scenario_id = scenario_id, vuf_limit = VUF_LIMIT),
            joinpath(figdir, "congestion_scatter_vuf__$(scenario_id).png"))

    push!(summary_rows, out.summary)

    # Add sweep curve row (for easy threshold detection)
    push!(severity_curve_rows, (
        ev_kw = ev_kw,
        scenario_id = scenario_id,
        status = out.status,
        vmin_pu = out.summary.vmin_pu,
        n_phase_viol = out.summary.n_phase_viol,
        reach_km = out.summary.reach_km,
        vuf_max = out.summary.vuf_max,
        n_vuf_viol = out.summary.n_vuf_viol,
        vuf_reach_km = out.summary.vuf_reach_km,
        gen_q_at_qmin = out.summary.gen_q_at_qmin,
        gen_q_at_qmax = out.summary.gen_q_at_qmax
    ))
end

# 2) Two EV clustered at weakest (same phase)
begin
    scenario_id = "twoev_clustered_samephase"
    ev_desc = "2xEV@$(weak_bus) ph$(PH_LABEL[weak_phase]) size=$(EV2_KW)kW each pf=$(EV_PF)"
    math_case = deepcopy(math)

    p_mw = kw_to_mw(Float64(EV2_KW))
    q_mvar = q_from_pf(p_mw, EV_PF)
    add_single_phase_ev!(math_case, string(weak_bus), Int(weak_phase); p_mw = p_mw, q_mvar = q_mvar, name = "ev1")
    add_single_phase_ev!(math_case, string(weak_bus), Int(weak_phase); p_mw = p_mw, q_mvar = q_mvar, name = "ev2")

    out = run_case(scenario_id, "interaction_two_ev", ev_desc, math_case, lines_df, dist)

    CSV.write(joinpath(tabledir, "scenario_bus_metrics__$(scenario_id).csv"), out.df_bus)
    CSV.write(joinpath(tabledir, "scenario_gen_q_metrics__$(scenario_id).csv"), out.df_gen)

    savefig(plot_congestion_scatter_voltage(out.df_bus; scenario_id = scenario_id, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU),
            joinpath(figdir, "congestion_scatter_voltage__$(scenario_id).png"))
    savefig(plot_congestion_scatter_margin(out.df_bus; scenario_id = scenario_id),
            joinpath(figdir, "congestion_scatter_margin__$(scenario_id).png"))
    savefig(plot_congestion_scatter_vuf(out.df_bus; scenario_id = scenario_id, vuf_limit = VUF_LIMIT),
            joinpath(figdir, "congestion_scatter_vuf__$(scenario_id).png"))

    push!(summary_rows, out.summary)
end

# 3) Two EV distributed: source + weakest (same phase)
begin
    scenario_id = "twoev_distributed_source_plus_weakest"
    ev_desc = "EV@sourcebus + EV@$(weak_bus) both ph$(PH_LABEL[weak_phase]) size=$(EV2_KW)kW each pf=$(EV_PF)"
    math_case = deepcopy(math)

    p_mw = kw_to_mw(Float64(EV2_KW))
    q_mvar = q_from_pf(p_mw, EV_PF)
    add_single_phase_ev!(math_case, "sourcebus", Int(weak_phase); p_mw = p_mw, q_mvar = q_mvar, name = "ev1")
    add_single_phase_ev!(math_case, string(weak_bus), Int(weak_phase); p_mw = p_mw, q_mvar = q_mvar, name = "ev2")

    out = run_case(scenario_id, "interaction_two_ev", ev_desc, math_case, lines_df, dist)

    CSV.write(joinpath(tabledir, "scenario_bus_metrics__$(scenario_id).csv"), out.df_bus)
    CSV.write(joinpath(tabledir, "scenario_gen_q_metrics__$(scenario_id).csv"), out.df_gen)

    savefig(plot_congestion_scatter_voltage(out.df_bus; scenario_id = scenario_id, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU),
            joinpath(figdir, "congestion_scatter_voltage__$(scenario_id).png"))
    savefig(plot_congestion_scatter_margin(out.df_bus; scenario_id = scenario_id),
            joinpath(figdir, "congestion_scatter_margin__$(scenario_id).png"))
    savefig(plot_congestion_scatter_vuf(out.df_bus; scenario_id = scenario_id, vuf_limit = VUF_LIMIT),
            joinpath(figdir, "congestion_scatter_vuf__$(scenario_id).png"))

    push!(summary_rows, out.summary)
end

# -----------------------
# Write master summary tables
# -----------------------
summary_df = DataFrame(summary_rows)
CSV.write(joinpath(tabledir, "scenario_summary.csv"), summary_df)

severity_curve_df = DataFrame(severity_curve_rows)
CSV.write(joinpath(tabledir, "severity_sweep_curve.csv"), severity_curve_df)

# -----------------------
# Congestion-specific sweep plots (threshold + non-linear effects)
# -----------------------
# We plot:
# - overall min voltage vs EV size
# - max VUF vs EV size
# - reach (closest violator distance) vs EV size

# filter out baseline row (we only have sweep rows here)
xs = Float64.(severity_curve_df.ev_kw)

minV = [r isa Missing ? NaN : Float64(r) for r in severity_curve_df.vmin_pu]
vufmax = [r isa Missing ? NaN : Float64(r) for r in severity_curve_df.vuf_max]
reach = [r isa Missing ? NaN : Float64(r) for r in severity_curve_df.reach_km]
vufreach = [r isa Missing ? NaN : Float64(r) for r in severity_curve_df.vuf_reach_km]

p1 = plot(xs, minV, marker = :circle, legend = false)
xlabel!("EV size (kW)")
ylabel!("Minimum voltage on feeder (pu)")
title!("Severity sweep: min voltage vs EV size")
hline!([VMIN_PU], linestyle = :dash, label = false)
savefig(p1, joinpath(figdir, "severity_minV_vs_kw.png"))

p2 = plot(xs, vufmax, marker = :circle, legend = false)
xlabel!("EV size (kW)")
ylabel!("Max VUF (-)")
title!("Severity sweep: max VUF vs EV size")
hline!([VUF_LIMIT], linestyle = :dash, label = false)
savefig(p2, joinpath(figdir, "severity_vufmax_vs_kw.png"))

p3 = plot(xs, reach, marker = :circle, legend = false)
xlabel!("EV size (kW)")
ylabel!("Reach = closest voltage violator distance (km)")
title!("Severity sweep: voltage violation region reaches toward source")
savefig(p3, joinpath(figdir, "severity_reach_vs_kw.png"))

println("\nDone.")
println("Tables saved to: ", tabledir)
println("Figures saved to: ", figdir)
println("\nSTATCOM script can consume:")
println("  - scenario_summary.csv (high-level selection of stress case + target region)")
println("  - scenario_bus_metrics__<scenario_id>.csv (candidate buses ranked by bus_risk + vuf)")
println("  - scenario_gen_q_metrics__<scenario_id>.csv (reactive limits binding evidence)")
println("  - severity_sweep_curve.csv (thresholds and non-linear effects)")
