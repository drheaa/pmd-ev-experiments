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

const VBASE_LN = 230.0
const VMIN_PU = 0.94
const VMAX_PU = 1.10
const VUF_LIMIT = 0.02
const EV_PF_DEFAULT = 0.95

const ENFORCE_V_LIMITS = true

# Scenario selection
const TARGET_SCENARIO_ID = ""   # if blank: auto-select from severity sweep
const MIN_STRESS_MARGIN_PU = 0.002  # pick first sweep point where vmin <= VMIN + margin

# STATCOM size
const STATCOM_QCAP_KVAR = 400.0
const STATCOM_QCAP_MVAR = STATCOM_QCAP_KVAR / 1000.0

# Candidate selection
const TOPK_CANDIDATES = 5
const ONLY_IN_CONGESTION_REGION = true
const FORCE_STATCOM_BUS = ""    # optional override, e.g. "899"

# Paths
cong_root = joinpath(@__DIR__, "..", "results", "congestion_identification")
cong_tables = joinpath(cong_root, "tables")

out_root = joinpath(@__DIR__, "..", "results", "statcom_mitigation")
out_figdir = joinpath(out_root, "figures")
out_tabledir = joinpath(out_root, "tables")
mkpath(out_figdir)
mkpath(out_tabledir)

# -----------------------
# Helpers: parse + transform
# -----------------------
function parse_and_transform(file::String)
    eng = parse_file(file, transformations = [transform_loops!, reduce_lines!])
    eng["settings"]["sbase_default"] = 1
    eng["voltage_source"]["source"]["rs"] .= 0
    eng["voltage_source"]["source"]["xs"] .= 0
    math = transform_data_model(eng, multinetwork = false, kron_reduce = true, phase_project = true)
    return eng, math
end

# -----------------------
# Helpers: distances
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
# Voltages + VUF
# -----------------------
function solved_bus_voltages(result_opf, math; vbase_ln = VBASE_LN)
    sol_bus = result_opf["solution"]["bus"]
    buses = Dict{String, Dict{String,Any}}()

    for (bus_id, bus_data) in math["bus"]
        name = bus_data["name"]
        haskey(sol_bus, bus_id) || continue
        sb = sol_bus[bus_id]
        d = Dict{String,Any}()

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

        if haskey(sb, "vr") && haskey(sb, "vi") && length(sb["vr"]) >= 3 && length(sb["vi"]) >= 3
            d["vra"] = [sb["vr"][1]]; d["via"] = [sb["vi"][1]]
            d["vrb"] = [sb["vr"][2]]; d["vib"] = [sb["vi"][2]]
            d["vrc"] = [sb["vr"][3]]; d["vic"] = [sb["vi"][3]]
        end

        !isempty(d) && (buses[name] = d)
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
# Bus metrics (same fields)
# -----------------------
function compute_bus_metrics(buses, dist; t = 1, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU, vuf_limit = VUF_LIMIT)
    rows = NamedTuple[]
    for (bus, d) in buses
        distance_km = haskey(dist, bus) ? dist[bus] : (haskey(d, "distance") ? d["distance"] : missing)

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

        risks = Float64[]
        for r in (ra, rb, rc)
            if !(r isa Missing); push!(risks, Float64(r)); end
        end
        bus_risk = isempty(risks) ? missing : minimum(risks)

        vuf = haskey(d, "vuf") ? d["vuf"][t] : missing
        vuf_viol = (vuf isa Missing) ? missing : (isfinite(vuf) && vuf > vuf_limit)

        push!(rows, (
            bus = bus, distance_km = distance_km,
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

# -----------------------
# Congestion plots (same style)
# -----------------------
function plot_congestion_scatter_voltage(df::DataFrame; scenario_id::String, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU)
    p = plot(legend = :topright)
    xlabel!("Distance from source (km)")
    ylabel!("Voltage magnitude (pu)")
    title!("Congestion view (voltage): $(scenario_id)")

    for (ph, colv, lowc, highc) in (
        ("a", :va_pu, :viol_a_low, :viol_a_high),
        ("b", :vb_pu, :viol_b_low, :viol_b_high),
        ("c", :vc_pu, :viol_c_low, :viol_c_high)
    )
        xs = Float64[]; ys = Float64[]
        xsv = Float64[]; ysv = Float64[]
        for i in 1:nrow(df)
            v = df[i, colv]; d = df[i, :distance_km]
            if v isa Missing || d isa Missing; continue; end
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
    xs = Float64[]; ys = Float64[]
    for i in 1:nrow(df)
        d = df.distance_km[i]; r = df.bus_risk[i]
        if d isa Missing || r isa Missing; continue; end
        push!(xs, Float64(d)); push!(ys, Float64(r))
    end
    p = scatter(xs, ys, markersize = 3, legend = false)
    xlabel!("Distance from source (km)")
    ylabel!("Bus risk = min(V - VMIN, VMAX - V) (pu)")
    title!("Congestion view (margin): $(scenario_id)")
    hline!([0.0], linestyle = :dash, label = false)
    return p
end

function plot_congestion_scatter_vuf(df::DataFrame; scenario_id::String, vuf_limit = VUF_LIMIT)
    xs = Float64[]; ys = Float64[]
    xsv = Float64[]; ysv = Float64[]
    for i in 1:nrow(df)
        d = df.distance_km[i]; v = df.vuf[i]
        if d isa Missing || v isa Missing || !isfinite(v); continue; end
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
# Enforce voltage limits inside OPF
# -----------------------
function enforce_bus_voltage_limits!(math; vmin_pu = VMIN_PU, vmax_pu = VMAX_PU)
    haskey(math, "bus") || return
    for (i, bus) in math["bus"]
        n = haskey(bus, "vmin") ? length(bus["vmin"]) : 3
        bus["vmin"] = vmin_pu .* ones(n)
        n2 = haskey(bus, "vmax") ? length(bus["vmax"]) : 3
        bus["vmax"] = vmax_pu .* ones(n2)
    end
end

# -----------------------
# EV injection (reconstruct scenarios)
# -----------------------
kw_to_mw(kW::Float64) = kW / 1000.0
function q_from_pf(p_mw::Float64, pf::Float64)
    pf >= 0.999 && return 0.0
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
    pd = zeros(3); qd = zeros(3)
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
    if haskey(new_load, "connections"); new_load["connections"] = [1,2,3]; end
    math["load"][new_id] = new_load
    return new_id
end

function weakest_bus_phase_from_baseline(df_bus::DataFrame)
    vmin = Inf; best_bus = missing; best_phase = missing
    for i in 1:nrow(df_bus)
        b = df_bus.bus[i]
        for (ph, col) in (("a", :va_pu), ("b", :vb_pu), ("c", :vc_pu))
            v = df_bus[i, col]
            v isa Missing && continue
            vv = Float64(v)
            if vv < vmin
                vmin = vv; best_bus = b; best_phase = ph
            end
        end
    end
    best_phase === missing && return (missing, missing)
    phase_int = best_phase == "a" ? 1 : best_phase == "b" ? 2 : 3
    return (best_bus, phase_int)
end

function parse_first_float_after(s::String, key::String)
    idx = findfirst(key, s); idx === nothing && return missing
    tail = s[(idx.start + length(key)) : end]
    m = match(r"^\s*([0-9]+(?:\.[0-9]+)?)", tail)
    m === nothing && return missing
    return parse(Float64, m.captures[1])
end

parse_pf_from_desc(ev_desc::String) = (x = parse_first_float_after(ev_desc, "pf="); x isa Missing ? EV_PF_DEFAULT : x)
parse_size_kw_from_desc(ev_desc::String) = parse_first_float_after(ev_desc, "size=")

function parse_sweep_kw_from_id(scenario_id::String)
    startswith(scenario_id, "sev_") || return missing
    body = replace(replace(scenario_id, "sev_" => ""), "kw" => "")
    body = replace(body, "p" => ".")
    x = tryparse(Float64, body)
    return x === nothing ? missing : x
end

function build_math_for_scenario(math0, scenario_id::String, ev_desc::String, weak_bus::String, weak_phase::Int)
    math_case = deepcopy(math0)
    pf = parse_pf_from_desc(ev_desc)
    size_kw = parse_size_kw_from_desc(ev_desc)

    scenario_id == "baseline" && return math_case

    if startswith(scenario_id, "sev_")
        kw = parse_sweep_kw_from_id(scenario_id)
        kw === missing && (kw = size_kw)
        kw === missing && error("Could not parse sweep kW for $(scenario_id).")
        if kw > 1e-9
            p_mw = kw_to_mw(kw)
            q_mvar = q_from_pf(p_mw, pf)
            add_single_phase_ev!(math_case, weak_bus, weak_phase; p_mw = p_mw, q_mvar = q_mvar, name = "ev")
        end
        return math_case
    end

    if scenario_id == "twoev_clustered_samephase"
        kw_each = size_kw
        kw_each === missing && error("Missing EV size in ev_desc for $(scenario_id).")
        p_mw = kw_to_mw(kw_each)
        q_mvar = q_from_pf(p_mw, pf)
        add_single_phase_ev!(math_case, weak_bus, weak_phase; p_mw = p_mw, q_mvar = q_mvar, name = "ev1")
        add_single_phase_ev!(math_case, weak_bus, weak_phase; p_mw = p_mw, q_mvar = q_mvar, name = "ev2")
        return math_case
    end

    if scenario_id == "twoev_distributed_source_plus_weakest"
        kw_each = size_kw
        kw_each === missing && error("Missing EV size in ev_desc for $(scenario_id).")
        p_mw = kw_to_mw(kw_each)
        q_mvar = q_from_pf(p_mw, pf)
        add_single_phase_ev!(math_case, "sourcebus", weak_phase; p_mw = p_mw, q_mvar = q_mvar, name = "ev1")
        add_single_phase_ev!(math_case, weak_bus, weak_phase; p_mw = p_mw, q_mvar = q_mvar, name = "ev2")
        return math_case
    end

    error("Scenario not recognized by this STATCOM script: $(scenario_id)")
end

# -----------------------
# STATCOM insertion (phase-specific Q bounds)
# -----------------------
function ensure_vector_len3(x, fillval::Float64)
    if x isa AbstractVector
        n = length(x)
        if n == 3
            return Float64.(x)
        else
            return fillval .* ones(n)  # preserve length if template isn’t 3
        end
    end
    return fillval .* ones(3)
end

function add_statcom_gen!(math, bus::String; qcap_mvar::Float64, name::String = "statcom")
    haskey(math, "gen") || error("math model has no 'gen' table (needed to template STATCOM).")

    tpl_id, tpl = first(collect(math["gen"]))
    new_gen = deepcopy(tpl)
    new_id = string(next_numeric_id(math["gen"]))

    new_gen["bus"] = bus
    new_gen["status"] = 1
    new_gen["name"] = "$(name)_$(bus)"

    # Zero P
    if haskey(new_gen, "pg");   new_gen["pg"]   = ensure_vector_len3(new_gen["pg"], 0.0); end
    if haskey(new_gen, "pmin"); new_gen["pmin"] = ensure_vector_len3(new_gen["pmin"], 0.0); end
    if haskey(new_gen, "pmax"); new_gen["pmax"] = ensure_vector_len3(new_gen["pmax"], 0.0); end

    # Q bounds per phase (this is the key upgrade)
    qmin_tpl = get(new_gen, "qmin", [-qcap_mvar, -qcap_mvar, -qcap_mvar])
    qmax_tpl = get(new_gen, "qmax", [ qcap_mvar,  qcap_mvar,  qcap_mvar])

    new_gen["qmin"] = ensure_vector_len3(qmin_tpl, -qcap_mvar)
    new_gen["qmax"] = ensure_vector_len3(qmax_tpl,  qcap_mvar)

    # optional initial qg
    if haskey(new_gen, "qg"); new_gen["qg"] = ensure_vector_len3(new_gen["qg"], 0.0); end

    math["gen"][new_id] = new_gen

    # Copy a gencost row so OPF doesn’t complain in some setups
    if haskey(math, "gencost")
        if haskey(math["gencost"], tpl_id)
            math["gencost"][new_id] = deepcopy(math["gencost"][tpl_id])
        else
            cid, ctpl = first(collect(math["gencost"]))
            math["gencost"][new_id] = deepcopy(ctpl)
        end
    end

    return new_id
end

function extract_statcom_q(result, math, gen_id::String; eps = 1e-6)
    # returns (qg_vec, qmin_vec, qmax_vec, at_qmin_vec, at_qmax_vec)
    sol = result["solution"]["gen"]
    mg = math["gen"][gen_id]
    qg = get(sol[gen_id], "qg", missing)
    qmin = get(mg, "qmin", missing)
    qmax = get(mg, "qmax", missing)

    qg_v = qg isa AbstractVector ? Float64.(qg) : [qg, qg, qg]
    qmin_v = qmin isa AbstractVector ? Float64.(qmin) : [qmin, qmin, qmin]
    qmax_v = qmax isa AbstractVector ? Float64.(qmax) : [qmax, qmax, qmax]

    atmin = [qg_v[i] <= qmin_v[i] + eps for i in 1:length(qg_v)]
    atmax = [qg_v[i] >= qmax_v[i] - eps for i in 1:length(qg_v)]
    return qg_v, qmin_v, qmax_v, atmin, atmax
end

# -----------------------
# OPF runner
# -----------------------
function run_opf(math_case, dist; enforce_limits::Bool)
    enforce_limits && enforce_bus_voltage_limits!(math_case; vmin_pu = VMIN_PU, vmax_pu = VMAX_PU)
    result = solve_mc_opf(math_case, IVRUPowerModel, ipopt_solver)
    status = string(result["termination_status"])

    buses = solved_bus_voltages(result, math_case; vbase_ln = VBASE_LN)
    for (bus, d) in dist
        haskey(buses, bus) && (buses[bus]["distance"] = d)
    end
    add_vuf!(buses; t = 1)

    df_bus = compute_bus_metrics(buses, dist; t = 1, vmin_pu = VMIN_PU, vmax_pu = VMAX_PU, vuf_limit = VUF_LIMIT)
    return status, result, buses, df_bus
end

# -----------------------
# Scenario selection (auto)
# -----------------------
function pick_scenario_id()
    if TARGET_SCENARIO_ID != ""
        return TARGET_SCENARIO_ID
    end

    sweep_path = joinpath(cong_tables, "severity_sweep_curve.csv")
    isfile(sweep_path) || error("Missing: $(sweep_path)")

    sweep = CSV.read(sweep_path, DataFrame)
    # remove 0kW rows and any missing vmin
    sweep = sweep[(sweep.ev_kw .> 1e-9) .& .!ismissing.(sweep.vmin_pu), :]
    nrow(sweep) == 0 && error("severity_sweep_curve.csv has no EV points > 0.")

    sort!(sweep, :ev_kw)

    # Pick first point where vmin <= VMIN + margin
    for i in 1:nrow(sweep)
        if Float64(sweep.vmin_pu[i]) <= VMIN_PU + MIN_STRESS_MARGIN_PU
            return String(sweep.scenario_id[i])
        end
    end

    # If none cross, pick the most severe EV point (max kW)
    return String(sweep.scenario_id[end])
end

# -----------------------
# Candidate selection from congestion outputs
# -----------------------
function select_candidates(scenario_id::String)
    if FORCE_STATCOM_BUS != ""
        return [FORCE_STATCOM_BUS]
    end

    summ_path = joinpath(cong_tables, "scenario_summary.csv")
    isfile(summ_path) || error("Missing: $(summ_path)")
    summ = CSV.read(summ_path, DataFrame)

    row = summ[summ.scenario_id .== scenario_id, :]
    nrow(row) == 1 || error("Scenario id not found or not unique in scenario_summary.csv: $(scenario_id)")

    bus_path = joinpath(cong_tables, "scenario_bus_metrics__$(scenario_id).csv")
    isfile(bus_path) || error("Missing: $(bus_path)")
    df = CSV.read(bus_path, DataFrame)

    # Region filter using congestion-identification region (post-check region)
    if ONLY_IN_CONGESTION_REGION
        rs = row.region_start_km[1]
        re = row.region_end_km[1]
        if !(rs isa Missing) && !(re isa Missing) && (Float64(re) - Float64(rs) > 1e-9)
            df = df[(.!ismissing.(df.distance_km)) .& (df.distance_km .>= rs) .& (df.distance_km .<= re), :]
        end
    end

    # If region filter empties everything, fall back to far end 20%
    if nrow(df) == 0
        bus_path2 = joinpath(cong_tables, "scenario_bus_metrics__$(scenario_id).csv")
        df = CSV.read(bus_path2, DataFrame)
        dmax = maximum(skipmissing(df.distance_km))
        df = df[(.!ismissing.(df.distance_km)) .& (df.distance_km .>= 0.8*dmax), :]
    end

    df = df[.!ismissing.(df.bus_risk), :]
    sort!(df, :bus_risk)

    out = String[]
    for i in 1:min(TOPK_CANDIDATES, nrow(df))
        push!(out, String(df.bus[i]))
    end
    isempty(out) && error("No candidate buses found. Set FORCE_STATCOM_BUS or relax filters.")
    return out
end

# -----------------------
# MAIN
# -----------------------
scenario_id = pick_scenario_id()
println("Selected scenario_id = ", scenario_id)

# Read scenario_summary to get ev_desc
summ = CSV.read(joinpath(cong_tables, "scenario_summary.csv"), DataFrame)
row = summ[summ.scenario_id .== scenario_id, :]
nrow(row) == 1 || error("Scenario row not found: $(scenario_id)")
ev_desc = String(row.ev_desc[1])
println("ev_desc = ", ev_desc)

candidate_buses = select_candidates(scenario_id)
println("Candidate STATCOM buses: ", join(candidate_buses, ", "))

# Build base network
eng, math0 = parse_and_transform(file)
lines_df = make_lines_df_from_eng(eng)
dist = compute_bus_distances(lines_df; source_bus = "sourcebus")

# Baseline to recover weakest bus-phase (matches congestion logic)
status_bl, res_bl, buses_bl, df_bl = run_opf(deepcopy(math0), dist; enforce_limits = false)
weak_bus, weak_phase = weakest_bus_phase_from_baseline(df_bl)
weak_bus === missing && error("Could not detect weakest bus-phase from baseline.")
println("Recovered weakest from baseline: bus=$(weak_bus), phase=$(weak_phase)")

# Reconstruct scenario math
math_scn = build_math_for_scenario(math0, scenario_id, ev_desc, String(weak_bus), Int(weak_phase))

# BEFORE
status_before, res_before, buses_before, df_before = run_opf(deepcopy(math_scn), dist; enforce_limits = ENFORCE_V_LIMITS)
println("Before STATCOM status = ", status_before)

savefig(plot_congestion_scatter_voltage(df_before; scenario_id = "$(scenario_id)__before"),
        joinpath(out_figdir, "congestion_scatter_voltage__$(scenario_id)__before.png"))
savefig(plot_congestion_scatter_margin(df_before; scenario_id = "$(scenario_id)__before"),
        joinpath(out_figdir, "congestion_scatter_margin__$(scenario_id)__before.png"))
savefig(plot_congestion_scatter_vuf(df_before; scenario_id = "$(scenario_id)__before"),
        joinpath(out_figdir, "congestion_scatter_vuf__$(scenario_id)__before.png"))

# Helpers for KPIs
function min_voltage(df::DataFrame)
    vals = Float64[]
    for col in (:va_pu, :vb_pu, :vc_pu)
        for i in 1:nrow(df)
            v = df[i, col]
            v isa Missing && continue
            push!(vals, Float64(v))
        end
    end
    return isempty(vals) ? missing : minimum(vals)
end

function max_vuf(df::DataFrame)
    vals = Float64[]
    for i in 1:nrow(df)
        v = df.vuf[i]
        v isa Missing && continue
        isfinite(v) || continue
        push!(vals, Float64(v))
    end
    return isempty(vals) ? missing : maximum(vals)
end

function count_voltage_viol(df::DataFrame)
    nphase = 0
    nbus = 0
    for i in 1:nrow(df)
        anyv = false
        for (low, high) in ((df.viol_a_low[i], df.viol_a_high[i]),
                            (df.viol_b_low[i], df.viol_b_high[i]),
                            (df.viol_c_low[i], df.viol_c_high[i]))
            if low == true || high == true
                nphase += 1
                anyv = true
            end
        end
        anyv && (nbus += 1)
    end
    return nphase, nbus
end

function count_vuf_viol(df::DataFrame)
    n = 0
    for i in 1:nrow(df)
        df.vuf_viol[i] == true && (n += 1)
    end
    return n
end

# Run STATCOM placements
run_rows = NamedTuple[]

for bus in candidate_buses
    math_stat = deepcopy(math_scn)
    gen_id = add_statcom_gen!(math_stat, String(bus); qcap_mvar = STATCOM_QCAP_MVAR, name = "statcom")

    status_after, res_after, buses_after, df_after = run_opf(math_stat, dist; enforce_limits = ENFORCE_V_LIMITS)
    println("After STATCOM @$(bus) status = ", status_after)

    tag = "statcom_$(bus)"

    savefig(plot_congestion_scatter_voltage(df_after; scenario_id = "$(scenario_id)__$(tag)"),
            joinpath(out_figdir, "congestion_scatter_voltage__$(scenario_id)__$(tag).png"))
    savefig(plot_congestion_scatter_margin(df_after; scenario_id = "$(scenario_id)__$(tag)"),
            joinpath(out_figdir, "congestion_scatter_margin__$(scenario_id)__$(tag).png"))
    savefig(plot_congestion_scatter_vuf(df_after; scenario_id = "$(scenario_id)__$(tag)"),
            joinpath(out_figdir, "congestion_scatter_vuf__$(scenario_id)__$(tag).png"))

    p_cmp_v = plot(
        plot_congestion_scatter_voltage(df_before; scenario_id = "before"),
        plot_congestion_scatter_voltage(df_after; scenario_id = "after"),
        layout = (1,2),
        size = (1200, 450)
    )
    title!(p_cmp_v, "Voltage: $(scenario_id) | STATCOM @ $(bus)")
    savefig(p_cmp_v, joinpath(out_figdir, "compare_voltage_before_after__$(scenario_id)__$(tag).png"))

    p_cmp_u = plot(
        plot_congestion_scatter_vuf(df_before; scenario_id = "before"),
        plot_congestion_scatter_vuf(df_after; scenario_id = "after"),
        layout = (1,2),
        size = (1200, 450)
    )
    title!(p_cmp_u, "VUF: $(scenario_id) | STATCOM @ $(bus)")
    savefig(p_cmp_u, joinpath(out_figdir, "compare_vuf_before_after__$(scenario_id)__$(tag).png"))

    # STATCOM Q dispatch logging (per phase)
    qg_v, qmin_v, qmax_v, atmin_v, atmax_v = extract_statcom_q(res_after, math_stat, gen_id)

    qdf = DataFrame(
        scenario_id = [scenario_id],
        statcom_bus = [String(bus)],
        gen_id = [gen_id],
        qcap_kvar = [STATCOM_QCAP_KVAR],
        qg_a_mvar = [qg_v[1]], qg_b_mvar = [qg_v[2]], qg_c_mvar = [qg_v[3]],
        qg_total_mvar = [sum(qg_v)],
        qmin_a = [qmin_v[1]], qmin_b = [qmin_v[2]], qmin_c = [qmin_v[3]],
        qmax_a = [qmax_v[1]], qmax_b = [qmax_v[2]], qmax_c = [qmax_v[3]],
        at_qmin_a = [atmin_v[1]], at_qmin_b = [atmin_v[2]], at_qmin_c = [atmin_v[3]],
        at_qmax_a = [atmax_v[1]], at_qmax_b = [atmax_v[2]], at_qmax_c = [atmax_v[3]]
    )
    CSV.write(joinpath(out_tabledir, "statcom_q_dispatch__$(scenario_id)__$(tag).csv"), qdf)

    # KPIs
    minV_b = min_voltage(df_before)
    minV_a = min_voltage(df_after)
    maxU_b = max_vuf(df_before)
    maxU_a = max_vuf(df_after)

    nph_b, nbus_b = count_voltage_viol(df_before)
    nph_a, nbus_a = count_voltage_viol(df_after)

    nvuf_b = count_vuf_viol(df_before)
    nvuf_a = count_vuf_viol(df_after)

    push!(run_rows, (
        scenario_id = scenario_id,
        ev_desc = ev_desc,
        statcom_bus = String(bus),
        statcom_qcap_kvar = STATCOM_QCAP_KVAR,
        enforce_v_limits = ENFORCE_V_LIMITS,
        status_before = status_before,
        status_after = status_after,
        min_voltage_before_pu = minV_b,
        min_voltage_after_pu = minV_a,
        n_phase_viol_before = nph_b,
        n_phase_viol_after = nph_a,
        n_bus_viol_before = nbus_b,
        n_bus_viol_after = nbus_a,
        max_vuf_before = maxU_b,
        max_vuf_after = maxU_a,
        n_vuf_viol_before = nvuf_b,
        n_vuf_viol_after = nvuf_a,
        statcom_qg_total_mvar = sum(qg_v),
        statcom_qg_a_mvar = qg_v[1],
        statcom_qg_b_mvar = qg_v[2],
        statcom_qg_c_mvar = qg_v[3],
        statcom_at_qmax_anyphase = any(atmax_v),
        statcom_at_qmin_anyphase = any(atmin_v)
    ))
end

run_df = DataFrame(run_rows)
CSV.write(joinpath(out_tabledir, "statcom_run_summary.csv"), run_df)

println("\nDone.")
println("Figures: ", out_figdir)
println("Summary table: ", joinpath(out_tabledir, "statcom_run_summary.csv"))
