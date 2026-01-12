# src/core/ev.jl
#
# Purpose
# -------
# This file contains helper functions for adding EV loads to the PMD math model.
#
# The EV is modelled as a constant PQ load connected to a single phase.
# This is sufficient for snapshot voltage studies and sensitivity analysis.
#
# Assumptions
# -----------
# - three-wire network (A, B, C)
# - Kron reduced (neutral removed)
# - snapshot OPF (not time-series)
#
# Notes
# -----
# In PMD, loads are stored in math["load"] as dictionaries.
# Each load has:
# - a bus (string id)
# - a list of connections (phases)
# - pd and qd values in per-unit
#
# This file does not solve OPF. It only modifies the math model.

"""
scenario_copy(math)

Creates a deep copy of the math model.

This is important because each EV scenario should start
from the same baseline and not modify it in place.
"""
function scenario_copy(math)
    return deepcopy(math)
end


"""
kw_to_pu(P_kw, sbase)

Converts active power from kW to per-unit.

sbase is the system base used by PMD (usually in MVA-like units).
The convention used in this project is:
    per-unit = kW / (sbase * 1000)
"""
function kw_to_pu(P_kw::Float64, sbase::Float64)
    return P_kw / (sbase * 1000.0)
end


"""
add_single_ev!(
    math,
    bus_id;
    phase=1,
    P_kw=7.0,
    Q_kvar=0.0
)

Adds a single EV load to the math model.

Arguments
---------
- math: PMD math model dictionary
- bus_id: string id of the bus where the EV is connected
- phase: phase index (1=A, 2=B, 3=C)
- P_kw: active power of the EV charger
- Q_kvar: reactive power (usually 0 for unity power factor)

Notes
-----
- The EV is modelled as a constant power load.
- Only one phase is used.
- No validation of charger realism is done here.
"""
function add_single_ev!(
    math,
    bus_id::String;
    phase::Int=1,
    P_kw::Float64=7.0,
    Q_kvar::Float64=0.0
)

    # Read system base from math model
    sbase = get(math["settings"], "sbase", 1.0)

    # Convert kW / kvar to per-unit
    P_pu = kw_to_pu(P_kw, sbase)
    Q_pu = kw_to_pu(Q_kvar, sbase)

    # Create a new load id
    existing_ids = parse.(Int, collect(keys(math["load"])))
    new_id = string(isempty(existing_ids) ? 1 : maximum(existing_ids) + 1)

    # Add EV load entry
    math["load"][new_id] = Dict(
        "bus" => bus_id,
        "connections" => [phase],        # single-phase EV
        "pd" => [P_pu],
        "qd" => [Q_pu],
        "status" => 1,
        "model" => "constant_power"
    )

    return new_id
end


"""
add_multiple_evs!(
    math,
    placements
)

Adds multiple EVs to the math model.

placements is a vector of NamedTuples, for example:

placements = [
    (bus_id="12", phase=1, P_kw=7.0),
    (bus_id="12", phase=1, P_kw=7.0),
    (bus_id="sourcebus", phase=2, P_kw=7.0)
]

This helper is useful for clustered vs distributed EV studies.
"""
function add_multiple_evs!(
    math,
    placements::Vector{NamedTuple}
)

    for ev in placements
        add_single_ev!(
            math,
            ev.bus_id;
            phase=get(ev, :phase, 1),
            P_kw=get(ev, :P_kw, 7.0),
            Q_kvar=get(ev, :Q_kvar, 0.0)
        )
    end

    return math
end


"""
print_ev_summary(math)

Prints a simple summary of EV-type loads added to the math model.

This is a debugging helper to check:
- how many loads exist
- where EVs were added
"""
function print_ev_summary(math)
    println("EV / load summary:")
    for (id, load) in math["load"]
        println(
            "  load ", id,
            " at bus ", load["bus"],
            ", phase(s) ", load["connections"],
            ", pd=", load["pd"]
        )
    end
    return nothing
end
