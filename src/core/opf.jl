# src/core/opf.jl
#
# Purpose
# -------
# This file handles the "math model + OPF" stage:
# - converting the engineering model into the mathematical optimisation model
# - solving snapshot OPF using a chosen formulation and solver
# - extracting voltages from the OPF solution
#
# Notes (three-wire + Kron reduced workflow)
# -----------------------------------------
# kron_reduce=true removes the neutral and reduces the network to a three-wire model.
# This is required by current project instructions.
#
# The OPF result provides voltages in rectangular form:
# - vr = real part
# - vi = imaginary part
# Voltage magnitude is computed as sqrt(vr^2 + vi^2).
# These magnitudes are in per-unit.

import PowerModelsDistribution as PMD
import Ipopt
import JuMP
import LinearAlgebra: norm


"""
make_math_model(eng; kron_reduce=true, phase_project=false, multinetwork=false)

Transforms the engineering model (eng) to the math model (math).

Important choices:
- kron_reduce=false: network is already kro-reduced
- phase_project=false: keep phases explicit (A/B/C), not projected into a reduced form
- multinetwork=false: snapshot study (not time-series / multi-period)
"""
function make_math_model(eng;
                         kron_reduce::Bool=false,
                         phase_project::Bool=false,
                         multinetwork::Bool=false)

    math = PMD.transform_data_model(eng;
                                   multinetwork=multinetwork,
                                   kron_reduce=kron_reduce,
                                   phase_project=phase_project)

    return math
end


"""
default_ipopt_solver()

Creates an Ipopt solver with quiet-ish output.
"""
function default_ipopt_solver()
    return JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "sb" => "yes"
    )
end


"""
solve_snapshot_opf(math; formulation=IVRENPowerModel, optimizer=nothing)

Runs one snapshot OPF and returns the result dictionary.

The formulation is chosen outside this file if needed.
If optimizer is not provided, a default Ipopt configuration is used.
"""
function solve_snapshot_opf(math;
                           formulation=IVRENPowerModel,
                           optimizer=nothing)

    opt = optimizer === nothing ? default_ipopt_solver() : optimizer

    result = PMD.solve_mc_opf(
        math,
        formulation,
        opt
    )

    return result
end


"""
busid_to_name_map(math)

Maps internal bus ids (strings like "1", "2", ...) to bus names (like "sourcebus", "12", ...)

This helps connect OPF outputs to topology / plotting which often uses bus names.
"""
function busid_to_name_map(math)
    id2name = Dict{String,String}()
    for (id, b) in math["bus"]
        # PMD stores a "name" field most of the time
        # The split removes phase suffixes if they exist
        id2name[string(id)] = split(get(b, "name", string(id)), ".")[1]
    end
    return id2name
end


"""
extract_vm_by_busid(result)

Returns a Dict bus_id -> voltage magnitudes vector.

Example:
vm["12"] = [Va, Vb, Vc]   (three-wire)
"""
function extract_vm_by_busid(result)
    vm = Dict{String, Vector{Float64}}()

    sol_bus = result["solution"]["bus"]
    for (id, b) in sol_bus
        vr = b["vr"]
        vi = b["vi"]

        # magnitude per conductor/phase
        vm[string(id)] = sqrt.(vr.^2 .+ vi.^2)
    end

    return vm
end


"""
min_phase_voltage(vm_byid)

Computes the minimum voltage magnitude across all buses and phases (A/B/C only).
"""
function min_phase_voltage(vm_byid::Dict{String,Vector{Float64}})
    mins = Float64[]
    for (id, vm) in vm_byid
        # three-wire: use first 3 entries
        nph = min(3, length(vm))
        push!(mins, minimum(vm[1:nph]))
    end
    return minimum(mins)
end


"""
count_voltage_violations(vm_byid; vmin=0.9, vmax=1.1)

Counts how many buses have at least one phase violating voltage bounds.
This is a simple "bus-level" count, not a phase-level count.
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
