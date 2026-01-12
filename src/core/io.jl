# src/core/io.jl
#
# Purpose
# -------
# This file handles the "engineering data" stage:
# - reading a feeder from OpenDSS (Master.dss)
# - cleaning up topology (radialising if loops exist)
# - applying consistent settings (like base power)
# - applying voltage bounds at the engineering level (optional)
#
# Notes (three-wire + Kron reduced workflow)
# -----------------------------------------
# The project is using three-wire networks (phases A, B, C only).
# The neutral is removed later during transform_data_model by setting kron_reduce=true.
# That means voltage bounds will mainly be for phase-to-neutral magnitudes in per-unit.

import PowerModelsDistribution as PMD

"""
load_eng_model(file; sbase_default_pu=1.0, apply_radial_fix=true)

Reads an OpenDSS feeder into an "engineering model" dictionary (eng).

- file: path to Master.dss
- sbase_default_pu: sets eng["settings"]["sbase_default"] to a consistent value
- apply_radial_fix: applies transform_loops! so OPF assumptions are satisfied

Returns: eng::Dict
"""
function load_eng_model(file::AbstractString;
                        sbase_default_pu::Float64=1.0,
                        apply_radial_fix::Bool=true)

    # parse_file reads the OpenDSS file into PMD's engineering data structure
    if apply_radial_fix
        eng = PMD.parse_file(file, transformations=[PMD.transform_loops!])
    else
        eng = PMD.parse_file(file)
    end

    # sbase_default is the power base used when converting values to per-unit.
    # Keeping this consistent helps avoid confusion across notebooks.
    eng["settings"]["sbase_default"] = sbase_default_pu

    return eng
end


"""
print_eng_summary(eng)

Quick sanity checks. This does not change anything.
"""
function print_eng_summary(eng)
    nb = length(get(eng, "bus", Dict()))
    nl = length(get(eng, "line", Dict()))
    nd = length(get(eng, "load", Dict()))
    println("Engineering model summary:")
    println("  buses:  ", nb)
    println("  lines:  ", nl)
    println("  loads:  ", nd)
    return nothing
end


"""
apply_voltage_bounds_eng!(eng; phase_lb_pu=0.9, phase_ub_pu=1.1)

Adds voltage bounds at the engineering level (per-unit).

For three-wire studies, only phase bounds matter.
Some PMD helpers also accept a neutral bound. For three-wire networks,
the neutral is not kept, so neutral bounds are not used in practice.
"""
function apply_voltage_bounds_eng!(eng;
                                  phase_lb_pu::Float64=0.9,
                                  phase_ub_pu::Float64=1.1)

    # There are different PMD helper functions depending on PMD version.
    # The most common one is add_bus_absolute_vbounds!.
    # If the function signature requires neutral_ub_pu, a safe value is provided.
    try
        PMD.add_bus_absolute_vbounds!(eng,
                                     phase_lb_pu=phase_lb_pu,
                                     phase_ub_pu=phase_ub_pu)
    catch
        # fallback for versions that expect neutral_ub_pu as well
        PMD.add_bus_absolute_vbounds!(eng,
                                     phase_lb_pu=phase_lb_pu,
                                     phase_ub_pu=phase_ub_pu,
                                     neutral_ub_pu=0.0)
    end

    return eng
end
