import Pkg
# Pkg.add("JuMP")
# Pkg.add("Ipopt")
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using PowerModelsDistribution
using Ipopt
using JuMP
using LinearAlgebra

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
