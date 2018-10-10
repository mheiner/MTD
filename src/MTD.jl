module MTD

## distributed packages
using Distributions
using Combinatorics
using StatsBase

## personal packages
using BayesInference
using SparseProbVec

include("general.jl")
include("mmtd.jl")
# include("mmtd_M1.jl") # for sparseProbVec paper simulations
include("gof.jl")

end # module
