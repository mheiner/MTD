module MTD

## distributed packages
using Distributions
using Iterators
using Combinatorics
using StatsBase

## personal packages
using BayesInference

include("general.jl")
include("mmtd.jl")
# include("mmtd_M1.jl") # for sparseProbVec paper simulations
include("gof.jl")

end # module
