module MTD

## distributed packages
using Distributions
using Combinatorics
using StatsBase
using Dates

## personal packages
using BayesInference
using SparseProbVec

include("general.jl")
include("mtd_inference.jl")
include("mmtd_inference.jl")
include("gof.jl")

end # module
