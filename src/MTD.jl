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
include("gof.jl")

end # module
