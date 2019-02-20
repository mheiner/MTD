module MTD

## distributed packages
using Combinatorics
using Dates
using Distributions
using StatsBase

## personal packages
using SparseProbVec

include("general.jl")
include("mtd_inference.jl")
include("mtdg_inference.jl")
include("mmtd_inference.jl")
include("gof.jl")

end # module
