using BayesInference
using MTD

const R = 3
const M = 2
const K = 2

λ_indx = build_λ_indx(R, M)
println(λ_indx[1])
println(λ_indx[1][1])
println(λ_indx[2])
println(λ_indx[2][1])
println(λ_indx[2][3][1])

z = rand(1:5, 100)
z = [1, 1, 2, 4]
z = [2, 2, 3, 4]
using StatsBase
println(StatsBase.counts(z))
println(StatsBase.counts(z, 1:5))
