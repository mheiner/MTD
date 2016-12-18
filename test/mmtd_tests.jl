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

### simulate a data set
Λ = [0.8, 0.2]
λ = [[0.2, 0.6, 0.2], [0.5, 0.3, 0.2]]
Q = [ reshape([0.75, 0.25, 0.5, 0.5], (K,K)),
      reshape([0.1, 0.9,  0.3, 0.7,  0.4, 0.6, 0.8, 0.2], (K,K,K)) ] # organized with first index as current state

S, Z, ζ = sim_mmtd(500, 100, R, M, K, λ_indx, Λ, λ, Q)
