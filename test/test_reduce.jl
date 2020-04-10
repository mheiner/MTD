using SparseProbVec
using MTD


Omeg = reshape([0.5, 0.5, 0.4, 0.6, 0.9, 0.1, 0.2, 0.8], (2,2,2))
Omeg = reshape([0.5, 0.5, 0.4, 0.6, 0.9, 0.1, 0.2, 0.8, 0.55, 0.45, 0.4, 0.6, 0.85, 0.15, 0.2, 0.8], fill(2,3+1)...)
Omeg = [0.2, 0.5, 0.3]


Q0 = reduce_intercept(Omeg)
A = deepcopy(Q0[3])


# function reduce_ord2(A::Array{T}) where T <: Real
#     order = ndims(A) - 1
#     K = size(A)[1]
#     order == 2 || error("reduce_ord2 requires three dimensional array.")
#
#     lam_cand = Vector{T}(undef, K) # in generalization, K will be replaced with number of lag combos
#     take_cand = Array{T}(undef, (K, K, order)) # in generalization, first current_order dimensions will be the "take candidates", the remaining will index lag combos
#
#     for ℓ = 1:order # in generalization, we'll somehow iterate over all combinations
#         # Mins = reshape( minimum(A, dims=1+setdiff(1:2, ℓ)[1]), fill(K, order)... ) # in generalization, I think this will need to be an array with one less dimension than A
#         Mins = minimum(A, dims=1+setdiff(1:2, ℓ)[1]) # in generalization, I think this will need to be an array with one less dimension than A
#         sums_Mins = sum(Mins, dims=1)
#         lam_cand[ℓ] = minimum(sums_Mins)
#         take_cand[:,:,ℓ] = reshape( Mins ./ sums_Mins , fill(K, order)...)
#     end
#
#     lam_sel, lag_sel_indx = findmax(lam_cand)
#
#     if abs(lam_sel) < 1.0e-6
#         lam_sel = 0.0
#         leftover = deepcopy(A)
#         take = zeros(T, (K,K))
#         lag_sel_indx = 0
#     else
#         leftover = A .- lam_sel .* take_cand[:,:,lag_sel_indx]
#         take = take_cand[:,:,lag_sel_indx]
#     end
#
#     leftover_sums = vec(sum(leftover, dims=1))
#     maximum(abs.( leftover_sums .- leftover_sums[1] )) < 1.0e-6 || error("All leftover marginal sums should be equal.")
#
#     return lam_sel, lag_sel_indx, take, leftover, leftover_sums[1]
# end

# Q1 = reduce_ord2(A)
# A1 = deepcopy(Q1[4])
# reduce_ord2(A1)

## tests of reduce_gen
Omeg = TransTens(reshape([0.5, 0.5, 0.4, 0.6, 0.9, 0.1, 0.2, 0.8, 0.55, 0.45, 0.4, 0.6, 0.85, 0.15, 0.2, 0.8], fill(2,3+1)...))
Q0 = reduce_intercept(Omeg)
Juno.@enter reduce_gen(Q0[3], 1)
Q1a = reduce_gen(Q0[3], 1)
reduce_gen(Q1a[4], 1)
Juno.@enter reduce_gen(Q1a[4], 2)
Q2a = reduce_gen(Q1a[4], 2)
reduce_gen(Q2a[4], 2)
reduce_gen(Q2a[4], 1)

## Other tests
reduce_transTens(Omeg)
reduce_transTens(Omeg, toler=0.5)

Omeg = reshape([0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6], (2,2,2)) # only lag 1 matters
Omeg = reshape([0.7, 0.3, 0.7, 0.3, 0.4, 0.6, 0.4, 0.6], (2,2,2)) # only lag 2 matters
Omeg = reshape([0.7, 0.3, 0.65, 0.35, 0.4, 0.6, 0.4, 0.6], (2,2,2)) # lag 1 matters when lag 2 is first state, doesn't matter when lag 2 is second state

r = reduce_transTens(TransTens(Omeg), leftoversum_tol=1.0e-6)
print(r)

# Juno.@enter reduce_transTens(TransTens(Omeg))

## rDirichlet
K = 2
R = 5
Omeg = reshape(hcat([ rDirichlet(fill(1.0, K)) for i = 1:K^R ]...), fill(K, R+1)...)
out = reduce_transTens(TransTens(Omeg, sum1tol=1.0e-6), leftoversum_tol=1.0e-6)
print(out)
sum(out.levels[:,2])


reduce_transTens(TransTens(out[3][5])) # what we were looking for



## from data
# using BSON: @load, parse
# BSON.parse("/Users/mjheiner/Documents/research/MMTD/MMTDpaper_codeExamples/data/sim1_sets.bson")
# @load "/Users/mjheiner/Documents/research/MMTD/MMTDpaper_codeExamples/data/sim1_sets.bson" y
#
# K = length(unique(y[1]))
# L = 4
# N = zeros(Int, fill(K, L+1)...)
# for i = 1:length(y)
#     N .+= count_trans_L(y[i], K, L)
# end
#
# Omeg_hat = TransTens(transTens_MLE(N))
# out = reduce_transTens(Omeg_hat)
# print(out)

## simulation 1 scenario (not the same as used in paper simulations)
using Random
Random.seed!(11)
L = 4
M = 3
K = 3
λ_indx = build_λ_indx(L, M)
Λ = [0.0, 0.0, 0.0, 1.0]
λ = [ zeros(Float64, λ_indx.lens[m]) for m in 1:M ]
λ[3][3] = 1.0
println(λ_indx.indxs[3][3])
Q0 = zeros(Float64, K)
Q = [ reshape( hcat([ rDirichlet(ones(K)) for k in 1:(K)^m ]...), fill(K, m+1)... ) for m in 1:M ]

Omeg = TransTens(transTensor_mmtd(L, M, K, λ_indx, Λ, λ, Q0, Q))

out = reduce_transTens(Omeg)
sum(out[1][:,2])


## simulation 2 scenario (not the same as used in simulations)
Random.seed!(12)
L = 5
M = 5
K = 2
λ_indx = build_λ_indx(L, M)
Λ = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
λ = [ zeros(Float64, λ_indx.lens[m]) for m in 1:M ]
λ[5][1] = 1.0
println(λ_indx.indxs[5][1])
Q0 = zeros(Float64, K)
Q = [ reshape( hcat([ rDirichlet(ones(K)) for k in 1:(K)^m ]...), fill(K, m+1)... ) for m in 1:M ]
Omeg = TransTens(transTensor_mmtd(L, M, K, λ_indx, Λ, λ, Q0, Q))

out = reduce_transTens(Omeg)
sum(out[1][:,2])

## reduce sum to unity standard in reduce functions
Q2 = reshape([0.32727557, 0.20496972, 0.21370286, 0.25405185, 0.31204655, 0.35311113, 0.17880579, 0.15603652, 0.27311296, 0.27393458, 0.17974408, 0.27320838, 0.29620478, 0.14803316, 0.13574267, 0.42001939, 0.55194678, 0.24776699, 0.11573846, 0.08454777, 0.24961685, 0.32725009, 0.22171401, 0.20141904, 0.29946564, 0.36172226, 0.16753204, 0.17128006, 0.26105888, 0.21806586, 0.22125031, 0.29962494, 0.42724184, 0.27502159, 0.13734542, 0.16039115, 0.26375264, 0.26357236, 0.21962539, 0.25304961, 0.10465893, 0.10479424, 0.10476206, 0.68578476, 0.11517821, 0.11295583, 0.11774202, 0.65412393, 0.28004061, 0.21793946, 0.17852386, 0.32349607, 0.43282710, 0.16505096, 0.26887101, 0.13325093, 0.26559577, 0.35463422, 0.17626712, 0.20350288, 0.12663684, 0.33712198, 0.25674244, 0.27949874],
    fill(4,2+1)...) # pink posterior mean Q2 when p(Lam2) > .5
Q2 = reshape([0.33311281, 0.19747092, 0.21188314, 0.25753312, 0.29564221, 0.37250940, 0.17511323, 0.15673517, 0.27234339, 0.27394530, 0.18228286, 0.27142845, 0.28392956, 0.13990997, 0.12187194, 0.45428853, 0.59164182, 0.22542352, 0.10477350, 0.07816116, 0.24486780, 0.33018902, 0.21656417, 0.20837900, 0.28265837, 0.38362433, 0.16448888, 0.16922841, 0.26011627, 0.21857911, 0.21123421, 0.31007042, 0.46091674, 0.26167019, 0.11828426, 0.15912881, 0.25596878, 0.26842541, 0.21516935, 0.26043645, 0.09823094, 0.08611348, 0.08652709, 0.72912849, 0.10172308, 0.10267278, 0.10688165, 0.68872249, 0.29055474, 0.20131754, 0.18229348, 0.32583424, 0.46557378, 0.16005667, 0.25361717, 0.12075238, 0.26228341, 0.35979494, 0.17587809, 0.20204356, 0.10855796, 0.35744539, 0.25716761, 0.27682904],
    fill(4,2+1)...) # pink posterior mean Q2 when p(Lam2) > .8

sum(Q2, dims=1)
levels, configs, Q, involv = reduce_transTens(TransTens(Q2))
levels
configs
involv
Q[4]
