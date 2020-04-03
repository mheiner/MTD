# reduce.jl

Omeg = reshape([0.5, 0.5, 0.4, 0.6, 0.9, 0.1, 0.2, 0.8], (2,2,2))
Omeg = reshape([0.5, 0.5, 0.4, 0.6, 0.9, 0.1, 0.2, 0.8, 0.55, 0.45, 0.4, 0.6, 0.85, 0.15, 0.2, 0.8], fill(2,3+1)...)
Omeg = [0.2, 0.5, 0.3]

struct TransTens
    A::Array
    order::Int
    function TransTens(A::Array{T}) where T <: Real
        order = ndims(A) - 1
        d = size(A)
        all( d.==d[1]) || error("All dimensions of transition tensor must have same length.")
        all( A .>= 0.0 ) || error("Transition tensor must have non-negative entries.")
        maximum(abs.( sum(A, dims=1) .- 1.0 )) < 1.0e-10 || error("Transition tensor must add to unity along first dimension.")
        return new(A, order)
    end
end

Omeg = TransTens(Omeg)

function reduce_intercept(Omeg::TransTens)

    d = Omeg.order + 1
    d > 1 || error("Argument must be a multidimensional array.")

    a0 = vec(minimum(Omeg.A, dims=2:d))
    lam = sum(a0)
    if lam > 0.0
        a = a0 ./ lam
    else
        a = a0 .* 0.0
    end

    leftover = Omeg.A .- a0
    leftover_sums = vec(sum(leftover, dims=1))
    maximum(abs.( leftover_sums .+ lam .- 1.0 )) < 1.0e-10 || error("All leftover marginal sums should be equal.")

    return (lam, a, leftover, leftover_sums[1])
end

Q0 = reduce_intercept(Omeg)
A = deepcopy(Q0[3])

function reduce_ord2(A::Array{T}) where T <: Real
    order = ndims(A) - 1
    K = size(A)[1]
    order == 2 || error("reduce_ord2 requires three dimensional array.")

    lam_cand = Vector{T}(undef, K) # in generalization, K will be replaced with number of lag combos
    take_cand = Array{T}(undef, (K, K, order)) # in generalization, first current_order dimensions will be the "take candidates", the remaining will index lag combos

    for ℓ = 1:order # in generalization, we'll somehow iterate over all combinations
        # Mins = reshape( minimum(A, dims=1+setdiff(1:2, ℓ)[1]), fill(K, order)... ) # in generalization, I think this will need to be an array with one less dimension than A
        Mins = minimum(A, dims=1+setdiff(1:2, ℓ)[1]) # in generalization, I think this will need to be an array with one less dimension than A
        sums_Mins = sum(Mins, dims=1)
        lam_cand[ℓ] = minimum(sums_Mins)
        take_cand[:,:,ℓ] = reshape( Mins ./ sums_Mins , fill(K, order)...)
    end

    lam_sel, lag_sel_indx = findmax(lam_cand)

    if abs(lam_sel) < 1.0e-10
        lam_sel = 0.0
        leftover = deepcopy(A)
        take = zeros(T, (K,K))
        lag_sel_indx = 0
    else
        leftover = A .- lam_sel .* take_cand[:,:,lag_sel_indx]
        take = take_cand[:,:,lag_sel_indx]
    end

    leftover_sums = vec(sum(leftover, dims=1))
    maximum(abs.( leftover_sums .- leftover_sums[1] )) < 1.0e-10 || error("All leftover marginal sums should be equal.")

    return lam_sel, lag_sel_indx, take, leftover, leftover_sums[1]
end

## tests
Q1 = reduce_ord2(A)
A1 = deepcopy(Q1[4])

reduce_ord2(A1)


function reduce_gen(A::Array{T}, sub_order::Int) where T <: Real

    order = ndims(A) - 1
    d = size(A)
    K = d[1]

    sub_order > 0 && sub_order < order || order("Sub-order cannot be higher than tensor order.")
    all( d.==d[1]) || error("All dimensions of tensor must have same length.")
    # all( A .>= 0.0 ) || error("Tensor must have non-negative entries.")

    lam_indx = build_λ_indx(order, sub_order)
    n_cand = lam_indx.lens[sub_order]
    configs = lam_indx.indxs[sub_order]

    lam_cand = Vector{T}(undef, n_cand)
    take_cand = Array{T}(undef, vcat(fill(K, sub_order+1), n_cand)...)
    take_cand = Vector{Array{T}}(undef, n_cand)

    for ℓ = 1:n_cand
        Mins = minimum(A, dims= 1 .+ setdiff(1:order, configs[ℓ]))
        sums_Mins = sum(Mins, dims=1)
        lam_cand[ℓ] = minimum(sums_Mins)
        take_cand[ℓ] =  Mins ./ sums_Mins
    end

    lam_sel, lag_sel_indx = findmax(lam_cand)

    if abs(lam_sel) < 1.0e-10
        lam_sel = 0.0
        leftover = deepcopy(A)
        take = zeros(T, 1 + sub_order)
        lag_sel_indx = 0
        config_out = nothing
    else
        leftover = A .- lam_sel .* take_cand[lag_sel_indx] # do not reshpae yet; automatically broadcasts (should set up test in case broadcasting behavior changes someday)
        take = reshape(take_cand[lag_sel_indx], fill(K, 1+sub_order)...)
        config_out = configs[lag_sel_indx]
    end

    leftover_sums = vec(sum(leftover, dims=1))
    maximum(abs.( leftover_sums .- leftover_sums[1] )) < 1.0e-10 || error("All leftover marginal sums should be equal.")

    return lam_sel, (lag_sel_indx, config_out), take, leftover, leftover_sums[1]
end

## tests
Omeg = TransTens(reshape([0.5, 0.5, 0.4, 0.6, 0.9, 0.1, 0.2, 0.8, 0.55, 0.45, 0.4, 0.6, 0.85, 0.15, 0.2, 0.8], fill(2,3+1)...))
Q0 = reduce_intercept(Omeg)
Juno.@enter reduce_gen(Q0[3], 1)
Q1a = reduce_gen(Q0[3], 1)
reduce_gen(Q1a[4], 1)
Juno.@enter reduce_gen(Q1a[4], 2)
Q2a = reduce_gen(Q1a[4], 2)
reduce_gen(Q2a[4], 2)
reduce_gen(Q2a[4], 1)


function reduce_transTens(Omeg::TransTens; toler=0.001, maxiter=100)
    ord = Omeg.order

    lam = [0.0]
    Q = Vector{Array{Float64}}(undef, 1)
    configs = Vector{Vector{Int}}(undef, 1)
    configs[1] = [0]
    level = [0]
    total_lag_involvement = zeros(ord)

    lam_now, Q_now, A, whatsleft = reduce_intercept(Omeg)

    lam[1] = copy(lam_now)
    Q[1] = deepcopy(Q_now)

    iter = 1
    sub_ord_now = 1

    while sub_ord_now < ord && whatsleft >= toler && iter < maxiter

        lam_now, lagsel_now, Q_now, A, whatsleft = reduce_gen(A, sub_ord_now)

        if lam_now > 0
            push!(lam, lam_now)
            push!(configs, lagsel_now[2])
            push!(level, sub_ord_now)
            push!(Q, Q_now)
            total_lag_involvement[lagsel_now[2]] .+= lam_now
        else
            sub_ord_now += 1
        end

        iter += 1

    end

    if sub_ord_now == ord # means there was no more lower orders could do
        push!(lam, whatsleft)
        push!(configs, collect(1:ord))
        push!(level, sub_ord_now)
        push!(Q, A ./ whatsleft)
        total_lag_involvement .+= whatsleft
    end

    return hcat(level, lam), configs, Q, total_lag_involvement

end

## tests
using MTD
reduce_transTens(Omeg)
reduce_transTens(Omeg, toler=0.5)

Omeg = reshape([0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6], (2,2,2)) # only lag 1 matters
Omeg = reshape([0.7, 0.3, 0.7, 0.3, 0.4, 0.6, 0.4, 0.6], (2,2,2)) # only lag 2 matters
Omeg = reshape([0.7, 0.3, 0.65, 0.35, 0.4, 0.6, 0.4, 0.6], (2,2,2)) # lag 1 matters when lag 2 is first state, doesn't matter when lag 2 is second state

reduce_transTens(TransTens(Omeg))

# Juno.@enter reduce_transTens(TransTens(Omeg))

## rDirichlet
using SparseProbVec

K = 3
R = 5
Omeg = reshape(hcat([ rDirichlet(fill(1.0, K)) for i = 1:K^R ]...), fill(K, R+1)...)
out = reduce_transTens(TransTens(Omeg))
sum(out[1][:,2])


## from data
using BSON: @load, parse
BSON.parse("/Users/mjheiner/Documents/research/MMTD/MMTDpaper_codeExamples/data/sim1_sets.bson")
@load "/Users/mjheiner/Documents/research/MMTD/MMTDpaper_codeExamples/data/sim1_sets.bson" y

K = length(unique(y[1]))
L = 4
N = zeros(Int, fill(K, L+1)...)
for i = 1:length(y)
    N .+= count_trans_L(y[i], K, L)
end

Omeg_hat = TransTens(transTens_MLE(N))
out = reduce_transTens(Omeg_hat)
sum(out[1][:,2])

## simulation 1 scenario
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


## simulation 2 scenario
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
