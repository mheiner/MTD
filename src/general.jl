# general.jl

export create_froms, count_trans_R, transTens_MLE, forecDist;

## expand.grid() from R
# using Iterators # (this is part of Base as of Julia 1.0)
# collect(product([1 2 3],[1 2]))

function create_froms(K::UInt16, R::UInt16)
  x = UInt16(1):K
  xx = collect(Iterators.product(Iterators.repeated(x,Int(R))...))
  (vec(xx), length(xx))
end
function create_froms(K::Int, R::Int)
  x = 1:K
  xx = collect(Iterators.product(Iterators.repeated(x,R)...))
  (vec(xx), length(xx))
end


# similarly,
# x = collect(CartesianRange((2,2)))
# x[2]
# A = randn(2,2)
# A[x[2]]

### creating a transition array
# K = 3
# R = 3
# A = zeros(Float64, repeated(K,R)...)
# collect(CartesianRange(size(A)))

# try constructing the mtd transition array with comprehension :)

# froms = create_froms(UInt16(3), UInt16(3))
# A = reshape(collect(1:36)*1.0, (3,3,4))
# A[1,2,3]
# A[(1,2,3)...]
# A[[1,2,3]...]
# A[froms[22]...]
# x = CartesianRange(size(A))
# for i in x
#   @show i
#   println(A[i])
# end
# A[x[22]...] # doesn't work

## will be useful in the MTD extension to order lambdas
# using Combinatorics
# for i in combinations(1:5,3)
#   println(i)
# end
#
# A = collect(Combinatorics.combinations(1:5, 3))


"""
    count_trans_R(S, K, R)

Returns a tensor with counts of observed `R`th order transitions in `S`
  for a chain with `K` states. The first dimension is time now, the next
  dimension is lag 1, then lag 2, and so on.

### Example
```julia
  S = [1,2,1,1,1,2,4,1,3,1,1,3,1,2,1,1,2]
  count_trans_R(S, 4, 2)
```
"""
function count_trans_R(S::Vector{Int}, K::Int, R::Int)
    @assert minimum(S) >= 1 && maximum(S) <= K
    N = zeros(Int, fill(K,R+1)...)

    for tt in (R+1):length(S)
      Srev_now = S[range(tt, step=-1, length=(R+1))]
      N[(Srev_now)...] += 1
    end

    N
end



"""
    transTens_MLE(S, K, R)

Returns a tensor with estimated transition probabilities for `R`th order transitions in `S`
  for a chain with `K` states. The first dimension is time now, the next
  dimension is lag 1, then lag 2, and so on.

### Example
```julia
  S = [1,2,1,1,1,2,4,1,3,1,1,3,1,2,1,1,2]
  count_trans_R(S, 4, 2)
  transTens_MLE(S, 4, 2)
```
"""
function transTens_MLE(S::Vector{Int}, K::Int, R::Int)
    @assert minimum(S) >= 1 && maximum(S) <= K
    N = count_trans_R(S, K, R)
    N ./ sum(N, dims=1)
end
function transTens_MLE(N::Union{Array{Float64}, Array{Int64}})
    N ./ sum(N, dims=1)
end




"""
    forecDist(Slag::Vector{Int}, λ::Vector{Float64}, Q::Matrix{Float64})

Computes the forecast distribution given lagged values and parameters.

### Example
```julia
    lam = [0.6, 0.3, 0.1]
    Q = reshape([0.7, 0.3, 0.25, 0.75], (2,2))
    Slagrev = [1,1,2]
    forecDist(Slagrev, lam, Q)
```
"""
function forecDist(Slagrev::Vector{Int}, λ::Vector{Float64}, Q::Matrix{Float64})
    K = size(Q)[1]
    R = size(λ)[1]
    @assert size(Slagrev)[1] == R
    w = zeros(Float64, K)
    for k in 1:K
        for ℓ in 1:R
            w[k] += λ[ℓ] * Q[k, Slagrev[ℓ]]
        end
    end
    w
end
