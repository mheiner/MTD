# general.jl

export create_froms, count_trans_R;

## expand.grid() from R
# using Iterators
# collect(product([1 2 3],[1 2]))

function create_froms(K::UInt16, R::UInt16)
  x = UInt16(1):K
  xx = collect(product(repeated(x,Int(R))...))
  (xx, length(xx))
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
    assert(minimum(S) >= 1 && maximum(S) <= K)
    N = zeros(Int, (collect(repeated(K,R+1))...))

    for tt in (R+1):length(S)
      Srev_now = S[range(tt,-1,(R+1))]
      N[(Srev_now)...] += 1
    end

    N
end
