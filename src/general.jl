# general.jl

## expand.grid() from R
# using Iterators
# collect(product([1 2 3],[1 2]))

function create_froms(K::UInt16, R::UInt16)
  x = UInt16(1):K
  xx = collect(product(repeated(x,Int(R))...))
  xx
end
# similarly,
# x = collect(CartesianRange((2,2)))
# x[2]
# A = randn(2,2)
# A[x[2]]

### creating a transition array
K = 3
R = 3
A = zeros(Float64, repeated(K,R)...)
collect(CartesianRange(size(A)))


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
