# general.jl

## expand.grid() from R
# using Iterators
# collect(product([1 2 3],[1 2]))

function create_froms(K::UInt16, R::UInt16)
  x = UInt16(1):K
  xx = collect(product(repeated(x,Int(R))...))
  xx
end

# froms = create_froms(UInt16(3), UInt16(3))
# A = reshape(collect(1:36)*1.0, (3,3,4))
# A[1,2,3]
# A[(1,2,3)...]
# A[froms[22]...]
