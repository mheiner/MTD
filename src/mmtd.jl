# mmtd.jl

export ParamsMMTD, PriorMMTD, ModMMTD,
  build_λ_indx;

type ParamsMMTD
  lΛ::Vector{Float64}
  lλ::Vector{Vector{Float64}}
  Z::Vector{Int} # will be length TT - R
  ζ::Matrix{Int} # (TT-R) by M: these are single indexes of the ordered λ sets
  lQ::Vector{Array{Float64}} # organized so first index is now and lag 1 is the next index
end

immutable PriorMMTD
  α0_Λ::Vector{Float64}
  α0_λ::Vector{Vector{Float64}}
  α0_Q::Vector{Matrix{Float64}} # organized in matricized form
end

type ModMMTD
  R::Int # maximal order
  M::Int # largest order considered
  K::Int # number of states
  TT::UInt64
  S::Vector{Int}
  prior::PriorMMTD
  state::ParamsMMTD
  iter::UInt64

  ModMMTD(R, M, K, TT, S, prior, state) = new(R, M, K, TT, S, prior, state, UInt64(0))
end


"""
    build_λ_indx(R, M)

### Example
```julia
λ_indx = build_λ_indx(3, 2)
```
"""
function build_λ_indx(R::Int, M::Int)
  indxs = tuple( [ tuple(Combinatorics.combinations(1:R, m)...) for m in 1:M ]... )
  lens = tuple( [ length(xx) for xx in indxs ]... )
  (indxs, lens)
end


"""
    rpost_lΛ_mmtd(α0_Λ, Z, M)

Currently counts up whatever Z's it gets, rather than excluding the first R of them.
"""
function rpost_lΛ_mmtd(α0_Λ::Vector{Float64}, Z::Vector{Int}, M::Int)
  Nz = StatsBase.counts(Z, 1:M)
  α1_Λ = α0_Λ + Nz
  BayesInference.rDirichlet(α1_Λ, true)
end


"""
    rpost_lλ_mmtd(α0_λ, ζ, λ_lens, M)
"""
function rpost_lλ_mmtd(α0_λ::Vector{Vector{Float64}}, ζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]

  for m in 1:M
    Nζ = StatsBase.counts(ζ[:,m], 1:λ_lens[m])
    α1_λ = α0_λ[m] + Nζ
    lλ_out[m] = BayesInference.rDirichlet(α1_λ, true)
  end

  lλ_out
end


"""
    counttrans_mmtd(S, TT, Z, ζ, λ_indx, R, M, K)
"""
function counttrans_mmtd(S::Vector{Int}, TT::Int, Z::Vector{Int}, ζ::Matrix{Int},
  λ_indx::Tuple, R::Int, M::Int, K::Int)

  ## initialize
  N_out = [ zeros(Int, (fill(K, m+1)...)) for m in 1:M ]

  ## pass through data and add counts
  for tt in (R+1):TT
    Z_now = Z[tt-R]
    N_out[Z_now][append!(copy(S[tt]), copy(λ_indx[1][ Z_now ][ ζ[tt-R,Z_now] ]))...] += 1
  end

  N_out
end


"""
    rpost_lQ_mmtd(S, TT, α0_Q, Z, ζ, λ_indx, R, M, K)
"""
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int, α0_Q::Vector{Matrix{Float64}},
  Z::Vector{Int}, ζ::Matrix{Int},
  λ_indx::Tuple, R::Int, M::Int, K::Int)

  ## initialize
  lQ_mats = [ Matrix{Float64}(K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], (fill(K, m+1)...)) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Z, ζ, λ_indx, R, M, K)

  for m in 1:M
    ncol = K^m
    Nmat = reshape(N[m], (K, ncol))
    α1_Q = α0_Q[m] + Nmat
    for j in 1:ncol
      lQ_mats[m][:,j] = BayesInference.rDirichlet(α1_Q[:,j], true)
    end
    lQ_out[m] = reshape(lQ_mats[m], (fill(K, m+1)...))
  end

  lQ_out
end
