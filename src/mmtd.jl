# mmtd.jl

export ParamsMMTD, PriorMMTD, ModMMTD,
  build_λ_indx, sim_mmtd, symmetricPrior_mmtd,
  rpost_lΛ_mmtd, rpost_lλ_mmtd, counttrans_mmtd, rpost_lQ_mmtd,
  rpost_Z_mmtd, rpost_ζ_mmtd; # remove the inner functions after testing

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
  indxs = [ collect(Combinatorics.combinations(1:R, m)) for m in 1:M ]
  lens = [ length(xx) for xx in indxs ]
  (indxs, lens)
end


"""
    sim_mmtd(TT, nburn, R, M, K, λ_indx, Λ, λ, Q)
"""
function sim_mmtd(TT::Int, nburn::Int, R::Int, M::Int, K::Int, λ_indx::Tuple,
  Λ::Vector{Float64}, λ::Vector{Vector{Float64}}, Q::Vector{Array{Float64}})

  Nsim = nburn + TT

  Z = [ StatsBase.sample(WeightVec(Λ)) for i in 1:(Nsim-R) ]
  ζ = [ StatsBase.sample(WeightVec(λ[m])) for i in 1:(Nsim-R), m in 1:M ]

  S = Vector{Int}(Nsim)
  S[1:R] = StatsBase.sample(1:K, R)

  for tt in (R+1):(Nsim)
    i = tt - R
    Slagrev_now = S[range(tt-1, -1, R)]
    pvec = copy( Q[Z[i]][:, Slagrev_now[λ_indx[1][Z[i]][ζ[i,Z[i]]]]...] )
    S[tt] = StatsBase.sample(WeightVec( pvec ))
  end

  S[(nburn+1):(Nsim)], Z[(nburn+1):(Nsim-R)], ζ[(nburn+1):(Nsim-R),:]
end


"""
    symmetricPrior_mmtd(size_Λ, size_λ, size_Q, R, M, K, λ_indx)
"""
function symmetricPrior_mmtd(size_Λ::Float64, size_λ::Float64, size_Q::Float64,
  R::Int, M::Int, K::Int, λ_indx::Tuple)

  # λ_indx[2] contains a tuple of lengths of λ vectors

  α0_Λ = fill( size_Λ / float(M), M)
  α0_λ = [ fill( size_λ / float(λ_indx[2][m]), λ_indx[2][m] ) for m in 1:M ]
  a0_Q = size_Q / float(K)
  α0_Q = [ fill( a0_Q, (K, K^(m)) ) for m in 1:M ]

  (α0_Λ, α0_λ, α0_Q)
end


"""
    rpost_lΛ_mmtd(α0_Λ, Z, M)
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
  for tt in (R+1):(TT)
    Z_now = Z[tt-R]
    Slagrev_now = S[range(tt-1, -1, R)]
    N_out[Z_now][append!( copy(S[tt]), copy(Slagrev_now[ λ_indx[1][Z_now][ζ[tt-R,Z_now]] ]) )...] += 1
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


"""
    rpost_Z_mmtd(S, TT, lΛ, ζ, lQ, λ_indx, R, M)
"""
function rpost_Z_mmtd(S::Vector{Int}, TT::Int,
  lΛ::Vector{Float64}, ζ::Matrix{Int}, lQ::Vector{Array{Float64}},
  λ_indx::Tuple, R::Int, M::Int)

  Z_out = Vector{Float64}(TT-R)
  lp = Vector{Float64}(M)

  for i in 1:(TT-R)
    tt = i + R
    Slagrev_now = S[range(tt-1, -1, R)]
    for m in 1:M
      lp[m] = lΛ[m] + lQ[m][ append!(copy(S[tt]), copy( Slagrev_now[λ_indx[1][m][ ζ[i,m] ]] ))... ]
    end
    Z_out[i] = StatsBase.sample(WeightVec(exp(lp)))
  end

  Z_out
end


"""
    rpost_ζ_mmtd(S, TT, lλ, Z, lQ, λ_indx, R, M, K)
"""
function rpost_ζ_mmtd(S::Vector{Int}, TT::Int,
  lλ::Vector{Vector{Float64}}, Z::Vector{Int}, lQ::Vector{Array{Float64}},
  λ_indx::Tuple, R::Int, M::Int, K::Int)

  λ_lens = copy(λ_indx[2])
  ζ_out = Matrix{Int}(TT-R,M)

  for i in 1:(TT-R)
    tt = i + R
    Slagrev_now = S[range(tt-1, -1, R)]
    for m in 1:M
      if Z[tt] == m
        for j in 1:λ_lens[m]
          lp = Vector{Float64}(λ_lens[m])
          lp[j] = lλ[m][j] + lQ[m][ append!(copy(S[tt]), copy(Slagrev_now[λ_indx[1][m][j]]))... ]
        end
        ζ_out[i,m] = StatsBase.sample(WeightVec( exp(lp) ))
      else
        ζ_out[i,m] = StatsBase.sample(WeightVec( exp(lλ[m]) ))
      end
    end
  end

  ζ_out
end
