# mmtd.jl

export ParamsMMTD, PriorMMTD, ModMMTD, λindxMMTD,
  build_λ_indx, ZζtoZandζ, sim_mmtd, symmetricDirPrior_mmtd, transTensor_mmtd,
  rpost_lΛ_mmtd, rpost_lλ_mmtd, counttrans_mmtd, rpost_lQ_mmtd,
  rpost_Zζ_mtd_marg, MetropIndep_ΛλZζ, mcmc_mmtd!; # remove the inner functions after testing

type ParamsMMTD
  lΛ::Vector{Float64}
  lλ::Vector{Vector{Float64}}
  Zζ::Vector{Int} # will be length TT - R, mapped through ZζtoZandζ()
  lQ::Vector{Array{Float64}} # organized so first index is now and lag 1 is the next index
end

type PriorMMTD
  Λ::Union{Vector{Float64}, SparseDirMixPrior, SparseSBPrior, SparseSBPriorFull}
  λ::Union{Vector{Vector{Float64}}, Vector{SparseDirMixPrior}, Vector{SparseSBPrior}, Vector{SparseSBPriorP}, Vector{SparseSBPriorFull}}
  Q::Union{Vector{Array{Float64}}, Vector{Array{SparseDirMixPrior}}, Vector{Array{SparseSBPrior}}, Vector{Array{SparseSBPriorP}}}
end

type λindxMMTD
    indxs::Vector{Array{Array{Int64,1},1}}
    lens::Vector{Int64}
    Zζindx::Matrix{Int64}
    nZζ::Int64
end

type ModMMTD
  R::Int # maximal order
  M::Int # largest order considered
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMMTD
  state::ParamsMMTD
  λ_indx::λindxMMTD
  iter::UInt64

  ModMMTD(R, M, K, TT, S, prior, state, λ_indx) = new(R, M, K, TT, S, prior, state, λ_indx, UInt64(0))
end

type PostSimsMMTD
  Λ::Matrix{Float64}
  λ::Array{Matrix{Float64}}
  Q::Array{Matrix{Float64}} # stored matricized
  Zζ::Matrix{Int} # index 1 is iteration, index 2 is time, entries are Zζ
  p1λ::Matrix{Float64} # SBM π
  p1Q::Vector{Matrix{Float64}} # SBM π

  PostSimsMMTD(Λ, λ, Q, Zζ, p1λ, p1Q) = new(Λ, λ, Q, Zζ, p1λ, p1Q)
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
  ζindx = vcat([collect(1:zz) for zz in lens]...)
  Zindx = vcat([fill(m, lens[m]) for m in 1:M]...)
  nZζ = sum(lens)
  Zζindx = hcat(Zindx, ζindx)
  λindxMMTD(indxs, lens, Zζindx, nZζ)
end

function ZζtoZandζ(Zζ::Vector{Int64}, λ_indx)
    out = [λ_indx.Zζindx[Zζ[tt],j] for tt in 1:length(Zζ), j in 1:2]
end


"""
    sim_mmtd(TT, nburn, R, M, K, λ_indx, Λ, λ, Q)
"""
function sim_mmtd(TT::Int, nburn::Int, R::Int, M::Int, K::Int, λ_indx::λindxMMTD,
  Λ::Array{Float64}, λ::Array{Vector{Float64}}, Q#=::Array{Array{Float64}}=#)

  Nsim = nburn + TT

  Z = [ StatsBase.sample(Weights(Λ)) for i in 1:(Nsim-R) ]
  ζ = [ StatsBase.sample(Weights(λ[m])) for i in 1:(Nsim-R), m in 1:M ]

  S = Vector{Int}(Nsim)
  S[1:R] = StatsBase.sample(1:K, R)

  for tt in (R+1):(Nsim)
    i = tt - R
    Slagrev_now = S[range(tt-1, -1, R)]
    pvec = copy( Q[Z[i]][:, Slagrev_now[λ_indx.indxs[Z[i]][ζ[i,Z[i]]]]...] )
    S[tt] = StatsBase.sample(Weights( pvec ))
  end

  S[(nburn+1):(Nsim)], Z[(nburn+1):(Nsim-R)], ζ[(nburn+1):(Nsim-R),:]
end


"""
    symmetricPrior_mmtd(size_Λ, size_λ, size_Q, R, M, K, λ_indx)
"""
function symmetricDirPrior_mmtd(size_Λ::Float64, size_λ::Float64, size_Q::Float64,
  R::Int, M::Int, K::Int, λ_indx::λindxMMTD)

  # λ_indx.lens contains a tuple of lengths of λ vectors

  α0_Λ = fill( size_Λ / float(M), M)
  α0_λ = [ fill( size_λ / float(λ_indx.lens[m]), λ_indx.lens[m] ) for m in 1:M ]
  a0_Q = size_Q / float(K)
  # α0_Q = [ fill( a0_Q, (K, K^(m)) ) for m in 1:M ]
  α0_Q = [ fill( a0_Q, fill(K, (m+1))... ) for m in 1:M ]

  (α0_Λ, α0_λ, α0_Q)
end


"""
    transTensor_mmtd(R, M, K, λ_indx, Λ, λ, Q)

Calculate full transition tensor from Λ, λ, and Q.
"""
function transTensor_mmtd(R::Int, M::Int, K::Int, λ_indx::λindxMMTD,
  Λ::Vector{Float64}, λ::Vector{Vector{Float64}}, Q#=::Vector{Array{Float64}}=#)

  froms, nfroms = create_froms(K, R) # in this case, ordered by now, lag1, lag2, etc.
  Ωmat = zeros(Float64, (K, nfroms))
  for i in 1:nfroms
    for k in 1:K
      for m in 1:M
        for ℓ in 1:λ_indx.lens[m]
            Ωmat[k,i] += Λ[m] .* λ[m][ℓ] .* Q[m][k,froms[i][[λ_indx.indxs[m][ℓ]]...]...]
        end
      end
    end
  end
  reshape(Ωmat, (fill(K,R+1)...))
end


"""
    rpost_lΛ_mmtd(α0_Λ, Z, M)
"""
function rpost_lΛ_mmtd(α0_Λ::Vector{Float64}, Z::Vector{Int}, M::Int)
  Nz = StatsBase.counts(Z, 1:M)
  α1_Λ = α0_Λ + Nz
  BayesInference.rDirichlet(α1_Λ, true)
end
function rpost_lΛ_mmtd(prior::SparseDirMixPrior,
    Z::Vector{Int}, M::Int)
  Nz = StatsBase.counts(Z, 1:M)
  α1_Λ = prior.α + Nz
  BayesInference.rSparseDirMix(α1_Λ, prior.β, true)
end
function rpost_lΛ_mmtd!(prior::SparseSBPriorFull,
    Z::Vector{Int}, M::Int)

  lw_now, lz_now, ξ_now, prior.μ_now, prior.p1_now = BayesInference.rpost_sparseStickBreak(Z,
    prior.p1_now, prior.α, prior.μ_now, prior.M, prior.a_p1,
    prior.b_p1, prior.a_μ, prior.b_μ, true)

  lw_now
end

"""
    rpost_lλ_mmtd(α0_λ, ζ, λ_lens, M)
"""
function rpost_lλ_mmtd(α0_λ::Vector{Vector{Float64}}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]
  Z = copy(Zandζ[:,1])
  ζ = copy(Zandζ[:,2])

  for m in 1:M
    Zmindx = find(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    α1_λ = α0_λ[m] .+ Nζ
    lλ_out[m] = BayesInference.rDirichlet(α1_λ, true)
  end

  lλ_out
end
function rpost_lλ_mmtd(prior::Vector{SparseDirMixPrior}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]
  Z = copy(Zandζ[:,1])
  ζ = copy(Zandζ[:,2])

  for m in 1:M
    Zmindx = find(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    α1_λ = prior[m].α .+ Nζ
    lλ_out[m] = BayesInference.rSparseDirMix(α1_λ, prior[m].β, true)
  end

  lλ_out
end
function rpost_lλ_mmtd(prior::Vector{SparseSBPrior}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]
  Z = copy(Zandζ[:,1])
  ζ = copy(Zandζ[:,2])

  for m in 1:M
    Zmindx = find(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    lw_now, lz_now, ξ_now = BayesInference.rpost_sparseStickBreak(Nζ,
        prior[m].p1, prior[m].α,
        prior[m].μ, prior[m].M, true )

    lλ_out[m] = copy(lw_now)
  end

  lλ_out
end
function rpost_lλ_mmtd!(prior::Vector{SparseSBPriorP}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]
  Z = copy(Zandζ[:,1])
  ζ = copy(Zandζ[:,2])

  for m in 1:M
    Zmindx = find(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    lw_now, lz_now, ξ_now, prior[m].p1_now = BayesInference.rpost_sparseStickBreak(Nζ,
        prior[m].p1_now, prior[m].α, prior[m].μ, prior[m].M,
        prior[m].a_p1, prior[m].b_p1, true)

    lλ_out[m] = copy(lw_now)
  end

  lλ_out
end
function rpost_lλ_mmtd!(prior::Vector{SparseSBPriorFull}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]
  Z = copy(Zandζ[:,1])
  ζ = copy(Zandζ[:,2])

  for m in 1:M
    Zmindx = find(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    lw_now, lz_now, ξ_now, prior[m].μ_now, prior[m].p1_now = BayesInference.rpost_sparseStickBreak(Nζ,
        prior[m].p1_now, prior[m].α, prior[m].μ_now, prior[m].M,
        prior[m].a_p1, prior[m].b_p1, prior[m].a_μ, prior[m].b_μ, true)

    lλ_out[m] = copy(lw_now)
  end

  lλ_out
end


"""
    counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)

    Indexing on ouput is [t, t-1, t-2, etc.]

    ### Example
    ```julia
    R = 2
    M = 2
    K = 3
    TT = 12
    S = [1,2,1,3,3,1,2,1,3,2,1,1]
    Zandζ = [1,2; 1,2;, 1;2, 1,1; 2,1; 1;2, 2,1; 2,1; 2,1; 1,1]
    λ_indx = build_λ_indx(R, M)
    counttrans_mmtd(S::Vector{Int}, TT::Int, Zandζ::Matrix{Int},
        λ_indx::λindxMMTD, R::Int, M::Int, K::Int)
    ```
"""
function counttrans_mmtd(S::Vector{Int}, TT::Int, Zandζ::Matrix{Int},
  λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  Z = copy(Zandζ[:,1])
  ζ = copy(Zandζ[:,2])

  ## initialize
  N_out = [ zeros(Int, (fill(K, m+1)...)) for m in 1:M ]

  ## pass through data and add counts
  for tt in (R+1):(TT)
    Z_now = Z[tt-R]
    Slagrev_now = S[range(tt-1, -1, R)]
    N_out[Z_now][append!( [copy(S[tt])], copy(Slagrev_now[ λ_indx.indxs[Z_now][ζ[tt-R]] ]) )...] += 1
  end

  N_out
end


"""
    counttrans_mtd(S, TT, ζ, R, K)

    ### Example
    ```julia
    R = 2
    K = 3
    TT = 12
    S = [1,2,1,3,3,1,2,1,3,2,1,1]
    ζ = [1,2,1,2,1,2,1,2,1,2]
      counttrans_mtd(S, TT, ζ, R, K)
    ```
"""
function counttrans_mtd(S::Vector{Int}, TT::Int, ζ::Vector{Int},
  R::Int, K::Int)

  ## initialize
  N_out = zeros(Int, (K,K))

  ## pass through data and add counts
  for tt in (R+1):(TT)
    Slagrev_now = copy( S[range(tt-1, -1, R)] )
    from = copy( Slagrev_now[ ζ[tt-R] ] )
    N_out[ S[tt], from ] += 1
  end

  N_out
end


"""
    rpost_lQ_mmtd(S, TT, prior, Z, ζ, λ_indx, R, M, K)
"""
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int, prior::Vector{Array{Float64}},
  Zandζ::Matrix{Int},
  λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  ## initialize
  α0_Q = copy(prior) # vector of arrays
  lQ_mats = [ Matrix{Float64}(K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], (fill(K, m+1)...)) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K) # vector of arrays

  for m in 1:M
    ncol = K^m
    α1_Q = α0_Q[m] .+ N[m]
    α1_Q_mat = reshape(α1_Q, (K, ncol))
    for j in 1:ncol
      lQ_mats[m][:,j] = BayesInference.rDirichlet(α1_Q_mat[:,j], true)
    end
    lQ_out[m] = reshape(lQ_mats[m], (fill(K, m+1)...))
  end

  lQ_out
end
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int,
  prior::Vector{Array{SparseDirMixPrior}},
  Zandζ::Matrix{Int},
  λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  prior_vec = [ reshape(prior[m], K^m ) for m in 1:M ]

  ## initialize
  lQ_mats = [ Matrix{Float64}(K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], (fill(K, m+1)...)) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)

  for m in 1:M
    ncol = K^m
    Nmat = reshape(N[m], (K, ncol))
    for j in 1:ncol
      α1 = prior_vec[m][j].α .+ Nmat[:,j]
      lQ_mats[m][:,j] = BayesInference.rSparseDirMix(α1, prior_vec[m][j].β, true)
    end
    lQ_out[m] = reshape(lQ_mats[m], (fill(K, m+1)...))
  end

  lQ_out
end
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int, prior::Vector{Array{SparseSBPrior}},
    Zandζ::Matrix{Int}, λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  prior_vec = [ reshape(prior[m], K^m ) for m in 1:M ]

  ## initialize
  lQ_mats = [ Matrix{Float64}(K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], (fill(K, m+1)...)) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)

  for m in 1:M
    ncol = K^m
    Nmat = reshape(N[m], (K, ncol))
    for j in 1:ncol
      lQ_mats[m][:,j] = BayesInference.rpost_sparseStickBreak(Nmat[:,j],
        prior_vec[m][j].p1, prior_vec[m][j].α, prior_vec[m][j].μ, prior_vec[m][j].M, true)[1]
    end
    lQ_out[m] = reshape(lQ_mats[m], (fill(K, m+1)...))
  end

  lQ_out
end
function rpost_lQ_mmtd!(S::Vector{Int}, TT::Int, prior::Vector{Array{SparseSBPriorP}},
    Zandζ::Matrix{Int}, λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  prior_vec = [ reshape(prior[m], K^m ) for m in 1:M ]

  ## initialize
  lQ_mats = [ Matrix{Float64}(K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], (fill(K, m+1)...)) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)

  for m in 1:M
      ncol = K^m
      Nmat = reshape(N[m], (K, ncol))
      for j in 1:ncol

          lw_now, lz_now, ξ_now, p1_now = BayesInference.rpost_sparseStickBreak(
            Nmat[:,j], prior_vec[m][j].p1_now,
            prior_vec[m][j].α, prior_vec[m][j].μ, prior_vec[m][j].M,
            prior_vec[m][j].a_p1, prior_vec[m][j].b_p1, true )

          lQ_mats[m][:,j] = copy(lw_now)
          prior_vec[m][j].p1_now = copy(p1_now)

      end
      lQ_out[m] = reshape(lQ_mats[m], (fill(K, m+1)...))
  end

  prior = [ reshape(prior_vec[m], (fill(K,m))... ) for m in 1:M ]

  lQ_out
end


"""
    rpost_Zζ_marg(S, Zζ_old, λ_indx, prior_Q, lΛ, lλ, TT, R, K)

    Full conditinal updates for Zζ marginalizing over Q
"""
function rpost_Zζ_marg(S::Vector{Int}, Zζ_old::Vector{Int}, λ_indx::λindxMMTD,
    prior_Q::Vector{Array{Float64}},
    lΛ::Vector{Float64},
    lλ::Vector{Vector{Float64}},
    TT::Int, R::Int, M::Int, K::Int)

  α0_Q = copy(prior_Q) # vector of arrays
  # α0_Qarrays = [ reshape(α0_Q[m], (fill(K, m+1)...)) for m in 1:M ]

  Zζ_out = copy(Zζ_old)
  Zandζ_now = ZζtoZandζ(Zζ_out, λ_indx) # won't get updated

  ## calculate current log marginal likelihood
  N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
  # N_now_mats = [ reshape(N_now[m], K, K^m) for m in 1:M ]
  α1_Q = α0_Q .+ N_now # gets re-used and updated
  α1_Q_mats = [ reshape(α1_Q[m], K, K^m) for m in 1:M ] # used only for initial calculation

  llikmarg = 0.0
  for m in 1:M
      for kk in 1:(K^m)
          llikmarg += lmvbeta( α1_Q_mats[m][:,kk] )
      end
  end

  ## for each time point, modify llikmarg accordingly
  for i in 1:(TT-R)  # i indexes Zζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, -1, R)]

    # Zold, ζold = copy(Zandζ_out[i,1]), copy(Zandζ_out[i,2])
    Zold, ζold = copy(λ_indx.Zζindx[Zζ_out[i],1]), copy(λ_indx.Zζindx[Zζ_out[i],2])
    eSt = [ 1.0 * (kk==S[tt]) for kk in 1:K ]

    # remove S_t from llikmarg
    SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zold][ζold] ])
    llikmarg -= lmvbeta( α1_Q[Zold][:,SlagrevZζnow...] )
    α1_Q[Zold][:,SlagrevZζnow...] -= eSt
    llikmarg += lmvbeta( α1_Q[Zold][:,SlagrevZζnow...] )

    # calculate llikmarg and update probability under each possible Zζ
    llikmarg_cand = fill(llikmarg, λ_indx.nZζ)
    lw = zeros(Float64, λ_indx.nZζ)

    for ℓ in 1:λ_indx.nZζ
        Zcand, ζcand = copy(λ_indx.Zζindx[ℓ,1]), copy(λ_indx.Zζindx[ℓ,2])
        SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

        llikmarg_cand[ℓ] -= lmvbeta( α1_Q[Zcand][:,SlagrevZζnow...] )
        llikmarg_cand[ℓ] += lmvbeta( α1_Q[Zcand][:,SlagrevZζnow...] .+ eSt )

        lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
    end

    w = exp.( lw - maximum(lw) )
    newindx = StatsBase.sample(Weights( w ))
    Zζ_out[i] = copy(newindx)

    ## update running quantities
    Znew, ζnew = copy(λ_indx.Zζindx[newindx,1]), copy(λ_indx.Zζindx[newindx,2])
    α1_Q[Znew][:,Slagrev_now[ λ_indx.indxs[Znew][ζnew] ]...] += eSt
    llikmarg = copy(llikmarg_cand[newindx])

  end

  Zζ_out
end
function rpost_Zζ_marg(S::Vector{Int}, Zζ_old::Vector{Int},
    λ_indx::λindxMMTD,
    prior_Q::Vector{Array{SparseDirMixPrior}},
    lΛ::Vector{Float64},
    lλ::Vector{Vector{Float64}},
    TT::Int, R::Int, M::Int, K::Int)

  Zζ_out = copy(Zζ_old)
  Zandζ_now = ZζtoZandζ(Zζ_out, λ_indx) # won't get updated
  prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

  ## calculate current log marginal likelihood
  N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
  N_now_mats = [ reshape(N_now[m], K, K^m) for m in 1:M ] # Ns won't be updated, only llikmarg

  llikmarg = 0.0
  for m in 1:M
      for ℓ in 1:(K^m)
          llikmarg += logSDMmarginal(N_now_mats[m][:,ℓ], prior_Q_vec[m][ℓ].α, prior_Q_vec[m][ℓ].β)
      end
  end

  ## for each time point, modify llikmarg accordingly
  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, -1, R)]

    Zold, ζold = copy(λ_indx.Zζindx[Zζ_out[i],1]), copy(λ_indx.Zζindx[Zζ_out[i],2])
    eSt = [ 1*(kk==S[tt]) for kk in 1:K ]

    # remove S_t from llikmarg
    SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zold][ζold] ])
    llikmarg -= logSDMmarginal( N_now[Zold][:,SlagrevZζnow...],
                                prior_Q[Zold][SlagrevZζnow...].α,
                                prior_Q[Zold][SlagrevZζnow...].β )
            # prior for Q indexed with Slagrev_now[ λ_indx.indxs[Zold][ζold] ]...
    N_now[Zold][:,SlagrevZζnow...] -= eSt
    llikmarg += logSDMmarginal( N_now[Zold][:,SlagrevZζnow...],
                                prior_Q[Zold][SlagrevZζnow...].α,
                                prior_Q[Zold][SlagrevZζnow...].β )

    # calculate llikmarg and update probability under each possible Zζ
    llikmarg_cand = fill(llikmarg, λ_indx.nZζ)
    lw = zeros(Float64, λ_indx.nZζ)

    for ℓ in 1:λ_indx.nZζ
        Zcand, ζcand = copy(λ_indx.Zζindx[ℓ,1]), copy(λ_indx.Zζindx[ℓ,2])
        SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

        llikmarg_cand[ℓ] -= logSDMmarginal( N_now[Zcand][:,SlagrevZζnow...],
                                    prior_Q[Zcand][SlagrevZζnow].α,
                                    prior_Q[Zcand][SlagrevZζnow].β )
        llikmarg_cand[ℓ] += logSDMmarginal( N_now[Zcand][:,SlagrevZζnow...] .+ eSt,
                                    prior_Q[Zcand][SlagrevZζnow].α,
                                    prior_Q[Zcand][SlagrevZζnow].β )

        lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
    end

    w = exp.( lw - maximum(lw) )
    newindx = StatsBase.sample(Weights( w ))
    Zζ_out[i] = copy(newindx)

    ## update running quantities
    Znew, ζnew = copy(λ_indx.Zζindx[newindx,1]), copy(λ_indx.Zζindx[newindx,2])
    N_now[Znew][:,Slagrev_now[ λ_indx.indxs[Znew][ζnew] ]...] += eSt
    llikmarg = copy(llikmarg_cand[newindx])

  end

  Zζ_out
end
function rpost_Zζ_marg(S::Vector{Int}, Zζ_old::Vector{Int},
    λ_indx::λindxMMTD,
    prior_Q::Union{Vector{Array{SparseSBPrior}}, Vector{Array{SparseSBPriorP}}},
    lΛ::Vector{Float64},
    lλ::Vector{Vector{Float64}},
    TT::Int, R::Int, M::Int, K::Int)

  Zζ_out = copy(Zζ_old)
  Zandζ_now = ZζtoZandζ(Zζ_out, λ_indx) # won't get updated
  prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

  SBM_flag = typeof(prior_Q) == Vector{Array{SparseSBPrior}}
  SBMp_flag = typeof(prior_Q) == Vector{Array{SparseSBPriorP}}

  ## calculate current log marginal likelihood
  N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
  N_now_mats = [ reshape(N_now[m], K, K^m) for m in 1:M ] # Ns won't be updated, only llikmarg

  llikmarg = 0.0
  if SBM_flag
      for m in 1:M
          for ℓ in 1:(K^m)
            llikmarg += logSBMmarginal( N_now_mats[m][:,ℓ],
                                        prior_Q_vec[m][ℓ].p1,
                                        prior_Q_vec[m][ℓ].α,
                                        prior_Q_vec[m][ℓ].μ,
                                        prior_Q_vec[m][ℓ].M )

          end
      end
    elseif SBMp_flag
      for m in 1:M
          for ℓ in 1:(K^m)
            llikmarg += logSBMmarginal( N_now_mats[m][:,ℓ],
                                        prior_Q_vec[m][ℓ].p1_now,
                                        prior_Q_vec[m][ℓ].α,
                                        prior_Q_vec[m][ℓ].μ,
                                        prior_Q_vec[m][ℓ].M )
          end
      end
  end


  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
      tt = i + R
      Slagrev_now = S[range(tt-1, -1, R)]

      Zold, ζold = copy(λ_indx.Zζindx[Zζ_out[i],1]), copy(λ_indx.Zζindx[Zζ_out[i],2])
      eSt = [ 1*(kk==S[tt]) for kk in 1:K ]

      # remove S_t from llikmarg
      SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zold][ζold] ])

      if SBM_flag
          llikmarg -= logSBMmarginal( N_now[Zold][:,SlagrevZζnow...],
                                      prior_Q[Zold][SlagrevZζnow...].p1,
                                      prior_Q[Zold][SlagrevZζnow...].α,
                                      prior_Q[Zold][SlagrevZζnow...].μ,
                                      prior_Q[Zold][SlagrevZζnow...].M  )
                    # prior for Q indexed with Slagrev_now[ λ_indx.indxs[Zold][ζold] ]...
          N_now[Zold][:,SlagrevZζnow...] -= eSt
          llikmarg += logSBMmarginal( N_now[Zold][:,SlagrevZζnow...],
                                      prior_Q[Zold][SlagrevZζnow...].p1,
                                      prior_Q[Zold][SlagrevZζnow...].α,
                                      prior_Q[Zold][SlagrevZζnow...].μ,
                                      prior_Q[Zold][SlagrevZζnow...].M  )
      elseif SBMp_flag
          llikmarg -= logSBMmarginal( N_now[Zold][:,SlagrevZζnow...],
                                      prior_Q[Zold][SlagrevZζnow...].p1_now,
                                      prior_Q[Zold][SlagrevZζnow...].α,
                                      prior_Q[Zold][SlagrevZζnow...].μ,
                                      prior_Q[Zold][SlagrevZζnow...].M  )
                    # prior for Q indexed with Slagrev_now[ λ_indx.indxs[Zold][ζold] ]...
          N_now[Zold][:,SlagrevZζnow...] -= eSt
          llikmarg += logSBMmarginal( N_now[Zold][:,SlagrevZζnow...],
                                      prior_Q[Zold][SlagrevZζnow...].p1_now,
                                      prior_Q[Zold][SlagrevZζnow...].α,
                                      prior_Q[Zold][SlagrevZζnow...].μ,
                                      prior_Q[Zold][SlagrevZζnow...].M  )
      end

      # calculate llikmarg and update probability under each possible Zζ
      llikmarg_cand = fill(llikmarg, λ_indx.nZζ)
      lw = zeros(Float64, λ_indx.nZζ)

      if SBM_flag
          for ℓ in 1:λ_indx.nZζ
              Zcand, ζcand = copy(λ_indx.Zζindx[ℓ,1]), copy(λ_indx.Zζindx[ℓ,2])
              SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

              llikmarg_cand[ℓ] -= logSBMmarginal( N_now[Zcand][:,SlagrevZζnow...],
                                                  prior_Q[Zcand][SlagrevZζnow...].p1,
                                                  prior_Q[Zcand][SlagrevZζnow...].α,
                                                  prior_Q[Zcand][SlagrevZζnow...].μ,
                                                  prior_Q[Zcand][SlagrevZζnow...].M  )
              llikmarg_cand[ℓ] += logSBMmarginal( N_now[Zcand][:,SlagrevZζnow...] .+ eSt,
                                                  prior_Q[Zcand][SlagrevZζnow...].p1,
                                                  prior_Q[Zcand][SlagrevZζnow...].α,
                                                  prior_Q[Zcand][SlagrevZζnow...].μ,
                                                  prior_Q[Zcand][SlagrevZζnow...].M  )

              lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
          end
        elseif SBMp_flag
          for ℓ in 1:λ_indx.nZζ
              Zcand, ζcand = copy(λ_indx.Zζindx[ℓ,1]), copy(λ_indx.Zζindx[ℓ,2])
              SlagrevZζnow = copy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

              llikmarg_cand[ℓ] -= logSBMmarginal( N_now[Zcand][:,SlagrevZζnow...],
                                                  prior_Q[Zcand][SlagrevZζnow...].p1_now,
                                                  prior_Q[Zcand][SlagrevZζnow...].α,
                                                  prior_Q[Zcand][SlagrevZζnow...].μ,
                                                  prior_Q[Zcand][SlagrevZζnow...].M  )
              llikmarg_cand[ℓ] += logSBMmarginal( N_now[Zcand][:,SlagrevZζnow...] .+ eSt,
                                                  prior_Q[Zcand][SlagrevZζnow...].p1_now,
                                                  prior_Q[Zcand][SlagrevZζnow...].α,
                                                  prior_Q[Zcand][SlagrevZζnow...].μ,
                                                  prior_Q[Zcand][SlagrevZζnow...].M  )

              lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
            end
      end

      w = exp.( lw - maximum(lw) )
      newindx = StatsBase.sample(Weights( w ))
      Zζ_out[i] = copy(newindx)

      ## update running quantities
      Znew, ζnew = copy(λ_indx.Zζindx[newindx,1]), copy(λ_indx.Zζindx[newindx,2])
      N_now[Znew][:,Slagrev_now[ λ_indx.indxs[Znew][ζnew] ]...] += eSt
      llikmarg = copy(llikmarg_cand[newindx])
  end

  Zζ_out
end



"""
    MetropIndep_ΛλZζ(S, lΛ_old, lλ_old, Zζ_old,
        prior_Λ, prior_λ, prior_Q,
        λ_indx,
        TT, R, M, K)

    Independence Metropolis step for Λ, λ, Z, and ζ.
"""
function MetropIndep_ΛλZζ(S::Vector{Int}, lΛ_old::Vector{Float64},
    lλ_old::Vector{Vector{Float64}}, Zζ_old::Vector{Int},
    prior_Λ::Union{Vector{Float64}, SparseDirMixPrior},
    prior_λ::Union{Vector{Vector{Float64}}, Vector{SparseDirMixPrior}, Vector{SparseSBPrior}, Vector{SparseSBPriorP}},
    prior_Q::Union{Vector{Array{SparseSBPrior}}, Vector{Array{SparseSBPriorP}}},
    λ_indx::λindxMMTD,
    TT::Int, R::Int, M::Int, K::Int)

  Q_SBM_flag = typeof(prior_Q) == Vector{Array{SparseSBPrior}}
  Q_SBMp_flag = typeof(prior_Q) == Vector{Array{SparseSBPriorP}}

  prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

  if typeof(prior_Λ)==Vector{Float64}
      lΛ_cand = BayesInference.rDirichlet(prior_Λ, true)
    elseif typeof(prior_Λ)==SparseDirMixPrior
      lΛ_cand = rSparseDirMix(prior_Λ.α, prior_Λ.β, true)
  end

  if typeof(prior_λ)==Vector{Vector{Float64}}
      lλ_cand = [ BayesInference.rDirichlet(prior_λ[m], true) for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseDirMixPrior}
      lλ_cand = [ rSparseDirMix(prior_λ[m].α, prior_λ[m].β, true) for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseSBPrior}
      lλ_cand = [ rpost_sparseStickBreak(zeros(Int64, λ_indx.lens[m]), prior_λ[m].p1, prior_λ[m].α, prior_λ[m].μ, prior_λ[m].M, true)[1] for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseSBPriorP}
      lλ_cand = [ rpost_sparseStickBreak(zeros(Int64, λ_indx.lens[m]), prior_λ[m].p1_now, prior_λ[m].α, prior_λ[m].μ, prior_λ[m].M, true)[1] for m in 1:M ]
  end

  lΛλ_cand = [ lΛ_cand[λ_indx.Zζindx[j,1]] + lλ_cand[λ_indx.Zζindx[j,1]][λ_indx.Zζindx[j,2]] for j in 1:λ_indx.nZζ ]
  Λλ_cand = exp.(lΛλ_cand)

  Zζ_cand = [ StatsBase.sample( Weights( Λλ_cand ) ) for i in 1:(TT-R) ]
  Zandζ_cand = ZζtoZandζ(Zζ_cand, λ_indx)
  Zandζ_old = ZζtoZandζ(Zζ_old, λ_indx)

  N_cand = counttrans_mmtd(S, TT, Zandζ_cand, λ_indx, R, M, K)
  N_old = counttrans_mmtd(S, TT, Zandζ_old, λ_indx, R, M, K)

  N_cand_mats = [ reshape(N_cand[m], K, K^m) for m in 1:M ]
  N_old_mats = [ reshape(N_old[m], K, K^m) for m in 1:M ]

  llikmarg_cand = 0.0
  llikmarg_old = 0.0

  if Q_SBM_flag

      for m in 1:M
          for ℓ in 1:(K^m)
              llikmarg_cand += logSBMmarginal( N_cand_mats[m][:,ℓ],
                                          prior_Q_vec[m][ℓ].p1,
                                          prior_Q_vec[m][ℓ].α,
                                          prior_Q_vec[m][ℓ].μ,
                                          prior_Q_vec[m][ℓ].M )

              llikmarg_old += logSBMmarginal( N_old_mats[m][:,ℓ],
                                          prior_Q_vec[m][ℓ].p1,
                                          prior_Q_vec[m][ℓ].α,
                                          prior_Q_vec[m][ℓ].μ,
                                          prior_Q_vec[m][ℓ].M )
          end
      end

    elseif Q_SBMp_flag

      for m in 1:M
          for ℓ in 1:(K^m)
              llikmarg_cand += logSBMmarginal( N_cand_mats[m][:,ℓ],
                                  prior_Q_vec[m][ℓ].p1_now,
                                  prior_Q_vec[m][ℓ].α,
                                  prior_Q_vec[m][ℓ].μ,
                                  prior_Q_vec[m][ℓ].M )
              llikmarg_old += logSBMmarginal( N_old_mats[m][:,ℓ],
                                  prior_Q_vec[m][ℓ].p1_now,
                                  prior_Q_vec[m][ℓ].α,
                                  prior_Q_vec[m][ℓ].μ,
                                  prior_Q_vec[m][ℓ].M )
          end
      end

  end

  lu = log(rand())
  if lu < (llikmarg_cand - llikmarg_old)
      lΛ_out = lΛ_cand
      lλ_out = lλ_cand
      Zζ_out = Zζ_cand
  else
      lΛ_out = lΛ_old
      lλ_out = lλ_old
      Zζ_out = Zζ_old
  end

  (lΛ_out, lλ_out, Zζ_out)
end
function MetropIndep_ΛλZζ(S::Vector{Int}, lΛ_old::Vector{Float64},
    lλ_old::Vector{Vector{Float64}}, Zζ_old::Vector{Int},
    prior_Λ::Union{Vector{Float64}, SparseDirMixPrior},
    prior_λ::Union{Vector{Vector{Float64}}, Vector{SparseDirMixPrior}, Vector{SparseSBPrior}, Vector{SparseSBPriorP}},
    prior_Q::Vector{Array{Float64}},
    λ_indx::λindxMMTD,
    TT::Int, R::Int, M::Int, K::Int)

  if typeof(prior_Λ)==Vector{Float64}
      lΛ_cand = BayesInference.rDirichlet(prior_Λ, true)
    elseif typeof(prior_Λ)==SparseDirMixPrior
      lΛ_cand = rSparseDirMix(prior_Λ.α, prior_Λ.β, true)
  end

  if typeof(prior_λ)==Vector{Vector{Float64}}
      lλ_cand = [ BayesInference.rDirichlet(prior_λ[m], true) for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseDirMixPrior}
      lλ_cand = [ rSparseDirMix(prior_λ[m].α, prior_λ[m].β, true) for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseSBPrior}
      lλ_cand = [ rpost_sparseStickBreak(zeros(Int64, λ_indx.lens[m]), prior_λ[m].p1, prior_λ[m].α, prior_λ[m].μ, prior_λ[m].M, true)[1] for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseSBPriorP}
      lλ_cand = [ rpost_sparseStickBreak(zeros(Int64, λ_indx.lens[m]), prior_λ[m].p1_now, prior_λ[m].α, prior_λ[m].μ, prior_λ[m].M, true)[1] for m in 1:M ]
  end

  lΛλ_cand = [ lΛ_cand[λ_indx.Zζindx[j,1]] + lλ_cand[λ_indx.Zζindx[j,1]][λ_indx.Zζindx[j,2]] for j in 1:λ_indx.nZζ ]
  Λλ_cand = exp.(lΛλ_cand)

  Zζ_cand = [ StatsBase.sample( Weights( Λλ_cand ) ) for i in 1:(TT-R) ]
  Zandζ_cand = ZζtoZandζ(Zζ_cand, λ_indx)
  Zandζ_old = ZζtoZandζ(Zζ_old, λ_indx)

  N_cand = counttrans_mmtd(S, TT, Zandζ_cand, λ_indx, R, M, K)
  N_old = counttrans_mmtd(S, TT, Zandζ_old, λ_indx, R, M, K)

  N_cand_mats = [ reshape(N_cand[m], K, K^m) for m in 1:M ]
  N_old_mats = [ reshape(N_old[m], K, K^m) for m in 1:M ]

  prior_Q_mats = [ reshape(prior_Q[m], K, K^m) for m in 1:M ]

  llikmarg_cand = 0.0
  llikmarg_old = 0.0

  for m in 1:M
      for ℓ in 1:(K^m)
          α_now = copy(prior_Q_mats[m][:,ℓ])
          llikmarg_cand += lmvbeta( N_cand_mats[m][:,ℓ] + α_now )
          llikmarg_old += lmvbeta( N_old_mats[m][:,ℓ] + α_now )
      end
  end

  lu = log(rand())
  if lu < (llikmarg_cand - llikmarg_old)
      lΛ_out = lΛ_cand
      lλ_out = lλ_cand
      Zζ_out = Zζ_cand
  else
      lΛ_out = lΛ_old
      lλ_out = lλ_old
      Zζ_out = Zζ_old
  end

  (lΛ_out, lλ_out, Zζ_out)
end




"""
mcmc_mmtd!(model, n_keep[, save=true, report_filename="out_progress.txt",
thin=1, report_freq=500, monitor_indx=[1]])
"""
function mcmc_mmtd!(model::ModMMTD, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=10000;
    monitor_indx::Vector{Int}=[1])

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(now())
    for $(n_keep * thin) iterations.\n")

        if save
            monitor_len = length(monitor_indx)
            sims = PostSimsMMTD( Matrix{Float64}(n_keep, model.M), # Λ
                [ Matrix{Float64}(n_keep, model.λ_indx.lens[m]) for m in 1:model.M ], # λ
                [ Matrix{Float64}(n_keep, model.K^(m+1)) for m in 1:model.M ], # Q
                Matrix{Int}(n_keep, monitor_len), # Zζ
                Matrix{Float64}(n_keep, model.M), # p1λ
                [ Matrix{Float64}(n_keep, model.K^m) for m in 1:model.M ] #= p1Q =# )
        end

        ## flags
        λSBMp_flag = typeof(model.prior.λ) == Vector{BayesInference.SparseSBPriorP}
        λSBMfull_flag = typeof(model.prior.λ) == Vector{BayesInference.SparseSBPriorFull}
        QSBMp_flag = typeof(model.prior.Q) == Vector{Array{BayesInference.SparseSBPriorP}}
        QSBMfull_flag = typeof(model.prior.Q) == Vector{Array{BayesInference.SparseSBPriorFull}}

        ## sampling
        for i in 1:n_keep
            for j in 1:thin

                jmpstart = (model.iter % jmpstart_iter == 0)

                if jmpstart

                    model.state.lΛ, model.state.lλ, model.state.Zζ = MetropIndep_ΛλZζ(model.S,
                        model.state.lΛ, model.state.lλ, model.state.Zζ,
                        model.prior.Λ, model.prior.λ, model.prior.Q,
                        model.λ_indx,
                        model.TT, model.R, model.M, model.K)

                    Zandζnow = ZζtoZandζ(model.state.Zζ, model.λ_indx)

                else

                    model.state.Zζ = rpost_Zζ_marg(model.S, model.state.Zζ,
                        model.λ_indx,
                        model.prior.Q,
                        model.state.lΛ, model.state.lλ,
                        model.TT, model.R, model.M, model.K)

                    Zandζnow = ZζtoZandζ(model.state.Zζ, model.λ_indx)

                    model.state.lΛ = rpost_lΛ_mmtd(model.prior.Λ, Zandζnow[:,1], model.M)

                    if λSBMp_flag || λSBMfull_flag
                        model.state.lλ = rpost_lλ_mmtd!(model.prior.λ, Zandζnow,
                            model.λ_indx.lens, model.M)
                    else
                        model.state.lλ = rpost_lλ_mmtd(model.prior.λ, Zandζnow,
                            model.λ_indx.lens, model.M)
                    end

                end


                if QSBMp_flag || QSBMfull_flag
                    model.state.lQ = rpost_lQ_mmtd!(model.S, model.TT, model.prior.Q,
                        Zandζnow, model.λ_indx, model.R, model.M, model.K)
                else
                    model.state.lQ = rpost_lQ_mmtd(model.S, model.TT, model.prior.Q,
                        Zandζnow, model.λ_indx, model.R, model.M, model.K)
                end

                model.iter += 1
                if model.iter % report_freq == 0
                    write(report_file, "Iter $(model.iter) at $(now())\n")
                end
            end

            if save
                @inbounds sims.Λ[i,:] = exp.( model.state.lΛ )
                @inbounds sims.Zζ[i,:] = copy(model.state.Zζ[monitor_indx])
                for m in 1:model.M
                    @inbounds sims.λ[m][i,:] = exp.( model.state.lλ[m] )
                    @inbounds sims.Q[m][i,:] = exp.( vec( model.state.lQ[m] ) )
                    if λSBMp_flag || λSBMfull_flag
                        sims.p1λ[i,m] = copy(model.prior.λ[m].p1_now)
                    end
                    if QSBMp_flag || QSBMfull_flag
                        pQvec = vec( model.prior.lQ[m] )
                        for kk in 1:model.K^m
                            sims.p1Q[m][i,kk] = copy( pQvec[kk].p1_now )
                        end
                    end
                end
            end
        end

        close(report_file)

        if save
            return sims
        else
            return model.iter
        end

    end
