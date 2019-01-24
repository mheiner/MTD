# mmtd_inference.jl

export ParamsMMTD, PriorMMTD, ModMMTD, λindxMMTD,
  build_λ_indx, ZζtoZandζ, sim_mmtd, symmetricDirPrior_mmtd, transTensor_mmtd,
  counttrans_mmtd,
  mcmc_mmtd!, timemod!,
  bfact_MC, forecDist_MMTD; # remove the inner functions after testing

mutable struct ParamsMMTD
  lΛ::Vector{Float64}
  lλ::Vector{Vector{Float64}}
  Zζ::Vector{Int} # will be length TT - R, mapped through ZζtoZandζ()
  lQ::Vector{<:Array{Float64}} # organized so first index is now and lag 1 is the next index
end

mutable struct PriorMMTD
  Λ::Union{Vector{Float64}, SparseDirMix, SBMprior}
  λ::Union{Vector{Vector{Float64}}, Vector{SparseDirMix}, Vector{SBMprior}}
  Q::Union{Vector{<:Array{Float64}}, Vector{<:Array{SparseDirMix}}, Vector{<:Array{SparseSBMprior}}}
end

mutable struct λindxMMTD
    indxs::Vector{Array{Array{Int64,1},1}}
    lens::Vector{Int64}
    Zζindx::Matrix{Int64}
    nZζ::Int64
end

mutable struct ModMMTD
  R::Int # maximal order
  M::Int # largest order considered
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMMTD
  state::ParamsMMTD
  λ_indx::λindxMMTD
  iter::Int

  ModMMTD(R, M, K, TT, S, prior, state, λ_indx) = new(R, M, K, TT, S, prior, state, λ_indx, 0)
end

mutable struct PostSimsMMTD
  Λ::Matrix{Float64}
  λ::Array{Matrix{Float64}}
  Q::Array{Matrix{Float64}} # stored matricized
  Zζ::Matrix{Int} # index 1 is iteration, index 2 is time, entries are Zζ

  PostSimsMMTD(Λ, λ, Q, Zζ) = new(Λ, λ, Q, Zζ)
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
  Λ::Array{Float64}, λ::Array{Vector{Float64}}, Q::Vector{<:Array{Float64}}, Xtras::Bool=false)

  Nsim = nburn + TT

  Z = [ StatsBase.sample(Weights(Λ)) for i in 1:(Nsim-R) ]
  ζ = [ StatsBase.sample(Weights(λ[m])) for i in 1:(Nsim-R), m in 1:M ]

  S = Vector{Int}(undef, Nsim)
  S[1:R] = StatsBase.sample(1:K, R)

  if Xtras
      Pvecs = Matrix{Float64}(undef, Nsim, K)
      # Slagrevs = Matrix{Int}(undef, Nsim, R)
      Pvecs[1:R, :] = 0.0
      # Slagrevs[1:R, :] = 0
  end

  for tt in (R+1):(Nsim)
    i = tt - R
    Slagrev_now = deepcopy(S[range(tt-1, step=-1, length=R)])
    pvec = deepcopy( Q[Z[i]][:, Slagrev_now[λ_indx.indxs[Z[i]][ζ[i,Z[i]]]]...] )
    if Xtras
        Pvecs[tt,:] = deepcopy(pvec)
        # Slagrevs[tt,:] = deepcopy(Slagrev_now)
    end
    S[tt] = StatsBase.sample(Weights( pvec ))
  end

  if Xtras
      S[(nburn+1):(Nsim)], Z[(nburn+1):(Nsim-R)], ζ[(nburn+1):(Nsim-R),:], Pvecs[(nburn+1):(Nsim),:]#, Slagrevs[(nburn+1):(Nsim),:]
  else
      S[(nburn+1):(Nsim)], Z[(nburn+1):(Nsim-R)], ζ[(nburn+1):(Nsim-R),:]
  end
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
  Λ::Vector{Float64}, λ::Vector{Vector{Float64}}, Q::Vector{<:Array{Float64}})

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
  reshape(Ωmat, fill(K,R+1)...)
end


"""
rpost_lΛ_mmtd(α0_Λ, Z, M)
"""
function rpost_lΛ_mmtd(α0_Λ::Vector{Float64}, Z::Vector{Int}, M::Int)
    Nz = StatsBase.counts(Z, 1:M)
    α1_Λ = α0_Λ .+ Nz
    SparseProbVec.rDirichlet(α1_Λ, logout=true)
end
function rpost_lΛ_mmtd(prior::SparseDirMix, Z::Vector{Int}, M::Int)
    Nz = StatsBase.counts(Z, 1:M)
    α1_Λ = prior.α .+ Nz
    d = SparseProbVec.SparseDirMix(α1_Λ, prior.β)
    SparseProbVec.rand(d, logout=true)
end
function rpost_lΛ_mmtd(prior::SBMprior, Z::Vector{Int}, M::Int)

    Nz = StatsBase.counts(Z, 1:M)
    post_lΛ = SparseProbVec.SBM_multinom_post(prior, Nz)
    SparseProbVec.rand(post_lΛ, logout=true)
end

"""
    rpost_lλ_mmtd(α0_λ, ζ, λ_lens, M)
"""
function rpost_lλ_mmtd(α0_λ::Vector{Vector{Float64}}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(undef, λ_lens[m]) for m in 1:M ]
  Z = deepcopy(Zandζ[:,1])
  ζ = deepcopy(Zandζ[:,2])

  for m in 1:M
    Zmindx = findall(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    α1_λ = α0_λ[m] .+ Nζ
    lλ_out[m] = SparseProbVec.rDirichlet(α1_λ, logout=true)
  end

  lλ_out
end
function rpost_lλ_mmtd(prior::Vector{SparseDirMix}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(undef, λ_lens[m]) for m in 1:M ]
  Z = deepcopy(Zandζ[:,1])
  ζ = deepcopy(Zandζ[:,2])

  for m in 1:M
    Zmindx = findall(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    α1_λ = prior[m].α .+ Nζ
    d = SparseDirMix(α1_λ, prior[m].β)
    lλ_out[m] = SparseProbVec.rand(d, logout=true)
  end

  lλ_out
end
function rpost_lλ_mmtd(prior::Vector{SBMprior}, Zandζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(undef, λ_lens[m]) for m in 1:M ]
  Z = deepcopy(Zandζ[:,1])
  ζ = deepcopy(Zandζ[:,2])

  for m in 1:M
    Zmindx = findall(Z .== m)
    Nζ = StatsBase.counts(ζ[Zmindx], 1:λ_lens[m])
    post_lλ = SparseProbVec.SBM_multinom_post(prior[m], Nζ)
    lλ_out[m] = SparseProbVec.rand(post_lλ, logout=true)
  end

  lλ_out
end


"""
    counttrans_mmtd(S::Vector{Int}, TT::Int, Zandζ::Matrix{Int},
        λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

    Indexing on ouput is [t, t-1, t-2, etc.]

    ### Example
    ```julia
    R = 2
    M = 2
    K = 3
    TT = 12
    S = [1,2,1,3,3,1,2,1,3,2,1,1]
    Zandζ = [1 2; 1 2; 1 2; 1 1; 2 1; 1 2; 2 1; 2 1; 2 1; 1 1]
    λ_indx = build_λ_indx(R, M)
    counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)
    ```
"""
function counttrans_mmtd(S::Vector{Int}, TT::Int, Zandζ::Matrix{Int},
  λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  Z = deepcopy(Zandζ[:,1])
  ζ = deepcopy(Zandζ[:,2])

  ## initialize
  N_out = [ zeros(Int, fill(K, m+1)...) for m in 1:M ]

  ## pass through data and add counts
  for tt in (R+1):(TT)
    Z_now = Z[tt-R]
    Slagrev_now = S[range(tt-1, step=-1, length=R)]
    N_out[Z_now][append!( [deepcopy(S[tt])], deepcopy(Slagrev_now[ λ_indx.indxs[Z_now][ζ[tt-R]] ]) )...] += 1
  end

  N_out
end


"""
    rpost_lQ_mmtd(S, TT, prior, Z, ζ, λ_indx, R, M, K)
"""
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int, prior::Vector{<:Array{Float64}},
  Zandζ::Matrix{Int},
  λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  ## initialize
  α0_Q = deepcopy(prior) # vector of arrays
  lQ_mats = [ Matrix{Float64}(undef, K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], fill(K, m+1)...) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K) # vector of arrays

  for m in 1:M
    ncol = K^m
    α1_Q = α0_Q[m] .+ N[m]
    α1_Q_mat = reshape(α1_Q, (K, ncol))
    for j in 1:ncol
      lQ_mats[m][:,j] = SparseProbVec.rDirichlet(α1_Q_mat[:,j], logout=true)
    end
    lQ_out[m] = reshape(lQ_mats[m], fill(K, m+1)...)
  end

  lQ_out
end
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int,
  prior::Vector{<:Array{SparseDirMix}},
  Zandζ::Matrix{Int},
  λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  prior_vec = [ reshape(prior[m], K^m ) for m in 1:M ]

  ## initialize
  lQ_mats = [ Matrix{Float64}(undef, K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], fill(K, m+1)...) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)

  for m in 1:M
    ncol = K^m
    Nmat = reshape(N[m], (K, ncol))
    for j in 1:ncol
      α1 = prior_vec[m][j].α .+ Nmat[:,j]
      d = SparseProbVec.SparseDirMix(α1, prior_vec[m][j].β)
      lQ_mats[m][:,j] = SparseProbVec.rand(d, true)
    end
    lQ_out[m] = reshape(lQ_mats[m], fill(K, m+1)...)
  end

  lQ_out
end
function rpost_lQ_mmtd(S::Vector{Int}, TT::Int, prior::Vector{<:Array{SBMprior}},
    Zandζ::Matrix{Int}, λ_indx::λindxMMTD, R::Int, M::Int, K::Int)

  prior_vec = [ reshape(prior[m], K^m ) for m in 1:M ]

  ## initialize
  lQ_mats = [ Matrix{Float64}(undef, K, K^m) for m in 1:M ]
  lQ_out = [ reshape(lQ_mats[m], fill(K, m+1)...) for m in 1:M ]

  N = counttrans_mmtd(S, TT, Zandζ, λ_indx, R, M, K)

  for m in 1:M
    ncol = K^m
    Nmat = reshape(N[m], (K, ncol))
    for j in 1:ncol
        d = SparseProbVec.SBM_multinom_post(prior_vec[m][j], Nmat[:,j])
        lQ_mats[m][:,j] = SparseProbVec.rand(d, logout=true)
    end
    lQ_out[m] = reshape(lQ_mats[m], fill(K, m+1)...)
  end

  lQ_out
end


"""
    rpost_Zζ_marg(S, Zζ_old, λ_indx, prior_Q, lΛ, lλ, TT, R, K)

    Full conditinal updates for Zζ marginalizing over Q
"""
function rpost_Zζ_marg(S::Vector{Int}, Zζ_old::Vector{Int}, λ_indx::λindxMMTD,
    prior_Q::Vector{<:Array{Float64}},
    lΛ::Vector{Float64},
    lλ::Vector{Vector{Float64}},
    TT::Int, R::Int, M::Int, K::Int)

  α0_Q = deepcopy(prior_Q) # vector of arrays
  # α0_Qarrays = [ reshape(α0_Q[m], fill(K, m+1)...) for m in 1:M ]

  Zζ_out = deepcopy(Zζ_old)
  Zandζ_now = ZζtoZandζ(Zζ_out, λ_indx) # won't get updated

  ## calculate current log marginal likelihood
  N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
  # N_now_mats = [ reshape(N_now[m], K, K^m) for m in 1:M ]
  α1_Q = α0_Q .+ N_now # gets re-used and updated
  α1_Q_mats = [ reshape(α1_Q[m], K, K^m) for m in 1:M ] # used only for initial calculation

  llikmarg = 0.0
  for m in 1:M
      for kk in 1:(K^m)
          llikmarg += SparseProbVec.lmvbeta( α1_Q_mats[m][:,kk] )
      end
  end

  ## for each time point, modify llikmarg accordingly
  for i in 1:(TT-R)  # i indexes Zζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, step=-1, length=R)]

    # Zold, ζold = deepcopy(Zandζ_out[i,1]), deepcopy(Zandζ_out[i,2])
    Zold, ζold = deepcopy(λ_indx.Zζindx[Zζ_out[i],1]), deepcopy(λ_indx.Zζindx[Zζ_out[i],2])
    eSt = [ 1.0 * (kk==S[tt]) for kk in 1:K ]

    # remove S_t from llikmarg
    SlagrevZζnow = deepcopy(Slagrev_now[ λ_indx.indxs[Zold][ζold] ])
    llikmarg -= SparseProbVec.lmvbeta( α1_Q[Zold][:,SlagrevZζnow...] )
    α1_Q[Zold][:,SlagrevZζnow...] -= eSt
    llikmarg += SparseProbVec.lmvbeta( α1_Q[Zold][:,SlagrevZζnow...] )

    # calculate llikmarg and update probability under each possible Zζ
    llikmarg_cand = fill(llikmarg, λ_indx.nZζ)
    lw = zeros(Float64, λ_indx.nZζ)

    for ℓ in 1:λ_indx.nZζ
        Zcand, ζcand = deepcopy(λ_indx.Zζindx[ℓ,1]), deepcopy(λ_indx.Zζindx[ℓ,2])
        SlagrevZζnow = deepcopy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

        llikmarg_cand[ℓ] -= SparseProbVec.lmvbeta( α1_Q[Zcand][:,SlagrevZζnow...] )
        llikmarg_cand[ℓ] += SparseProbVec.lmvbeta( α1_Q[Zcand][:,SlagrevZζnow...] .+ eSt )

        lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
    end

    w = exp.( lw .- maximum(lw) )
    newindx = StatsBase.sample(Weights( w ))
    Zζ_out[i] = deepcopy(newindx)

    ## update running quantities
    Znew, ζnew = deepcopy(λ_indx.Zζindx[newindx,1]), deepcopy(λ_indx.Zζindx[newindx,2])
    α1_Q[Znew][:,Slagrev_now[ λ_indx.indxs[Znew][ζnew] ]...] += eSt
    llikmarg = deepcopy(llikmarg_cand[newindx])

  end

  Zζ_out
end
function rpost_Zζ_marg(S::Vector{Int}, Zζ_old::Vector{Int},
    λ_indx::λindxMMTD,
    prior_Q::Vector{<:Array{SparseDirMix}},
    lΛ::Vector{Float64},
    lλ::Vector{Vector{Float64}},
    TT::Int, R::Int, M::Int, K::Int)

  Zζ_out = deepcopy(Zζ_old)
  Zandζ_now = ZζtoZandζ(Zζ_out, λ_indx) # won't get updated
  prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

  ## calculate current log marginal likelihood
  N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
  N_now_mats = [ reshape(N_now[m], K, K^m) for m in 1:M ] # Ns won't be updated, only llikmarg

  llikmarg = 0.0
  for m in 1:M
      for ℓ in 1:(K^m)
          llikmarg += SparseProbVec.logMarginal(prior_Q_vec[m][ℓ], N_now_mats[m][:,ℓ])
      end
  end

  ## for each time point, modify llikmarg accordingly
  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, step=-1, length=R)]

    Zold, ζold = deepcopy(λ_indx.Zζindx[Zζ_out[i],1]), deepcopy(λ_indx.Zζindx[Zζ_out[i],2])
    eSt = [ 1*(kk==S[tt]) for kk in 1:K ]

    # remove S_t from llikmarg
    SlagrevZζnow = deepcopy(Slagrev_now[ λ_indx.indxs[Zold][ζold] ])
    llikmarg -= SparseProbVec.logMarginal( prior_Q[Zold][SlagrevZζnow...],
                                N_now[Zold][:,SlagrevZζnow...] )
            # prior for Q indexed with Slagrev_now[ λ_indx.indxs[Zold][ζold] ]...
    N_now[Zold][:,SlagrevZζnow...] -= eSt
    llikmarg += SparseProbVec.logMarginal( prior_Q[Zold][SlagrevZζnow...],
                                N_now[Zold][:,SlagrevZζnow...] )

    # calculate llikmarg and update probability under each possible Zζ
    llikmarg_cand = fill(llikmarg, λ_indx.nZζ)
    lw = zeros(Float64, λ_indx.nZζ)

    for ℓ in 1:λ_indx.nZζ
        Zcand, ζcand = deepcopy(λ_indx.Zζindx[ℓ,1]), deepcopy(λ_indx.Zζindx[ℓ,2])
        SlagrevZζnow = deepcopy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

        llikmarg_cand[ℓ] -= SparseProbVec.logMarginal( prior_Q[Zcand][SlagrevZζnow],
                                    N_now[Zcand][:,SlagrevZζnow...] )
        llikmarg_cand[ℓ] += SparseProbVec.logMarginal( prior_Q[Zcand][SlagrevZζnow],
                                    N_now[Zcand][:,SlagrevZζnow...] .+ eSt )

        lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
    end

    w = exp.( lw .- maximum(lw) )
    newindx = StatsBase.sample(Weights( w ))
    Zζ_out[i] = deepcopy(newindx)

    ## update running quantities
    Znew, ζnew = deepcopy(λ_indx.Zζindx[newindx,1]), deepcopy(λ_indx.Zζindx[newindx,2])
    N_now[Znew][:,Slagrev_now[ λ_indx.indxs[Znew][ζnew] ]...] += eSt
    llikmarg = deepcopy(llikmarg_cand[newindx])

  end

  Zζ_out
end
function rpost_Zζ_marg(S::Vector{Int}, Zζ_old::Vector{Int},
    λ_indx::λindxMMTD,
    prior_Q::Vector{<:Array{SBMprior}},
    lΛ::Vector{Float64},
    lλ::Vector{Vector{Float64}},
    TT::Int, R::Int, M::Int, K::Int)

  Zζ_out = deepcopy(Zζ_old)
  Zandζ_now = ZζtoZandζ(Zζ_out, λ_indx) # won't get updated
  prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

  ## calculate current log marginal likelihood
  N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
  N_now_mats = [ reshape(N_now[m], K, K^m) for m in 1:M ] # Ns won't be updated, only llikmarg

  llikmarg = 0.0
  for m in 1:M
      for ℓ in 1:(K^m)
          llikmarg += SparseProbVec.logMarginal( prior_Q_vec[m][ℓ], N_now_mats[m][:,ℓ] )
      end
  end

  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
      tt = i + R
      Slagrev_now = S[range(tt-1, step=-1, length=R)]

      Zold, ζold = deepcopy(λ_indx.Zζindx[Zζ_out[i],1]), deepcopy(λ_indx.Zζindx[Zζ_out[i],2])
      eSt = [ 1*(kk==S[tt]) for kk in 1:K ]

      # remove S_t from llikmarg
      SlagrevZζnow = deepcopy(Slagrev_now[ λ_indx.indxs[Zold][ζold] ])

      llikmarg -= SparseProbVec.logMarginal( prior_Q[Zold][SlagrevZζnow...],
                                  N_now[Zold][:,SlagrevZζnow...] )
      # prior for Q indexed with Slagrev_now[ λ_indx.indxs[Zold][ζold] ]...
      N_now[Zold][:,SlagrevZζnow...] -= eSt
      llikmarg += SparseProbVec.logMarginal( prior_Q[Zold][SlagrevZζnow...],
                                  N_now[Zold][:,SlagrevZζnow...] )

      # calculate llikmarg and update probability under each possible Zζ
      llikmarg_cand = fill(llikmarg, λ_indx.nZζ)
      lw = zeros(Float64, λ_indx.nZζ)

      for ℓ in 1:λ_indx.nZζ
          Zcand, ζcand = deepcopy(λ_indx.Zζindx[ℓ,1]), deepcopy(λ_indx.Zζindx[ℓ,2])
          SlagrevZζnow = deepcopy(Slagrev_now[ λ_indx.indxs[Zcand][ζcand] ])

          llikmarg_cand[ℓ] -= SparseProbVec.logMarginal( prior_Q[Zcand][SlagrevZζnow...],
                                    N_now[Zcand][:,SlagrevZζnow...] )
          llikmarg_cand[ℓ] += SparseProbVec.logMarginal( prior_Q[Zcand][SlagrevZζnow...],
                                    N_now[Zcand][:,SlagrevZζnow...] .+ eSt )

          lw[ℓ] = llikmarg_cand[ℓ] + lΛ[Zcand] + lλ[Zcand][ζcand]
      end

      w = exp.( lw .- maximum(lw) )
      newindx = StatsBase.sample(Weights( w ))
      Zζ_out[i] = deepcopy(newindx)

      ## update running quantities
      Znew, ζnew = deepcopy(λ_indx.Zζindx[newindx,1]), deepcopy(λ_indx.Zζindx[newindx,2])
      N_now[Znew][:,Slagrev_now[ λ_indx.indxs[Znew][ζnew] ]...] += eSt
      llikmarg = deepcopy(llikmarg_cand[newindx])
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
    prior_Λ::Union{Vector{Float64}, SparseDirMix, SBMprior},
    prior_λ::Union{Vector{Vector{Float64}}, Vector{SparseDirMix}, Vector{SBMprior}},
    prior_Q::Union{Vector{<:Array{Float64}}, Vector{<:Array{SparseDirMix}}, Vector{<:Array{SBMprior}}},
    λ_indx::λindxMMTD,
    TT::Int, R::Int, M::Int, K::Int)

  prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

  if typeof(prior_Λ) == Vector{Float64}
      lΛ_cand = SparseProbVec.rDirichlet(prior_Λ, logout=true)
    elseif typeof(prior_Λ) <: Union{SparseDirMix, SBMprior}
      lΛ_cand = SparseProbVec.rand(prior_Λ, logout=true)
  end

  if typeof(prior_λ)==Vector{Vector{Float64}}
      lλ_cand = [ SparseProbVec.rDirichlet(prior_λ[m], logout=true) for m in 1:M ]
    elseif typeof(prior_λ)==Vector{SparseDirMixPrior}
      lλ_cand = [ rand(prior_λ[m], logout=true) for m in 1:M ]
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

  if typeof(prior_Q) <: Vector{<:Array{Float64}}

      for m in 1:M
          for ℓ in 1:(K^m)
            α_now = deepcopy(prior_Q_mats[m][:,ℓ])
            llikmarg_cand += SparseProbVec.lmvbeta( N_cand_mats[m][:,ℓ] + α_now )
            llikmarg_old += SparseProbVec.lmvbeta( N_old_mats[m][:,ℓ] + α_now )
          end
      end

  elseif typeof(prior_Q) <: Union{Vector{<:Array{SparseDirMix}}, Vector{<:Array{SBMprior}}}

      for m in 1:M
          for ℓ in 1:(K^m)
              llikmarg_cand += SparseProbVec.logMarginal( prior_Q_vec[m][ℓ],
                                                N_cand_mats[m][:,ℓ] )
              llikmarg_old += SparseProbVec.logMarginal( prior_Q_vec[m][ℓ],
                                                N_old_mats[m][:,ℓ] )
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



"""
mcmc_mmtd!(model, n_keep[, save=true, report_filename="out_progress.txt",
thin=1, jmpstart_iter=25, report_freq=500, monitor_indx=[1]])
"""
function mcmc_mmtd!(model::ModMMTD, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=10000;
    monitor_indx::Vector{Int}=[1])

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    if save
        monitor_len = length(monitor_indx)
        sims = PostSimsMMTD( Matrix{Float64}(undef, n_keep, model.M), # Λ
        [ Matrix{Float64}(undef, n_keep, model.λ_indx.lens[m]) for m in 1:model.M ], # λ
        [ Matrix{Float64}(undef, n_keep, model.K^(m+1)) for m in 1:model.M ], # Q
        Matrix{Int}(undef, n_keep, monitor_len) #= Zζ =# )
    end

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

                model.state.lλ = rpost_lλ_mmtd(model.prior.λ, Zandζnow,
                    model.λ_indx.lens, model.M)

            end

            model.state.lQ = rpost_lQ_mmtd(model.S, model.TT, model.prior.Q,
                Zandζnow, model.λ_indx, model.R, model.M, model.K)

            model.iter += 1
            if model.iter % report_freq == 0
                write(report_file, "Iter $(model.iter) at $(Dates.now())\n")
            end
        end

        if save
            @inbounds sims.Λ[i,:] = exp.( model.state.lΛ )
            @inbounds sims.Zζ[i,:] = deepcopy(model.state.Zζ[monitor_indx])
            for m in 1:model.M
                @inbounds sims.λ[m][i,:] = exp.( model.state.lλ[m] )
                @inbounds sims.Q[m][i,:] = exp.( vec( model.state.lQ[m] ) )
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



## timing for benchmarks
function timemod!(n::Int64, model::ModMMTD, niter::Int, outfilename::String)
    outfile = open(outfilename, "a+")
    write(outfile, "timing for $(niter) iterations each:\n")
    for i in 1:n
        tinfo = @timed mcmc_mmtd!(model, niter, false, outfilename)
        write(outfile, "trial $(i), elapsed: $(tinfo[2]) seconds, allocation: $(tinfo[3]/1.0e6) Megabytes\n")
    end
    close(outfile)
end



"""
Bayes factors for high order Markov chains

    The report options are:
        "lag1" : all factors reported with respect to the first order lag one model.
        "max" : all factors reported with respect to the maximum.
        "recipmax" : (default) all factors reported with respect to maximum (with maximum as numerator).

### Example
```julia
R = 4
M = 2
K = 2
λ_indx = build_λ_indx(R, M)
TT = 200
## simulate a chain with order 2, lags 1 and 3 important
S = sim_mmtd(TT, 100, R, M, K, λ_indx, # non-sparse transitions
    [ 0.0, 1.0 ],
    [ [1.0, 0.0, 0.0, 0.0 ],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] ],
    [ reshape( [1.0, 0.0, 1.0, 0.0], K, K),
      reshape( [0.8, 0.2, 0.3, 0.7, 0.5, 0.5, 0.1, 0.9], K, K, K ) ] )[1]
S = sim_mmtd(TT, 100, R, M, K, λ_indx, # sparse transitions
    [ 0.0, 1.0 ],
    [ [1.0, 0.0, 0.0, 0.0 ],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] ],
    [ reshape( [1.0, 0.0, 1.0, 0.0], K, K),
      reshape( [0.9, 0.1, 0.2, 0.8, 0.15, 0.85, 0.9, 0.1], K, K, K ) ] )[1]
priQ_type = "Dir"
priQ_type = "SBM"
if priQ_type == "Dir"
    prior_Q = symmetricDirPrior_mmtd(1.0, 1.0, 1.0, R, M, K, λ_indx)[3]
elseif priQ_type == "SBM"
    p1_Q = 0.95
    η_Q = 1.0e3
    μ_Q = 0.9
    M_Q = 4.0
    prior_Q = [ [ SparseSBPrior(η_Q, p1_Q, μ_Q, M_Q) for kk in 1:(K^m) ] for m in 1:M ] # SBM
    prior_Q = [ reshape(prior_Q[m], fill(K, m)... ) for m in 1:M ]
end

bf = bfact_MC(S, R, M, K, prior_Q)
```
"""
function bfact_MC(S::Vector{Int}, R::Int, M::Int, K::Int,
    prior_Q::Vector{<:Array{Float64}}, report="recipmax")

      TT = length(S)
      n = TT - R

      λ_indx = build_λ_indx(R, M)
      llik = zeros(Float64, λ_indx.nZζ)

      for i in 1:λ_indx.nZζ

          Znow = deepcopy(λ_indx.Zζindx[i,1])
          Zandζ_now = ZζtoZandζ( fill(i, n) , λ_indx)

          ## calculate current log marginal likelihood
          N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time

          α0_Q = deepcopy(prior_Q[Znow])
          α0_Qmat = reshape(α0_Q, K, K^Znow)
          α1_Q = α0_Q .+ N_now[Znow]
          α1_Qmat = reshape(α1_Q, K, K^Znow)

          llikmarg = 0.0
          for kk in 1:(K^Znow)
            llikmarg += SparseProbVec.lmvbeta( α1_Qmat[:,kk] ) - SparseProbVec.lmvbeta( α0_Qmat[:,kk] )
          end

          llik[i] = deepcopy(llikmarg)
      end

      if report == "recipmax"
          lbfact = maximum(llik) .- llik
      elseif report == "max"
          lbfact = llik .- maximum(llik)
      elseif report == "lag1"
          lbfact = llik .- llik[1]
      end

      bfact = exp.(lbfact)

      (λ_indx, hcat(λ_indx.Zζindx, bfact))
end
function bfact_MC(S::Vector{Int}, R::Int, M::Int, K::Int,
    prior_Q::Union{Vector{<:Array{SparseDirMix}}, Vector{<:Array{SBMprior}}},
    report="recipmax")

      prior_Q_vec = [ reshape(prior_Q[m], (K^m)) for m in 1:M ]

      TT = length(S)
      n = TT - R

      λ_indx = build_λ_indx(R, M)
      llik = zeros(Float64, λ_indx.nZζ)

      for i in 1:λ_indx.nZζ

          Znow, ζnow = deepcopy(λ_indx.Zζindx[i,1:2])
          Zandζ_now = ZζtoZandζ( fill(i, n) , λ_indx)

          ## calculate current log marginal likelihood
          N_now = counttrans_mmtd(S, TT, Zandζ_now, λ_indx, R, M, K) # vector of arrays, indices in reverse time
          N_now_mat = reshape(N_now[Znow], K, K^Znow)

          llikmarg = 0.0
          for kk in 1:(K^Znow)
            llikmarg += SparseProbVec.logMarginal( prior_Q_vec[Znow][kk],
                            N_now_mat[:,kk] )
          end

          llik[i] = deepcopy(llikmarg)
      end

      if report == "recipmax"
          lbfact = maximum(llik) .- llik
      elseif report == "max"
          lbfact = llik .- maximum(llik)
      elseif report == "lag1"
          lbfact = llik .- llik[1]
      end

      bfact = exp.(lbfact)

      (λ_indx, hcat(λ_indx.Zζindx, bfact))
end


"""
    forecDist_MMTD(Slagrev::Vector{Int}, Λ::Vector{Float64},
      λ::Vector{Vector{Float64}}, Q::Vector{Array{Float64}},
      λ_indx::λindxMMTD)

Computes the forecast distribution given lagged values and parameters.

### Example
```julia
    Λ = [0.0, 1.0]
    λ = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    Q = [ reshape([0.7, 0.3, 0.25, 0.75], (2,2)),
          reshape([0.1, 0.9, 0.4, 0.6, 0.5, 0.5, 0.2, 0.8], (2,2,2)) ]
    λ_indx = build_λ_indx(3,2)
    Slagrev = [1,1,2]
    forecDist_MMTD(Slagrev, Λ, λ, Q, λ_indx)
```
"""
function forecDist_MMTD(Slagrev::Vector{Int}, Λ::Vector{Float64},
  λ::Vector{Vector{Float64}}, Q::Vector{<:Array{Float64}},
  λ_indx::λindxMMTD)
    K = size(Q[1])[1]
    R = size(λ[1])[1]
    M = size(Λ)[1]
    @assert size(Slagrev)[1] == R
    w = zeros(Float64, K)
    for k in 1:K
        for ℓ in 1:λ_indx.nZζ
            Znow, ζnow = λ_indx.Zζindx[ℓ, 1:2]
            w[k] += Λ[Znow] * λ[Znow][ζnow] * Q[Znow][k, Slagrev[λ_indx.indxs[Znow][ζnow]]...]
        end
    end
    w
end
