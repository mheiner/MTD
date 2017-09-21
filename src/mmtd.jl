# mmtd.jl

export ParamsMMTD, PriorMMTD, ModMMTD,
  build_λ_indx, sim_mmtd, symmetricPrior_mmtd, transTensor_mmtd,
  rpost_lΛ_mmtd, rpost_lλ_mmtd, counttrans_mmtd, rpost_lQ_mmtd,
  rpost_Z_mmtd, rpost_ζ_mmtd, rpost_ζ_mtd_marg, mcmc_mmtd!; # remove the inner functions after testing

type ParamsMMTD
  lΛ::Vector{Float64}
  lλ::Vector{Vector{Float64}}
  Z::Vector{Int} # will be length TT - R
  ζ::Matrix{Int} # (TT-R) by M: these are single indexes of the ordered λ sets
  lQ::Vector{Array{Float64}} # organized so first index is now and lag 1 is the next index
end

type PriorMMTD
  Λ::Union{Vector{Float64}, SparseDirMixPrior, SparseSBPrior, SparseSBPriorFull}
  λ::Union{Vector{Vector{Float64}}, Vector{SparseDirMixPrior}, Vector{SparseSBPrior}, Vector{SparseSBPriorP}, Vector{SparseSBPriorFull}}
  α0_Q::Vector{Matrix{Float64}} # organized in matricized form
end

type ModMMTD
  R::Int # maximal order
  M::Int # largest order considered
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMMTD
  state::ParamsMMTD
  λ_indx::Tuple
  iter::UInt64

  ModMMTD(R, M, K, TT, S, prior, state, λ_indx) = new(R, M, K, TT, S, prior, state, λ_indx, UInt64(0))
end

type PostSimsMMTD
  Λ::Matrix{Float64}
  λ::Array{Matrix{Float64}}
  Q::Array{Matrix{Float64}}
  Z::Matrix{Int}
  ζ::Array{Matrix{Int}}
  p1::Matrix{Float64}
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
  Λ::Array{Float64}, λ::Array{Vector{Float64}}, Q#=::Array{Array{Float64}}=#)

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
    transTensor_mmtd(R, M, K, λ_indx, Λ, λ, Q)

Calculate full transition tensor from Λ, λ, and Q.
"""
function transTensor_mmtd(R::Int, M::Int, K::Int, λ_indx::Tuple,
  Λ::Vector{Float64}, λ::Vector{Vector{Float64}}, Q#=::Vector{Array{Float64}}=#)

  froms, nfroms = create_froms(K, R) # in this case, ordered by now, lag1, lag2, etc.
  Ωmat = zeros(Float64, (K, nfroms))
  for i in 1:nfroms
    for k in 1:K
      for m in 1:M
        for ℓ in 1:λ_indx[2][m]
            Ωmat[k,i] += Λ[m] .* λ[m][ℓ] .* Q[m][k,froms[i][[λ_indx[1][m][ℓ]]...]...]
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

  w_now, z_now, ξ_now, prior.μ_now, prior.p1_now = BayesInference.rpost_sparseStickBreak(Z,
    prior.p1_now, prior.α, prior.μ_now, prior.M, prior.a_p1,
    prior.b_p1, prior.a_μ, prior.b_μ)

  log(w_now)
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
function rpost_lλ_mmtd(prior::Vector{SparseDirMixPrior}, ζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]

  for m in 1:M
    Nζ = StatsBase.counts(ζ[:,m], 1:λ_lens[m])
    α1_λ = prior[m].α + Nζ
    lλ_out[m] = BayesInference.rSparseDirMix(α1_λ, prior[m].β, true)
  end

  lλ_out
end
function rpost_lλ_mmtd(prior::Array{SparseSBPrior}, ζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]

  for m in 1:M
    Nζ = StatsBase.counts(ζ[:,m], 1:λ_lens[m])
    w_now, z_now, ξ_now = BayesInference.rpost_sparseStickBreak(Nζ,
        prior[m].p1, prior[m].α,
        prior[m].μ, prior[m].M )

    lλ_out[m] = log( copy(w_now) )
  end

  lλ_out
end
function rpost_lλ_mmtd!(prior::Array{SparseSBPriorP}, ζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]

  for m in 1:M
    Nζ = StatsBase.counts(ζ[:,m], 1:λ_lens[m])
    w_now, z_now, ξ_now, prior[m].p1_now = BayesInference.rpost_sparseStickBreak(Nζ,
        prior[m].p1_now, prior[m].α, prior[m].μ, prior[m].M,
        prior[m].a_p1, prior[m].b_p1)

    lλ_out[m] = log( copy(w_now) )
  end

  lλ_out
end
function rpost_lλ_mmtd!(prior::Array{SparseSBPriorFull}, ζ::Matrix{Int},
  λ_lens::Vector{Int}, M::Int)

  lλ_out = [ Vector{Float64}(λ_lens[m]) for m in 1:M ]

  for m in 1:M
    Nζ = StatsBase.counts(ζ[:,m], 1:λ_lens[m])
    w_now, z_now, ξ_now, prior[m].μ_now, prior[m].p1_now = BayesInference.rpost_sparseStickBreak(Nζ,
        prior[m].p1_now, prior[m].α, prior[m].μ_now, prior[m].M,
        prior[m].a_p1, prior[m].b_p1, prior[m].a_μ, prior[m].b_μ)

    lλ_out[m] = log( copy(w_now) )
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
    N_out[Z_now][append!( [copy(S[tt])], copy(Slagrev_now[ λ_indx[1][Z_now][ζ[tt-R,Z_now]] ]) )...] += 1
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
  lw = Vector{Float64}(M)

  for i in 1:(TT-R)
    tt = i + R
    Slagrev_now = S[range(tt-1, -1, R)]
    for m in 1:M
      lw[m] = lΛ[m] + lQ[m][ append!([copy(S[tt])], copy( Slagrev_now[λ_indx[1][m][ ζ[i,m] ]] ))... ]
    end
    Z_out[i] = StatsBase.sample(WeightVec( exp(lw) ))
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
      if Z[i] == m
        lp = Vector{Float64}(λ_lens[m])
        for j in 1:λ_lens[m]
          lp[j] = lλ[m][j] + lQ[m][ append!([copy(S[tt])], copy(Slagrev_now[λ_indx[1][m][j]]))... ]
        end
        ζ_out[i,m] = StatsBase.sample(WeightVec( exp(lp) ))
      else
        ζ_out[i,m] = StatsBase.sample(WeightVec( exp(lλ[m]) ))
      end
    end
  end

  ζ_out
end


"""
    rpost_ζ_mtd_marg(S, ζ_old, lλ, α0_Q, TT, R, K)

    Full conditinal updates for ζ marginalizing over Q
"""
function rpost_ζ_mtd_marg(S::Vector{Int}, ζ_old::Vector{Int},
    α0_Q::Matrix{Float64}, lλ::Vector{Float64},
    TT::Int, R::Int, K::Int)

  ζ_out = copy(ζ_old)
  N_now = counttrans_mtd(S, TT, ζ, R, K) # rows are tos, cols are froms

  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, -1, R)]
    N0 = copy(N_now)
    N0[ S[tt], Slagrev_now[ ζ_out[i] ] ] -= 1
    α1_Q = α0_Q + N0
    eSt = [1.0*(ii==S[tt]) for ii in 1:K]

    kuse = unique(Slagrev_now)
    nkuse = length(kuse)

    lmvbn0 = [ lmvbeta( α1_Q[:,kk] ) for kk in kuse ]
    lmvbn1 = [ lmvbeta( α1_Q[:,kk] + eSt ) for kk in kuse ]

    lw = zeros(Float64, R)

    for ℓ in 1:R
        lw[ℓ] = copy(lλ[ℓ])
        for kk in 1:nkuse
            if Slagrev_now[ℓ] == kuse[kk]
                lw[ℓ] += lmvbn1[kk]
            else
                lw[ℓ] += lmvbn0[kk]
            end
        end
    end

    w = exp( lw - maximum(lw) )
    ζ_out[i] = StatsBase.sample(WeightVec( w ))
    N_now = copy(N0)
    N_now[ S[tt], Slagrev_now[ ζ_out[i] ] ] += 1

  end

  ζ_out
end



"""
    mcmc_mmtd!(model, n_keep[, save=true, report_filename="out_progress.txt",
       thin=1, report_freq=500, monitor_indx=[1]])
"""
function mcmc_mmtd!(model::ModMMTD, n_keep::Int, save::Bool=true,
  report_filename::String="out_progress.txt", thin::Int=1, report_freq::Int=1000;
  monitor_indx::Vector{Int}=[1])

  ## output files
  report_file = open(report_filename, "a+")
  write(report_file, "Commencing MCMC at $(now())
    for $(n_keep * thin) iterations.\n")

  if save
    monitor_len = length(monitor_indx)
    sims = PostSimsMMTD( Matrix{Float64}(n_keep, model.M),
      [ Matrix{Float64}(n_keep, model.λ_indx[2][m]) for m in 1:model.M ],
      [ Matrix{Float64}(n_keep, model.K^(m+1)) for m in 1:model.M ],
      Matrix{Int}(n_keep, monitor_len),
      [ Matrix{Int}(n_keep, monitor_len) for m in 1:model.M ],
      Matrix{Float64}(n_keep, model.M) )
  end

  ## flags
  SBMp_flag = typeof(model.prior.λ) == Vector{BayesInference.SparseSBPriorP}
  SBMfull_flag = typeof(model.prior.λ) == Vector{BayesInference.SparseSBPriorFull}
  Mbig_flag = model.M > 1

  ## sampling
  for i in 1:n_keep
    for j in 1:thin

      if Mbig_flag
        model.state.Z = rpost_Z_mmtd(model.S, model.TT,
          model.state.lΛ, model.state.ζ, model.state.lQ, model.λ_indx,
          model.R, model.M)
      end

      if model.M == 1
          model.state.ζ = rpost_ζ_mtd_marg(model.S, model.state.ζ[:,1],
            model.prior.α0_Q[1], model.state.lλ[1],
            model.TT, model.R, model.K)
        else
          model.state.ζ = rpost_ζ_mmtd(model.S, model.TT,
            model.state.lλ, model.state.Z, model.state.lQ,
            model.λ_indx, model.R, model.M, model.K)
      end

      if model.M > 1
        model.state.lΛ = rpost_lΛ_mmtd(model.prior.Λ, model.state.Z, model.M)
      end

      if SBMp_flag || SBMfull_flag
        model.state.lλ = rpost_lλ_mmtd!(model.prior.λ, model.state.ζ,
          model.λ_indx[2], model.M)
      else
        model.state.lλ = rpost_lλ_mmtd(model.prior.λ, model.state.ζ,
          model.λ_indx[2], model.M)
      end

      model.state.lQ = rpost_lQ_mmtd(model.S, model.TT, model.prior.α0_Q,
        model.state.Z, model.state.ζ, model.λ_indx, model.R, model.M, model.K)

      model.iter += 1
      if model.iter % report_freq == 0
        write(report_file, "Iter $(model.iter) at $(now())\n")
      end
    end

    if save
      @inbounds sims.Λ[i,:] = exp( model.state.lΛ )
      @inbounds sims.Z[i,:] = copy(model.state.Z[monitor_indx])
      for m in 1:model.M
        @inbounds sims.λ[m][i,:] = exp( model.state.lλ[m] )
        @inbounds sims.Q[m][i,:] = exp( vec( model.state.lQ[m] ) )
        @inbounds sims.ζ[m][i,:] = copy(model.state.ζ[monitor_indx,m])
        if SBMp_flag || SBMfull_flag
            sims.p1[i,m] = copy(model.prior.λ[m].p1_now)
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
