# mtdg_inference.jl

export ParamsMTDg, PriorMTDg, ModMTDg, PostSimsMTDg,
  sim_mtdg, symmetricDirPrior_mtdg, transTensor_mtdg,
  counttrans_mtdg, mcmc_mtdg!, timemod!, TankReduction;


mutable struct ParamsMTDg
    lλ::Vector{Float64} # will be a prob. vector of length R + 1
    ζ::Vector{Int} # will be length TT - R
    lQ0::Vector{Float64}
    lQ::Vector{Matrix{Float64}} # organized so first index is now and lag 1 is the next index
end

mutable struct PriorMTDg
  λ::Union{Vector{Float64}, SparseDirMix, SBMprior}
  Q0::Vector{Float64}
  Q::Union{Vector{Matrix{Float64}}, Vector{Vector{SparseDirMix}}, Vector{Vector{SBMprior}}}
end

mutable struct ModMTDg
  R::Int # maximal order
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMTDg
  state::ParamsMTDg
  iter::Int

  ModMTDg(R, K, TT, S, prior, state) = new(R, K, TT, S, prior, state, 0)
end

mutable struct PostSimsMTDg
  λ::Matrix{Float64}
  Q0::Matrix{Float64}
  Q::Array{Float64}
  ζ::Matrix{Int}
end


"""
    sim_mtdg(TT, nburn, R, K, λ, Q0, Q)
"""
function sim_mtdg(TT::Int, nburn::Int, R::Int, K::Int,
  λ::Vector{Float64}, Q0::Vector{Float64}, Q::Vector{Matrix{Float64}})

  Nsim = nburn + TT

  ζ = [ StatsBase.sample(Weights(λ)) - 1 for i in 1:(Nsim-R) ]

  S = Vector{Int}(undef, Nsim)
  S[1:R] = StatsBase.sample(1:K, R)

  for tt in (R+1):(Nsim)
    i = tt - R
    if ζ[i] == 0
        pvec = deepcopy(Q0)
    else
        Slagrev_now = S[range(tt-1, step=-1, length=R)]
        pvec = deepcopy( Q[ζ[i]][:, Slagrev_now[ζ[i]] ] )
    end
    S[tt] = StatsBase.sample(Weights( pvec ))
  end

  S[(nburn+1):(Nsim)], ζ[(nburn+1):(Nsim-R)]
end


"""
    symmetricPrior_mtdg(size_λ, size_Q, size_Q0, R, K)
"""
function symmetricDirPrior_mtdg(size_λ::Float64, size_Q0::Float64, size_Q::Float64,
  R::Int, K::Int)

  α0_λ = fill( size_λ / float(R), R )
  α0_Q0 = fill( size_Q0 / float(K), K )
  a0_Q = size_Q / float(K)
  α0_Q = [ fill( a0_Q, (K, K) ) for ℓ=1:R ]

  (α0_λ, α0_Q0, α0_Q)
end


"""
    transTensor_mtdg(R, K, λ, Q0, Q)

Calculate full transition tensor from λ and Q.
"""
function transTensor_mtdg(R::Int, K::Int, λ::Vector{Float64},
    Q0::Vector{Float64}, Q::Vector{Matrix{Float64}})

    froms, nfroms = create_froms(K, R) # in this case, ordered by now, lag1, lag2, etc.
    Ωmat = zeros(Float64, (K, nfroms))
    for i in 1:nfroms
        for k in 1:K
            Ωmat[k,i] += λ[1] * Q0[k]
            for ℓ in 1:R
                Ωmat[k,i] += λ[ℓ+1] .* Q[ℓ][k,froms[i][ℓ]]
            end
        end
    end
    reshape(Ωmat, fill(K,R+1)...)
end


"""
    rpost_lλ_mtdg(α0_λ, ζ, R)
"""
function rpost_lλ_mtdg(α0_λ::Vector{Float64}, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 0:R)
    α1_λ = α0_λ .+ Nζ
    lλ_out = SparseProbVec.rDirichlet(α1_λ, logout=true)

    lλ_out
end
function rpost_lλ_mtdg(prior_λ::SparseDirMix, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 0:R)
    α1_λ = prior_λ.α .+ Nζ
    d = SparseProbVec.SparseDirMix(α1_λ, prior_λ.β)
    lλ_out = SparseProbVec.rand(d, logout=true)

    lλ_out
end
function rpost_lλ_mtdg(prior_λ::SBMprior, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 0:R)
    post_lλ = SparseProbVec.SBM_multinom_post(prior_λ, Nζ)
    lλ_out = SparseProbVec.rand(post_lλ, logout=true)

    lλ_out
end



"""
    counttrans_mtd(S, TT, ζ, R, K)

    ### Example
    ```julia
    R = 2
    K = 3
    TT = 12
    S = [1,2,1,3,3,1,2,1,3,2,1,1]
    ζ = [1,2,0,0,1,2,1,2,1,2]
      counttrans_mtdg(S, TT, ζ, R, K)
    ```
"""
function counttrans_mtdg(S::Vector{Int}, TT::Int, ζ::Vector{Int},
  R::Int, K::Int)

  ## initialize
  N0_out = zeros(Int, K)
  N_out = [ zeros(Int, (K,K)) for ℓ = 1:R ]

  ## pass through data and add counts
  for tt in (R+1):(TT)
    if ζ[tt-R] == 0
        N0_out[ S[tt] ] += 1
    else
        Slagrev_now = deepcopy( S[range(tt-1, step=-1, length=R)] )
        from = deepcopy( Slagrev_now[ ζ[tt-R] ] )
        N_out[ζ[tt-R]][ S[tt], from ] += 1
    end
  end

  N0_out, N_out
end



"""
    rpost_lQ_mtdg(S, TT, prior_Q0, Prior_Q, ζ, R, K)
"""
function rpost_lQ_mtdg(S::Vector{Int}, TT::Int,
    prior_Q0::Vector{Float64}, prior_Q::Vector{Matrix{Float64}},
    ζ::Vector{Int}, R::Int, K::Int)

    N0, N = counttrans_mtdg(S, TT, ζ, R, K)

    ## initialize
    α0_Q0 = deepcopy(prior_Q0)
    α0_Q = deepcopy(prior_Q)
    lQ_out = [ Matrix{Float64}(undef, K, K) for ℓ in 1:R ]

    α1_Q0 = α0_Q0 .+ N0
    α1_Q = [ α0_Q[ℓ] .+ N[ℓ] for ℓ in 1:R ]

    lQ0_out = SparseProbVec.rDirichlet(α1_Q0, logout=true)
    for ℓ in 1:R
        for j in 1:K
            lQ_out[ℓ][:,j] = SparseProbVec.rDirichlet(α1_Q[ℓ][:,j], logout=true)
        end
    end

    lQ0_out, lQ_out
end
function rpost_lQ_mtdg(S::Vector{Int}, TT::Int,
    prior_Q0::Vector{Float64}, prior_Q::Vector{Vector{SparseDirMix}},
    ζ::Vector{Int}, R::Int, K::Int)

    N0, N = counttrans_mtdg(S, TT, ζ, R, K)

    ## initialize
    α0_Q0 = deepcopy(prior_Q0)
    lQ_out = [ Matrix{Float64}(undef, K, K) for ℓ in 1:R ]

    α1_Q0 = α0_Q0 .+ N0
    lQ0_out = SparseProbVec.rDirichlet(α1_Q0, logout=true)

    for ℓ in 1:R
        for j in 1:K
            α1 = prior_Q[ℓ][j].α .+ N[ℓ][:,j]
            d = SparseProbVec.SparseDirMix(α1, prior_Q[ℓ][j].β)
            lQ_out[ℓ][:,j] = SparseProbVec.rand(d, logout=true)
        end
    end

    lQ0_out, lQ_out
end
function rpost_lQ_mtdg(S::Vector{Int}, TT::Int,
    prior_Q0::Vector{Float64}, prior_Q::Vector{Vector{SBMprior}},
    ζ::Vector{Int}, R::Int, K::Int)

    N0, N = counttrans_mtdg(S, TT, ζ, R, K)

    ## initialize
    α0_Q0 = deepcopy(prior_Q0)
    lQ_out = [ Matrix{Float64}(undef, K, K) for ℓ in 1:R ]

    α1_Q0 = α0_Q0 .+ N0
    lQ0_out = SparseProbVec.rDirichlet(α1_Q0, logout=true)

    for ℓ in 1:R
        for j in 1:K
            d = SparseProbVec.SBM_multinom_post(prior_Q[ℓ][j], N[ℓ][:,j])
            lQ_out[ℓ][:,j] = SparseProbVec.rand(d, logout=true)
        end
    end

    lQ_out
end


"""
    rpost_ζ_mtdg(S, TT, lλ, lQ0, lQ, R, K)
"""
function rpost_ζ_mtdg(S::Vector{Int}, TT::Int,
  lλ::Vector{Float64}, lQ0::Vector{Float64}, lQ::Vector{Matrix{Float64}},
  R::Int, K::Int)

  ζ_out = Vector{Int}(undef, TT-R)

  for i in 1:(TT-R)
      tt = i + R
      Slagrev_now = S[range(tt-1, step=-1, length=R)]
      lp = Vector{Float64}(undef, R+1)

      lp[1] = lλ[1] + lQ0[S[tt]]
      for j in 1:R
          lp[j] = lλ[j+1] + lQ[j][ append!([deepcopy(S[tt])], deepcopy(Slagrev_now[j]))... ]
      end

      w = exp.( lp .- maximum(lp) )
      ζ_out[i] = StatsBase.sample(Weights(w)) - 1 # so that the values are 0:R
  end

  ζ_out
end


"""
    rpost_ζ_mtdg_marg(S, ζ_old, lλ, prior_Q0, prior_Q, TT, R, K)

Collapsed conditinal updates for ζ marginalizing over Q
"""
function rpost_ζ_mtdg_marg(S::Vector{Int}, ζ_old::Vector{Int},
    prior_Q0::Vector{Float64}, # always Dirichlet on intercept
    prior_Q::Union{Vector{Matrix{Float64}}, Vector{Vector{SparseDirMix}}, Vector{Vector{SBMprior}}},
    lλ::Vector{Float64},
    TT::Int, R::Int, K::Int)

    prior_Q_Dir_flag = typeof(prior_Q) <: Vector{Matrix{Float64}}

    α0_Q0 = deepcopy(prior_Q0)

    ζ_out = deepcopy(ζ_old)
    N0_now, N_now = counttrans_mtdg(S, TT, ζ_out, R, K) # in each N, rows are tos, cols are froms

    for i in 1:(TT-R)  # i indexes ζ, tt indexes S
        tt = i + R
        Slagrev_now = S[range(tt-1, step=-1, length=R)]

        N0init = deepcopy(N0_now)
        Ninit = deepcopy(N_now)
        if ζ_out[i] == 0
            N0init[ S[tt] ] -= 1
        else
            Ninit[ζ_out[i]][ S[tt], Slagrev_now[ ζ_out[i] ] ] -= 1
        end
        α1_Q0 = α0_Q0 .+ N0init
        if prior_Q_Dir_flag
            α1_Q = deepcopy(prior_Q)
            for ℓ in 1:R
                α1_Q[ℓ] += Ninit[ℓ]
            end
        end

        eSt = [1.0*(ii==S[tt]) for ii in 1:K]

        lmvbn_without = Vector{Float64}(undef, R+1)
        lmvbn_with = Vector{Float64}(undef, R+1)

        lmvbn_without[1] = SparseProbVec.lmvbeta( α1_Q0 ) # Dirichlet prior on Q0
        lmvbn_with[1] = SparseProbVec.lmvbeta( α1_Q0 + eSt ) # Dirichlet prior on Q0

        if prior_Q_Dir_flag
            for ℓ in 1:R
                lmvbn_without[ℓ+1] = SparseProbVec.lmvbeta( α1_Q[ℓ][:,Slagrev_now[ℓ]] )
                lmvbn_with[ℓ+1] = SparseProbVec.lmvbeta( α1_Q[ℓ][:,Slagrev_now[ℓ]] + eSt )
            end
        else # assumes a matching method for another prior type
            for ℓ in 1:R
                lmvbn_without[ℓ+1] = SparseProbVec.logMarginal( prior_Q[ℓ][Slagrev_now[ℓ]],
                        Ninit[ℓ][:,Slagrev_now[ℓ]] )
                lmvbn_with[ℓ+1] = SparseProbVec.logMarginal( prior_Q[ℓ][Slagrev_now[ℓ]],
                        Ninit[ℓ][:,Slagrev_now[ℓ]] + Int64.(eSt) )
            end
        end

        lw = lλ .+ sum(lmvbn_without)
        for ℓ in 0:R
            lw[ℓ+1] -= lmvbn_without[ℓ+1]
            lw[ℓ+1] += lmvbn_with[ℓ+1]
        end

        w = exp.( lw .- maximum(lw) )
        ζ_out[i] = StatsBase.sample(Weights( w )) - 1 # 0 refers to intercept (no lag dependence)
        N0_now = deepcopy(N0init)
        N_now = deepcopy(Ninit)
        if ζ_out[i] == 0
            N0_now[ S[tt] ] += 1
        else
            N_now[ζ_out[i]][ S[tt], Slagrev_now[ ζ_out[i] ] ] += 1
        end

    end

    ζ_out
end



"""
    MetropIndep_λζ(S::Vector{Int}, lλ_old::Vector{Float64}, ζ_old::,
        prior_λ, prior_Q0, prior_Q,
        TT::Int, R::Int, K::Int)

    Independence Metropolis step for λ and ζ.
"""
function MetropIndep_λζ(S::Vector{Int}, lλ_old::Vector{Float64}, ζ_old::Vector{Int},
    prior_λ::Union{Vector{Float64}, SparseDirMix, SBMprior},
    prior_Q0::Vector{Float64},
    prior_Q::Union{Vector{Matrix{Float64}}, Vector{Vector{SparseDirMix}}, Vector{Vector{SBMprior}}},
    TT::Int, R::Int, K::Int)

  if typeof(prior_λ) == Vector{Float64}
      lλ_cand = SparseProbVec.rDirichlet(prior_λ, logout=true)
  elseif typeof(prior_λ) == SBMprior
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  elseif typeof(prior_λ) == SparseDirMix
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  end

  λ_cand = exp.(lλ_cand)
  ζ_cand = [ StatsBase.sample( Weights(λ_cand) ) - 1 for i in 1:(TT-R) ]

  N0_cand, N_cand = counttrans_mtdg(S, TT, ζ_cand, R, K) # in each N, rows are tos, cols are froms
  N0_old, N_old = counttrans_mtdg(S, TT, ζ_old, R, K) # in each N, rows are tos, cols are froms

  ll_cand = SparseProbVec.lmvbeta(prior_Q0 .+ N0_cand) # can ignore denominators
  ll_old = SparseProbVec.lmvbeta(prior_Q0 .+ N0_old)

  if typeof(prior_Q) == Vector{Matrix{Float64}}
      for ℓ in 1:R
          for k in 1:K
              ll_cand += SparseProbVec.lmvbeta(prior_Q[ℓ][:,k] .+ N_cand[ℓ][:,k]) # can ignore denominators
              ll_old += SparseProbVec.lmvbeta(prior_Q[ℓ][:,k] .+ N_old[ℓ][:,k])
          end
      end
  else # assumes the logMarginal method exists for the type of prior
      for ℓ in 1:R
          for k in 1:K
              ll_cand += SparseProbVec.logMarginal(prior_Q[ℓ][k], N_cand[ℓ][:,k]) # can ignore denominators
              ll_old += SparseProbVec.logMarginal(prior_Q[ℓ][k], N_old[ℓ][:,k])
          end
      end
  end

  lu = log(rand())
  if lu < (ll_cand - ll_old)
      lλ_out = lλ_cand
      ζ_out = ζ_cand
  else
      lλ_out = lλ_old
      ζ_out = ζ_old
  end

  (lλ_out, ζ_out)
end



"""
mcmc_mtdg!(model::ModMTDg, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitorS_indx::Vector{Int}=[1])
"""
function mcmc_mtdg!(model::ModMTDg, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitorS_indx::Vector{Int}=[1])

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    if save
        monitorS_len = length(monitorS_indx)
        sims = PostSimsMTDg(  zeros(Float64, n_keep, model.R + 1), # λ
        zeros(Float64, n_keep, model.K), # Q0
        zeros(Float64, n_keep, model.R, model.K, model.K), # Q
        zeros(Int, n_keep, monitorS_len) #= ζ =# )
    end

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            jmpstart = (model.iter % jmpstart_iter == 0)

            if jmpstart

                model.state.lλ, model.state.ζ = MetropIndep_λζ(model.S,
                    model.state.lλ, model.state.ζ, model.prior.λ,
                    model.prior.Q0, model.prior.Q,
                    model.TT, model.R, model.K)

            else

                model.state.ζ = rpost_ζ_mtdg_marg(model.S, model.state.ζ,
                    model.prior.Q0, model.prior.Q, model.state.lλ,
                    model.TT, model.R, model.K)

                model.state.lλ = rpost_lλ_mtdg(model.prior.λ, model.state.ζ, model.R)

            end

            model.state.lQ0, model.state.lQ = rpost_lQ_mtdg(model.S, model.TT,
                    model.prior.Q0, model.prior.Q,
                    model.state.ζ, model.R, model.K)

            model.iter += 1
            if model.iter % report_freq == 0
                write(report_file, "Iter $(model.iter) at $(Dates.now())\n")
            end
        end

        if save
            @inbounds sims.λ[i,:] = exp.( model.state.lλ )
            @inbounds sims.Q0[i,:] = exp.( model.state.lQ0 )
            for ℓ in 1:model.R
                @inbounds sims.Q[i,ℓ,:,:] = exp.( model.state.lQ[ℓ] )
            end
            @inbounds sims.ζ[i,:] = deepcopy(model.state.ζ[monitorS_indx])
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
function timemod!(n::Int64, model::ModMTDg, niter::Int, outfilename::String)
    outfile = open(outfilename, "a+")
    write(outfile, "timing for $(niter) iterations each:\n")
    for i in 1:n
        tinfo = @timed mcmc_mtdg!(model, niter, false, outfilename)
        write(outfile, "trial $(i), elapsed: $(tinfo[2]) seconds, allocation: $(tinfo[3]/1.0e6) Megabytes\n")
    end
    close(outfile)
end


"""
    TankReduction(λ, Q0, Q)

    Computes the unique Tank Reduction of the MTDg with an intercept.

    ### Example
    ```julia
    lam = [0.4, 0.5, 0.1]
    Q0 = [0.1, 0.2, 0.7]
    Q = [ [0.4 0.4 0.6; 0.3 0.1 0.2; 0.3 0.5 0.2],
          [0.2 0.5 0.2; 0.5 0.2 0.1; 0.3 0.3 0.7] ]

    TankReduction(lam, Q0, Q)
    ```
"""
function TankReduction(λ::Vector{Float64},
    Q0::Vector{Float64}, Q::Vector{Matrix{Float64}})

    # here each Q is organized with tos as rows and froms as columns

    K = length(Q0)
    R = length(λ) - 1

    Z0 = λ[1] .* Q0
    Z = [ λ[ℓ+1] .* Q[ℓ] for ℓ = 1:R ]

    a = [ [ minimum( Z[ℓ][k,:] ) for k = 1:K ] for ℓ = 1:R ]

    ZZ0 = deepcopy(Z0)
    ZZ = deepcopy(Z)
    for ℓ = 1:R
        ZZ0 += a[ℓ]
        for k = 1:K
            ZZ[ℓ][:,k] -= a[ℓ]
        end
    end

    λr = vcat( sum(ZZ0), [ sum(ZZ[ℓ][:,1]) for ℓ = 1:R ]  )
    Q0r = deepcopy(ZZ0)
    Qr = deepcopy(ZZ)

    if λr[1] > 0.0
        Q0r ./= λr[1]
    end

    for ℓ = 1:R
        if λr[ℓ+1] > 0.0
            Qr[ℓ] ./= λr[ℓ+1]
        end
    end

    λr, Q0r, Qr
end
