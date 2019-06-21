# mtdg_inference.jl

export ParamsMTDg, PriorMTDg, ModMTDg,
  sim_mtdg, symmetricDirPrior_mtdg, transTensor_mtdg,
  counttrans_mtdg, mcmc!, timemod!, llik_MTDg,
  TankReduction, forecDist_MTDg;


mutable struct ParamsMTDg
    lλ::Vector{Float64} # will be a prob. vector of length L + 1
    ζ::Vector{Int} # will be length TT - L
    lQ0::Vector{Float64}
    lQ::Vector{Matrix{Float64}} # organized so first index is now and lag 1 is the next index

    ParamsMTDg(lλ, ζ, lQ0, lQ) = new(deepcopy(lλ), deepcopy(ζ), deepcopy(lQ0), deepcopy(lQ))
end

mutable struct PriorMTDg
  λ::Union{Vector{Float64}, SparseDirMix, SBMprior}
  Q0::Vector{Float64}
  Q::Union{Vector{Matrix{Float64}}, Vector{Vector{SparseDirMix}}, Vector{Vector{SBMprior}}}
  PriorMTDg(λ, Q0, Q) = new(deepcopy(λ), deepcopy(Q0), deepcopy(Q))
end

mutable struct ModMTDg
  L::Int # maximal order
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMTDg
  state::ParamsMTDg
  iter::Int

  ModMTDg(L, K, TT, S, prior, state) = new(L, K, TT, deepcopy(S), deepcopy(prior), deepcopy(state), 0)
end


"""
    sim_mtdg(TT, nburn, L, K, λ, Q0, Q)
"""
function sim_mtdg(TT::Int, nburn::Int, L::Int, K::Int,
  λ::Vector{Float64}, Q0::Vector{Float64}, Q::Vector{Matrix{Float64}})

  Nsim = nburn + TT

  ζ = [ StatsBase.sample(Weights(λ)) - 1 for i in 1:(Nsim-L) ]

  S = Vector{Int}(undef, Nsim)
  S[1:L] = StatsBase.sample(1:K, L)

  for tt in (L+1):(Nsim)
    i = tt - L
    if ζ[i] == 0
        pvec = deepcopy(Q0)
    else
        Slagrev_now = S[range(tt-1, step=-1, length=L)]
        pvec = deepcopy( Q[ζ[i]][:, Slagrev_now[ζ[i]] ] )
    end
    S[tt] = StatsBase.sample(Weights( pvec ))
  end

  S[(nburn+1):(Nsim)], ζ[(nburn+1):(Nsim-L)]
end


"""
    symmetricPrior_mtdg(size_λ, size_Q0, size_Q, L, K)
"""
function symmetricDirPrior_mtdg(size_λ::Float64, size_Q0::Float64, size_Q::Float64,
  L::Int, K::Int)

  α0_λ = fill( size_λ / float(L), L )
  α0_Q0 = fill( size_Q0 / float(K), K )
  a0_Q = size_Q / float(K)
  α0_Q = [ fill( a0_Q, (K, K) ) for ℓ=1:L ]

  (α0_λ, α0_Q0, α0_Q)
end


"""
    transTensor_mtdg(L, K, λ, Q0, Q)

Calculate full transition tensor from λ and Q.
"""
function transTensor_mtdg(L::Int, K::Int, λ::Vector{Float64},
    Q0::Vector{Float64}, Q::Vector{Matrix{Float64}})

    froms, nfroms = create_froms(K, L) # in this case, ordered by now, lag1, lag2, etc.
    Ωmat = zeros(Float64, (K, nfroms))
    for i in 1:nfroms
        for k in 1:K
            Ωmat[k,i] += λ[1] * Q0[k]
            for ℓ in 1:L
                Ωmat[k,i] += λ[ℓ+1] .* Q[ℓ][k,froms[i][ℓ]]
            end
        end
    end
    reshape(Ωmat, fill(K,L+1)...)
end


"""
    rpost_lλ_mtdg(α0_λ, ζ, L)
"""
function rpost_lλ_mtdg(α0_λ::Vector{Float64}, ζ::Vector{Int}, L::Int)

    Nζ = StatsBase.counts(ζ, 0:L)
    α1_λ = α0_λ .+ Nζ
    lλ_out = SparseProbVec.rDirichlet(α1_λ, logout=true)

    lλ_out
end
function rpost_lλ_mtdg(prior_λ::SparseDirMix, ζ::Vector{Int}, L::Int)

    Nζ = StatsBase.counts(ζ, 0:L)
    α1_λ = prior_λ.α .+ Nζ
    d = SparseProbVec.SparseDirMix(α1_λ, prior_λ.β)
    lλ_out = SparseProbVec.rand(d, logout=true)

    lλ_out
end
function rpost_lλ_mtdg(prior_λ::SBMprior, ζ::Vector{Int}, L::Int)

    Nζ = StatsBase.counts(ζ, 0:L)
    post_lλ = SparseProbVec.SBM_multinom_post(prior_λ, Nζ)
    lλ_out = SparseProbVec.rand(post_lλ, logout=true)

    lλ_out
end



"""
    counttrans_mtdg(S, TT, ζ, L, K)

    ### Example
    ```julia
    L = 2
    K = 3
    TT = 12
    S = [1,2,1,3,3,1,2,1,3,2,1,1]
    ζ = [1,2,0,0,1,2,1,2,1,2]
      counttrans_mtdg(S, TT, ζ, L, K)
    ```
"""
function counttrans_mtdg(S::Vector{Int}, TT::Int, ζ::Vector{Int},
  L::Int, K::Int)

  ## initialize
  N0_out = zeros(Int, K)
  N_out = [ zeros(Int, (K,K)) for ℓ = 1:L ]

  ## pass through data and add counts
  for tt in (L+1):(TT)
    if ζ[tt-L] == 0
        N0_out[ S[tt] ] += 1
    else
        Slagrev_now = deepcopy( S[range(tt-1, step=-1, length=L)] )
        from = deepcopy( Slagrev_now[ ζ[tt-L] ] )
        N_out[ζ[tt-L]][ S[tt], from ] += 1
    end
  end

  N0_out, N_out
end



"""
    rpost_lQ_mtdg(S, TT, prior_Q0, Prior_Q, ζ, L, K)
"""
function rpost_lQ_mtdg(S::Vector{Int}, TT::Int,
    prior_Q0::Vector{Float64}, prior_Q::Vector{Matrix{Float64}},
    ζ::Vector{Int}, L::Int, K::Int)

    N0, N = counttrans_mtdg(S, TT, ζ, L, K)

    ## initialize
    α0_Q0 = deepcopy(prior_Q0)
    α0_Q = deepcopy(prior_Q)
    lQ_out = [ Matrix{Float64}(undef, K, K) for ℓ in 1:L ]

    α1_Q0 = α0_Q0 .+ N0
    α1_Q = [ α0_Q[ℓ] .+ N[ℓ] for ℓ in 1:L ]

    lQ0_out = SparseProbVec.rDirichlet(α1_Q0, logout=true)
    for ℓ in 1:L
        for j in 1:K
            lQ_out[ℓ][:,j] = SparseProbVec.rDirichlet(α1_Q[ℓ][:,j], logout=true)
        end
    end

    lQ0_out, lQ_out
end
function rpost_lQ_mtdg(S::Vector{Int}, TT::Int,
    prior_Q0::Vector{Float64}, prior_Q::Vector{Vector{SparseDirMix}},
    ζ::Vector{Int}, L::Int, K::Int)

    N0, N = counttrans_mtdg(S, TT, ζ, L, K)

    ## initialize
    α0_Q0 = deepcopy(prior_Q0)
    lQ_out = [ Matrix{Float64}(undef, K, K) for ℓ in 1:L ]

    α1_Q0 = α0_Q0 .+ N0
    lQ0_out = SparseProbVec.rDirichlet(α1_Q0, logout=true)

    for ℓ in 1:L
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
    ζ::Vector{Int}, L::Int, K::Int)

    N0, N = counttrans_mtdg(S, TT, ζ, L, K)

    ## initialize
    α0_Q0 = deepcopy(prior_Q0)
    lQ_out = [ Matrix{Float64}(undef, K, K) for ℓ in 1:L ]

    α1_Q0 = α0_Q0 .+ N0
    lQ0_out = SparseProbVec.rDirichlet(α1_Q0, logout=true)

    for ℓ in 1:L
        for j in 1:K
            d = SparseProbVec.SBM_multinom_post(prior_Q[ℓ][j], N[ℓ][:,j])
            lQ_out[ℓ][:,j] = SparseProbVec.rand(d, logout=true)
        end
    end

    lQ0_out, lQ_out
end


"""
    rpost_ζ_mtdg(S, TT, lλ, lQ0, lQ, L, K)
"""
function rpost_ζ_mtdg(S::Vector{Int}, TT::Int,
  lλ::Vector{Float64}, lQ0::Vector{Float64}, lQ::Vector{Matrix{Float64}},
  L::Int, K::Int)

  ζ_out = Vector{Int}(undef, TT-L)

  for i in 1:(TT-L)
      tt = i + L
      Slagrev_now = S[range(tt-1, step=-1, length=L)]
      lp = Vector{Float64}(undef, L+1)

      lp[1] = lλ[1] + lQ0[S[tt]]
      for j in 1:L
          lp[j] = lλ[j+1] + lQ[j][ append!([deepcopy(S[tt])], deepcopy(Slagrev_now[j]))... ]
      end

      w = exp.( lp .- maximum(lp) )
      ζ_out[i] = StatsBase.sample(Weights(w)) - 1 # so that the values are 0:L
  end

  ζ_out
end


"""
    rpost_ζ_mtdg_marg(S, ζ_old, lλ, prior_Q0, prior_Q, TT, L, K)

Collapsed conditinal updates for ζ marginalizing over Q
"""
function rpost_ζ_mtdg_marg(S::Vector{Int}, ζ_old::Vector{Int},
    prior_Q0::Vector{Float64}, # always Dirichlet on intercept
    prior_Q::Union{Vector{Matrix{Float64}}, Vector{Vector{SparseDirMix}}, Vector{Vector{SBMprior}}},
    lλ::Vector{Float64},
    TT::Int, L::Int, K::Int)

    prior_Q_Dir_flag = typeof(prior_Q) <: Vector{Matrix{Float64}}

    α0_Q0 = deepcopy(prior_Q0)

    ζ_out = deepcopy(ζ_old)
    N0_now, N_now = counttrans_mtdg(S, TT, ζ_out, L, K) # in each N, rows are tos, cols are froms

    for i in 1:(TT-L)  # i indexes ζ, tt indexes S
        tt = i + L
        Slagrev_now = S[range(tt-1, step=-1, length=L)]

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
            for ℓ in 1:L
                α1_Q[ℓ] += Ninit[ℓ]
            end
        end

        eSt = [1.0*(ii==S[tt]) for ii in 1:K]

        lmvbn_without = Vector{Float64}(undef, L+1)
        lmvbn_with = Vector{Float64}(undef, L+1)

        lmvbn_without[1] = SparseProbVec.lmvbeta( α1_Q0 ) # Dirichlet prior on Q0
        lmvbn_with[1] = SparseProbVec.lmvbeta( α1_Q0 + eSt ) # Dirichlet prior on Q0

        if prior_Q_Dir_flag
            for ℓ in 1:L
                lmvbn_without[ℓ+1] = SparseProbVec.lmvbeta( α1_Q[ℓ][:,Slagrev_now[ℓ]] )
                lmvbn_with[ℓ+1] = SparseProbVec.lmvbeta( α1_Q[ℓ][:,Slagrev_now[ℓ]] + eSt )
            end
        else # assumes a matching method for another prior type
            for ℓ in 1:L
                lmvbn_without[ℓ+1] = SparseProbVec.logMarginal( prior_Q[ℓ][Slagrev_now[ℓ]],
                        Ninit[ℓ][:,Slagrev_now[ℓ]] )
                lmvbn_with[ℓ+1] = SparseProbVec.logMarginal( prior_Q[ℓ][Slagrev_now[ℓ]],
                        Ninit[ℓ][:,Slagrev_now[ℓ]] + Int64.(eSt) )
            end
        end

        lw = lλ .+ sum(lmvbn_without)
        for ℓ in 0:L
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
        TT::Int, L::Int, K::Int)

    Independence Metropolis step for λ and ζ.
"""
function MetropIndep_λζ(S::Vector{Int}, lλ_old::Vector{Float64}, ζ_old::Vector{Int},
    prior_λ::Union{Vector{Float64}, SparseDirMix, SBMprior},
    prior_Q0::Vector{Float64},
    prior_Q::Union{Vector{Matrix{Float64}}, Vector{Vector{SparseDirMix}}, Vector{Vector{SBMprior}}},
    TT::Int, L::Int, K::Int)

  if typeof(prior_λ) == Vector{Float64}
      lλ_cand = SparseProbVec.rDirichlet(prior_λ, logout=true)
  elseif typeof(prior_λ) == SBMprior
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  elseif typeof(prior_λ) == SparseDirMix
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  end

  λ_cand = exp.(lλ_cand)
  ζ_cand = [ StatsBase.sample( Weights(λ_cand) ) - 1 for i in 1:(TT-L) ]

  N0_cand, N_cand = counttrans_mtdg(S, TT, ζ_cand, L, K) # in each N, rows are tos, cols are froms
  N0_old, N_old = counttrans_mtdg(S, TT, ζ_old, L, K) # in each N, rows are tos, cols are froms

  ll_cand = SparseProbVec.lmvbeta(prior_Q0 .+ N0_cand) # can ignore denominators
  ll_old = SparseProbVec.lmvbeta(prior_Q0 .+ N0_old)

  if typeof(prior_Q) == Vector{Matrix{Float64}}
      for ℓ in 1:L
          for k in 1:K
              ll_cand += SparseProbVec.lmvbeta(prior_Q[ℓ][:,k] .+ N_cand[ℓ][:,k]) # can ignore denominators
              ll_old += SparseProbVec.lmvbeta(prior_Q[ℓ][:,k] .+ N_old[ℓ][:,k])
          end
      end
  else # assumes the logMarginal method exists for the type of prior
      for ℓ in 1:L
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
mcmc!(model::ModMTDg, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitor::Vector{Symbol}=[:lλ, :lQ0, :lQ])
"""
function mcmc!(model::ModMTDg, n_keep::Int; save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitor::Vector{Symbol}=[:lλ, :lQ0, :lQ])

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) at iteration $(model.iter) for $(n_keep * thin) iterations.\n")
    close(report_file)

    if save
        sims = postSimsInit(monitor, n_keep, model.state)
    end

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            jmpstart = (model.iter % jmpstart_iter == 0)

            if jmpstart

                model.state.lλ, model.state.ζ = MetropIndep_λζ(model.S,
                    model.state.lλ, model.state.ζ, model.prior.λ,
                    model.prior.Q0, model.prior.Q,
                    model.TT, model.L, model.K)

            else

                model.state.ζ = rpost_ζ_mtdg_marg(model.S, model.state.ζ,
                    model.prior.Q0, model.prior.Q, model.state.lλ,
                    model.TT, model.L, model.K)

                model.state.lλ = rpost_lλ_mtdg(model.prior.λ, model.state.ζ, model.L)

            end

            model.state.lQ0, model.state.lQ = rpost_lQ_mtdg(model.S, model.TT,
                    model.prior.Q0, model.prior.Q,
                    model.state.ζ, model.L, model.K)

            model.iter += 1
            if model.iter % report_freq == 0
                report_file = open(report_filename, "a+")
                write(report_file, "Iter $(model.iter) at $(Dates.now())\n")
                llik_now = llik_MTDg(model.S, model.state.lλ,
                    model.state.lQ0, model.state.lQ)
                write(report_file, "Log-likelihood $(llik_now)\n")
                write(report_file, "Current lag weights: $( round.(exp.(model.state.lλ), digits=3) )\n\n")
                close(report_file)
            end
        end

        if save
            for field in monitor
              sims[i][field] = deepcopy(getfield(model.state, field))
            end
            sims[i][:llik] = llik_MTDg(model.S, model.state.lλ,
                model.state.lQ0, model.state.lQ)
        end

    end

    if save
        return sims
    else
        return model.iter
    end

end


"""
    llik_MTDg(S::Vector{Int}, lλ::Vector{Float64},
        lQ0::Vector{Float64}, lQ::Vector{Matrix{Float64}})

Computes the (conditional on intial values) log-likelihood for a time series.

### Example
```julia
    lλ = log.([0.0, 0.0, 0.0, 1.0])
    lQ0 = log.([0.2, 0.8])
    lQ = [ log.(reshape([0.7, 0.3, 0.25, 0.75], (2,2))),
           log.(reshape([0.7, 0.3, 0.25, 0.75], (2,2))),
           log.(reshape([0.7, 0.3, 0.25, 0.75], (2,2))) ]
    S = [1,1,2,1,1,2,2,1,2,2,1,2,1,2,1,1,1,1,2,1,2]
    llik_MTDg(S, lλ, lQ0, lQ)
```
"""
function llik_MTDg(S::Vector{Int}, lλ::Vector{Float64},
    lQ0::Vector{Float64}, lQ::Vector{Matrix{Float64}})

    TT = length(S)
    L = length(lλ) - 1

    n = TT - L

    lpt = Matrix{Float64}(undef, n, L+1)

    for i in 1:n
        tt = i + L
        Slagrev_now = S[range(tt-1, step=-1, length=L)]

        lpt[i,1] = lλ[1] + lQ0[S[tt]]
        for ℓ in 1:L
            lpt[i,ℓ+1] = lλ[ℓ+1] + lQ[ℓ][S[tt], Slagrev_now[ℓ]]
        end
    end

    lptvec = [ SparseProbVec.logsumexp(lpt[i,:]) for i in 1:n ]
    llik = sum(lptvec)

    return llik
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
    L = length(λ) - 1

    Z0 = λ[1] .* Q0
    Z = [ λ[ℓ+1] .* Q[ℓ] for ℓ = 1:L ]

    a = [ [ minimum( Z[ℓ][k,:] ) for k = 1:K ] for ℓ = 1:L ]

    ZZ0 = deepcopy(Z0)
    ZZ = deepcopy(Z)
    for ℓ = 1:L
        ZZ0 += a[ℓ]
        for k = 1:K
            ZZ[ℓ][:,k] -= a[ℓ]
        end
    end

    λr = vcat( sum(ZZ0), [ sum(ZZ[ℓ][:,1]) for ℓ = 1:L ]  )
    Q0r = deepcopy(ZZ0)
    Qr = deepcopy(ZZ)

    if λr[1] > 0.0
        Q0r ./= λr[1]
    end

    for ℓ = 1:L
        if λr[ℓ+1] > 0.0
            Qr[ℓ] ./= λr[ℓ+1]
        end
    end

    λr, Q0r, Qr
end



"""
    forecDist_MTDg(Slag::Vector{Int}, λ::Vector{Float64}, Q::Matrix{Float64})

Computes the forecast distribution given lagged values and parameters.

### Example
```julia
    lam = [0.2, 0.4, 0.3, 0.1]
    Q0 = [0.5, 0.5]
    Q = [ reshape([0.7, 0.3, 0.25, 0.75], (2,2)) for ell = 1:3 ]
    Slagrev = [1,1,2]
    forecDist_MTDg(Slagrev, lam, Q0, Q)
```
"""
function forecDist_MTDg(Slagrev::Vector{Int}, λ::Vector{Float64},
    Q0::Vector{Float64}, Q::Vector{Matrix{Float64}})
    K = length(Q0)
    L = length(λ) - 1
    @assert length(Slagrev) == L
    w = zeros(Float64, K)
    for k in 1:K
        w[k] += λ[1] * Q0[k]
        for ℓ in 1:L
            w[k] += λ[ℓ+1] * Q[ℓ][k, Slagrev[ℓ]]
        end
    end
    w
end
