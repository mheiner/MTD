# mtd_inference.jl

export ParamsMTD, PriorMTD, ModMTD,
  sim_mtd, symmetricDirPrior_mtd, transTensor_mtd,
  counttrans_mtd,
  mcmc!, timemod!, etr, llik_MTD;

mutable struct ParamsMTD
  lλ::Vector{Float64}
  ζ::Vector{Int} # will be length TT - L
  lQ::Matrix{Float64} # organized so first index is now and lag 1 is the next index

  ParamsMTD(lλ, ζ, lQ) = new(deepcopy(lλ), deepcopy(ζ), deepcopy(lQ))
end

mutable struct PriorMTD
  λ::Union{Vector{Float64}, SparseDirMix, SBMprior}
  Q::Union{Matrix{Float64}, Vector{SparseDirMix}, Vector{SBMprior}}

  PriorMTD(λ, Q) = new(deepcopy(λ), deepcopy(Q))
end

mutable struct ModMTD
  L::Int # maximal order
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMTD
  state::ParamsMTD
  iter::Int

  ModMTD(L, K, TT, S, prior, state) = new(L, K, TT, deepcopy(S), deepcopy(prior), deepcopy(state), 0)
end


"""
    sim_mtd(TT, nburn, L, K, λ, Q)
"""
function sim_mtd(TT::Int, nburn::Int, L::Int, K::Int,
  λ::Vector{Float64}, Q::Matrix{Float64})

  Nsim = nburn + TT

  ζ = [ StatsBase.sample(Weights(λ)) for i in 1:(Nsim-L) ]

  S = Vector{Int}(undef, Nsim)
  S[1:L] = StatsBase.sample(1:K, L)

  for tt in (L+1):(Nsim)
    i = tt - L
    Slagrev_now = S[range(tt-1, step=-1, length=L)]
    pvec = deepcopy( Q[:, Slagrev_now[ζ[i]] ] )
    S[tt] = StatsBase.sample(Weights( pvec ))
  end

  S[(nburn+1):(Nsim)], ζ[(nburn+1):(Nsim-L)]
end


"""
    symmetricPrior_mtd(size_λ, size_Q, L, K)
"""
function symmetricDirPrior_mtd(size_λ::Float64, size_Q::Float64,
  L::Int, K::Int)

  α0_λ = fill( size_λ / float(L), L )
  a0_Q = size_Q / float(K)
  α0_Q = fill( a0_Q, (K, K) )

  (α0_λ, α0_Q)
end


"""
    transTensor_mtd(L, K, λ, Q)

Calculate full transition tensor from λ and Q.
"""
function transTensor_mtd(L::Int, K::Int, λ::Vector{Float64}, Q::Matrix{Float64})

    froms, nfroms = create_froms(K, L) # in this case, ordered by now, lag1, lag2, etc.
    Ωmat = zeros(Float64, (K, nfroms))
    for i in 1:nfroms
        for k in 1:K
            for ℓ in 1:L
                Ωmat[k,i] += λ[ℓ] .* Q[k,froms[i][ℓ]]
            end
        end
    end
    reshape(Ωmat, fill(K,L+1)...)
end

"""
    rpost_lλ_mtd(α0_λ, ζ, L)
"""
function rpost_lλ_mtd(α0_λ::Vector{Float64}, ζ::Vector{Int}, L::Int)

    Nζ = StatsBase.counts(ζ, 1:L)
    α1_λ = α0_λ .+ Nζ
    lλ_out = SparseProbVec.rDirichlet(α1_λ, logout=true)

    lλ_out
end
function rpost_lλ_mtd(prior::SparseDirMix, ζ::Vector{Int}, L::Int)

    Nζ = StatsBase.counts(ζ, 1:L)
    α1_λ = prior.α .+ Nζ
    d = SparseProbVec.SparseDirMix(α1_λ, prior.β)
    lλ_out = SparseProbVec.rand(d, logout=true)

    lλ_out
end
function rpost_lλ_mtd(prior::SBMprior, ζ::Vector{Int}, L::Int)

    Nζ = StatsBase.counts(ζ, 1:L)
    post_lλ = SparseProbVec.SBM_multinom_post(prior, Nζ)
    lλ_out = SparseProbVec.rand(post_lλ, logout=true)

    lλ_out
end


"""
    counttrans_mtd(S, TT, ζ, L, K)

    ### Example
    ```julia
    L = 2
    K = 3
    TT = 12
    S = [1,2,1,3,3,1,2,1,3,2,1,1]
    ζ = [1,2,1,2,1,2,1,2,1,2]
      counttrans_mtd(S, TT, ζ, L, K)
    ```
"""
function counttrans_mtd(S::Vector{Int}, TT::Int, ζ::Vector{Int},
  L::Int, K::Int)

  ## initialize
  N_out = zeros(Int, (K,K))

  ## pass through data and add counts
  for tt in (L+1):(TT)
    Slagrev_now = deepcopy( S[range(tt-1, step=-1, length=L)] )
    from = deepcopy( Slagrev_now[ ζ[tt-L] ] )
    N_out[ S[tt], from ] += 1
  end

  N_out
end

"""
    rpost_lQ_mtd(S, TT, prior, ζ, L, K)
"""
function rpost_lQ_mtd(S::Vector{Int}, TT::Int, prior::Matrix{Float64},
    ζ::Vector{Int}, L::Int, K::Int)

    ## initialize
    α0_Q = deepcopy(prior)
    lQ_out = Matrix{Float64}(undef, K, K)

    N = counttrans_mtd(S, TT, ζ, L, K)

    α1_Q = α0_Q .+ N
    for j in 1:K
        lQ_out[:,j] = SparseProbVec.rDirichlet(α1_Q[:,j], logout=true)
    end

    lQ_out
end
function rpost_lQ_mtd(S::Vector{Int}, TT::Int,
    prior::Vector{SparseDirMix},
    ζ::Vector{Int}, L::Int, K::Int)

    ## initialize
    lQ_out = Matrix{Float64}(undef, K, K)

    N = counttrans_mtd(S, TT, ζ, L, K)

    for j in 1:K
        α1 = prior[j].α .+ N[:,j]
        d = SparseProbVec.SparseDirMix(α1, prior[j].β)
        lQ_out[:,j] = SparseProbVec.rand(d, logout=true)
    end

    lQ_out
end
function rpost_lQ_mtd(S::Vector{Int}, TT::Int,
    prior::Vector{SBMprior}, ζ::Vector{Int},
    L::Int, K::Int)

    ## initialize
    lQ_out = Matrix{Float64}(undef, K, K)

    N = counttrans_mtd(S, TT, ζ, L, K)

    for j in 1:K
        d = SparseProbVec.SBM_multinom_post(prior[j], N[:,j])
        lQ_out[:,j] = SparseProbVec.rand(d, logout=true)
    end

    lQ_out
end


"""
    rpost_ζ_mtd(S, TT, lλ, lQ, L, K)
"""
function rpost_ζ_mtd(S::Vector{Int}, TT::Int,
  lλ::Vector{Float64}, lQ::Array{Float64,2},
  L::Int, K::Int)

  ζ_out = Vector{Int}(undef, TT-L)

  for i in 1:(TT-L)
      tt = i + L
      Slagrev_now = S[range(tt-1, step=-1, length=L)]
      lp = Vector{Float64}(undef, L)
      for j in 1:L
          lp[j] = lλ[j] + lQ[ append!([deepcopy(S[tt])], deepcopy(Slagrev_now[j]))... ]
      end

      w = exp.( lp .- maximum(lp) )
      ζ_out[i] = StatsBase.sample(Weights(w))
  end

  ζ_out
end


"""
    rpost_ζ_mtd_marg(S, ζ_old, lλ, prior_Q, TT, L, K)

Collapsed conditinal updates for ζ marginalizing over Q
"""
function rpost_ζ_mtd_marg(S::Vector{Int}, ζ_old::Vector{Int},
    prior_Q::Matrix{Float64},
    lλ::Vector{Float64},
    TT::Int, L::Int, K::Int)

  α0_Q = deepcopy(prior_Q)
  ζ_out = deepcopy(ζ_old)
  N_now = counttrans_mtd(S, TT, ζ_old, L, K) # rows are tos, cols are froms

  for i in 1:(TT-L)  # i indexes ζ, tt indexes S
    tt = i + L
    Slagrev_now = S[range(tt-1, step=-1, length=L)]
    N0 = deepcopy(N_now)
    N0[ S[tt], Slagrev_now[ ζ_out[i] ] ] -= 1
    α1_Q = α0_Q .+ N0
    eSt = [1.0*(ii==S[tt]) for ii in 1:K]

    kuse = unique(Slagrev_now)
    nkuse = length(kuse)

    lmvbn0 = [ SparseProbVec.lmvbeta( α1_Q[:,kk] ) for kk in kuse ]
    lmvbn1 = [ SparseProbVec.lmvbeta( α1_Q[:,kk] + eSt ) for kk in kuse ]

    lw = zeros(Float64, L)

    for ℓ in 1:L
        lw[ℓ] = deepcopy(lλ[ℓ])
        for kk in 1:nkuse
            if Slagrev_now[ℓ] == kuse[kk]
                lw[ℓ] += lmvbn1[kk]
            else
                lw[ℓ] += lmvbn0[kk]
            end
        end
    end

    w = exp.( lw .- maximum(lw) )
    ζ_out[i] = StatsBase.sample(Weights( w ))
    N_now = deepcopy(N0)
    N_now[ S[tt], Slagrev_now[ ζ_out[i] ] ] += 1

  end

  ζ_out
end
function rpost_ζ_mtd_marg(S::Vector{Int}, ζ_old::Vector{Int},
    prior_Q::Union{Vector{SparseDirMix}, Vector{SBMprior}},
    lλ::Vector{Float64},
    TT::Int, L::Int, K::Int)

  ζ_out = deepcopy(ζ_old)
  N_now = counttrans_mtd(S, TT, ζ_old, L, K) # rows are tos, cols are froms

  for i in 1:(TT-L)  # i indexes ζ, tt indexes S
    tt = i + L
    Slagrev_now = S[range(tt-1, step=-1, length=L)]
    N0 = deepcopy(N_now)
    N0[ S[tt], Slagrev_now[ ζ_out[i] ] ] -= 1
    eSt = [1*(ii==S[tt]) for ii in 1:K]

    kuse = unique(Slagrev_now)
    nkuse = length(kuse)

    lmarg0 = [ SparseProbVec.logMarginal(prior_Q[kk], N0[:,kk]) for kk in kuse ]
    lmarg1 = [ SparseProbVec.logMarginal(prior_Q[kk], N0[:,kk] + eSt) for kk in kuse ]

    lw = zeros(Float64, L)

    for ℓ in 1:L
        lw[ℓ] = deepcopy(lλ[ℓ])
        for kk in 1:nkuse
            if Slagrev_now[ℓ] == kuse[kk]
                lw[ℓ] += lmarg1[kk]
            else
                lw[ℓ] += lmarg0[kk]
            end
        end
    end

    w = exp.( lw .- maximum(lw) )
    ζ_out[i] = StatsBase.sample(Weights( w ))
    N_now = deepcopy(N0)
    N_now[ S[tt], Slagrev_now[ ζ_out[i] ] ] += 1

  end

  ζ_out
end


"""
    MetropIndep_λζ(S::Vector{Int}, lλ_old::Vector{Float64}, ζ_old::,
        prior_λ, prior_Q,
        TT::Int, L::Int, K::Int)

    Independence Metropolis step for λ and ζ.
"""
function MetropIndep_λζ(S::Vector{Int}, lλ_old::Vector{Float64}, ζ_old::Vector{Int},
    prior_λ::Union{Vector{Float64}, SparseDirMix, SBMprior},
    prior_Q::Union{Matrix{Float64}, Vector{SparseDirMix}, Vector{SBMprior}},
    TT::Int, L::Int, K::Int)

  if typeof(prior_λ) == Vector{Float64}
      lλ_cand = SparseProbVec.rDirichlet(prior_λ, logout=true)
  elseif typeof(prior_λ) == SBMprior
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  elseif typeof(prior_λ) == SparseDirMix
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  end

  λ_cand = exp.(lλ_cand)
  ζ_cand = [ StatsBase.sample( Weights(λ_cand) ) for i in 1:(TT-L) ]

  N_cand = counttrans_mtd(S, TT, ζ_cand, L, K) # rows are tos, cols are froms
  N_old = counttrans_mtd(S, TT, ζ_old, L, K) # rows are tos, cols are froms

  if typeof(prior_Q) == Matrix{Float64}
      lmargQ_cand = [ SparseProbVec.lmvbeta(N_cand[:,kk] .+ prior_Q[:,kk]) for kk in 1:K ] # can ignore denominator
      lmargQ_old = [ SparseProbVec.lmvbeta(N_old[:,kk] .+ prior_Q[:,kk]) for kk in 1:K ]
  else
      lmargQ_cand = [ SparseProbVec.logMarginal(prior_Q[kk], N_cand[:,kk]) for kk = 1:K ]
      lmargQ_old = [ SparseProbVec.logMarginal(prior_Q[kk], N_old[:,kk]) for kk = 1:K ]
  end

  ll_cand = sum( lmargQ_cand )
  ll_old = sum( lmargQ_old )

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
mcmc!(model::ModMTD, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitor::Vector{Symbol}=[:lλ, :lQ])
"""
function mcmc!(model::ModMTD, n_keep::Int; save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitor::Vector{Symbol}=[:lλ, :lQ])

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
                    model.prior.Q,
                    model.TT, model.L, model.K)

            else

                model.state.ζ = rpost_ζ_mtd_marg(model.S, model.state.ζ,
                    model.prior.Q, model.state.lλ,
                    model.TT, model.L, model.K)

                model.state.lλ = rpost_lλ_mtd(model.prior.λ, model.state.ζ, model.L)

            end

            model.state.lQ = rpost_lQ_mtd(model.S, model.TT, model.prior.Q,
                    model.state.ζ, model.L, model.K)

            model.iter += 1
            if model.iter % report_freq == 0
                report_file = open(report_filename, "a+")
                write(report_file, "Iter $(model.iter) at $(Dates.now())\n")
                llik_now = llik_MTD(model.S, model.state.lλ, model.state.lQ)
                write(report_file, "Log-likelihood $(llik_now)\n")
                write(report_file, "Current lag weights: $( round.(exp.(model.state.lλ), digits=3) )\n\n")
                close(report_file)
            end
        end

        if save
            for field in monitor
              sims[i][field] = deepcopy(getfield(model.state, field))
            end
            sims[i][:llik] = llik_MTD(model.S, model.state.lλ, model.state.lQ)
        end
    end

    if save
        return sims
    else
        return model.iter
    end

end


"""
    llik_MTD(S::Vector{Int}, lλ::Vector{Vector{Float64}}, lQ::Matrix{Float64})

Computes the (conditional on intial values) log-likelihood for a time series.

### Example
```julia
    lλ = log.([0.0, 0.0, 1.0])
    lQ = log.(reshape([0.7, 0.3, 0.25, 0.75], (2,2)))
    S = [1,1,2,1,1,2,2,1,2,2,1,2,1,2,1,1,1,1,2,1,2]
    llik_MTD(S, lλ, lQ)
```
"""
function llik_MTD(S::Vector{Int}, lλ::Vector{Float64}, lQ::Matrix{Float64})

    TT = length(S)
    L = length(lλ)

    n = TT - L

    lpt = Matrix{Float64}(undef, n, L)

    for i in 1:n
        tt = i + L
        Slagrev_now = S[range(tt-1, step=-1, length=L)]

        for ℓ in 1:L
            lpt[i,ℓ] = lλ[ℓ] + lQ[S[tt], Slagrev_now[ℓ]]
        end
    end

    lptvec = [ SparseProbVec.logsumexp(lpt[i,:]) for i in 1:n ]
    llik = sum(lptvec)

    return llik
end
