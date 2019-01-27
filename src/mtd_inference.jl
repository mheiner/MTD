# mtd_inference.jl

export ParamsMTD, PriorMTD, ModMTD, PostSimsMTD,
  sim_mtd, symmetricDirPrior_mtd, transTensor_mtd,
  counttrans_mtd,
  mcmc_mtd!, timemod!, etr; # remove the inner functions after testing

mutable struct ParamsMTD
  lλ::Vector{Float64}
  ζ::Vector{Int} # will be length TT - R
  lQ::Matrix{Float64} # organized so first index is now and lag 1 is the next index
end

mutable struct PriorMTD
  λ::Union{Vector{Float64}, SparseDirMix, SBMprior}
  Q::Union{Matrix{Float64}, Vector{SparseDirMix}, Vector{SBMprior}}
end

mutable struct ModMTD
  R::Int # maximal order
  K::Int # number of states
  TT::Int
  S::Vector{Int}
  prior::PriorMTD
  state::ParamsMTD
  iter::Int

  ModMTD(R, K, TT, S, prior, state) = new(R, K, TT, S, prior, state, 0)
end

mutable struct PostSimsMTD
  λ::Matrix{Float64}
  Q::Array{Float64}
  ζ::Matrix{Int}
end


"""
    sim_mtd(TT, nburn, R, K, λ, Q)
"""
function sim_mtd(TT::Int, nburn::Int, R::Int, K::Int,
  λ::Vector{Float64}, Q::Matrix{Float64})

  Nsim = nburn + TT

  ζ = [ StatsBase.sample(Weights(λ)) for i in 1:(Nsim-R) ]

  S = Vector{Int}(undef, Nsim)
  S[1:R] = StatsBase.sample(1:K, R)

  for tt in (R+1):(Nsim)
    i = tt - R
    Slagrev_now = S[range(tt-1, step=-1, length=R)]
    pvec = deepcopy( Q[:, Slagrev_now[ζ[i]] ] )
    S[tt] = StatsBase.sample(Weights( pvec ))
  end

  S[(nburn+1):(Nsim)], ζ[(nburn+1):(Nsim-R)]
end


"""
    symmetricPrior_mtd(size_λ, size_Q, R, K)
"""
function symmetricDirPrior_mtd(size_λ::Float64, size_Q::Float64,
  R::Int, K::Int)

  α0_λ = fill( size_λ / float(R), R )
  a0_Q = size_Q / float(K)
  α0_Q = fill( a0_Q, (K, K) )

  (α0_λ, α0_Q)
end


"""
    transTensor_mtd(R, K, λ, Q)

Calculate full transition tensor from λ and Q.
"""
function transTensor_mtd(R::Int, K::Int, λ::Vector{Float64}, Q::Matrix{Float64})

    froms, nfroms = create_froms(K, R) # in this case, ordered by now, lag1, lag2, etc.
    Ωmat = zeros(Float64, (K, nfroms))
    for i in 1:nfroms
        for k in 1:K
            for ℓ in 1:R
                Ωmat[k,i] += λ[ℓ] .* Q[k,froms[i][ℓ]]
            end
        end
    end
    reshape(Ωmat, fill(K,R+1)...)
end

"""
    rpost_lλ_mtd(α0_λ, ζ, R)
"""
function rpost_lλ_mtd(α0_λ::Vector{Float64}, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 1:R)
    α1_λ = α0_λ .+ Nζ
    lλ_out = SparseProbVec.rDirichlet(α1_λ, logout=true)

    lλ_out
end
function rpost_lλ_mtd(prior::SparseDirMix, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 1:R)
    α1_λ = prior.α .+ Nζ
    d = SparseProbVec.SparseDirMix(α1_λ, prior.β)
    lλ_out = SparseProbVec.rand(d, logout=true)

    lλ_out
end
function rpost_lλ_mtd(prior::SBMprior, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 1:R)
    post_lλ = SparseProbVec.SBM_multinom_post(prior, Nζ)
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
    Slagrev_now = deepcopy( S[range(tt-1, step=-1, length=R)] )
    from = deepcopy( Slagrev_now[ ζ[tt-R] ] )
    N_out[ S[tt], from ] += 1
  end

  N_out
end

"""
    rpost_lQ_mtd(S, TT, prior, ζ, R, K)
"""
function rpost_lQ_mtd(S::Vector{Int}, TT::Int, prior::Matrix{Float64},
    ζ::Vector{Int}, R::Int, K::Int)

    ## initialize
    α0_Q = deepcopy(prior)
    lQ_out = Matrix{Float64}(undef, K, K)

    N = counttrans_mtd(S, TT, ζ, R, K)

    α1_Q = α0_Q .+ N
    for j in 1:K
        lQ_out[:,j] = SparseProbVec.rDirichlet(α1_Q[:,j], logout=true)
    end

    lQ_out
end
function rpost_lQ_mtd(S::Vector{Int}, TT::Int,
    prior::Vector{SparseDirMix},
    ζ::Vector{Int}, R::Int, K::Int)

    ## initialize
    lQ_out = Matrix{Float64}(undef, K, K)

    N = counttrans_mtd(S, TT, ζ, R, K)

    for j in 1:K
        α1 = prior[j].α .+ N[:,j]
        d = SparseProbVec.SparseDirMix(α1, prior[j].β)
        lQ_out[:,j] = SparseProbVec.rand(d, logout=true)
    end

    lQ_out
end
function rpost_lQ_mtd(S::Vector{Int}, TT::Int,
    prior::Vector{SBMprior}, ζ::Vector{Int},
    R::Int, K::Int)

    ## initialize
    lQ_out = Matrix{Float64}(undef, K, K)

    N = counttrans_mtd(S, TT, ζ, R, K)

    for j in 1:K
        d = SparseProbVec.SBM_multinom_post(prior[j], N[:,j])
        lQ_out[:,j] = SparseProbVec.rand(d, logout=true)
    end

    lQ_out
end


"""
    rpost_ζ_mtd(S, TT, lλ, lQ, R, K)
"""
function rpost_ζ_mtd(S::Vector{Int}, TT::Int,
  lλ::Vector{Float64}, lQ::Array{Float64,2},
  R::Int, K::Int)

  ζ_out = Vector{Int}(undef, TT-R)

  for i in 1:(TT-R)
      tt = i + R
      Slagrev_now = S[range(tt-1, step=-1, length=R)]
      lp = Vector{Float64}(undef, R)
      for j in 1:R
          lp[j] = lλ[j] + lQ[ append!([deepcopy(S[tt])], deepcopy(Slagrev_now[j]))... ]
      end

      w = exp.( lp .- maximum(lp) )
      ζ_out[i] = StatsBase.sample(Weights(w))
  end

  ζ_out
end


"""
    rpost_ζ_mtd_marg(S, ζ_old, lλ, prior_Q, TT, R, K)

Collapsed conditinal updates for ζ marginalizing over Q
"""
function rpost_ζ_mtd_marg(S::Vector{Int}, ζ_old::Vector{Int},
    prior_Q::Matrix{Float64},
    lλ::Vector{Float64},
    TT::Int, R::Int, K::Int)

  α0_Q = deepcopy(prior_Q)
  ζ_out = deepcopy(ζ_old)
  N_now = counttrans_mtd(S, TT, ζ_old, R, K) # rows are tos, cols are froms

  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, step=-1, length=R)]
    N0 = deepcopy(N_now)
    N0[ S[tt], Slagrev_now[ ζ_out[i] ] ] -= 1
    α1_Q = α0_Q .+ N0
    eSt = [1.0*(ii==S[tt]) for ii in 1:K]

    kuse = unique(Slagrev_now)
    nkuse = length(kuse)

    lmvbn0 = [ SparseProbVec.lmvbeta( α1_Q[:,kk] ) for kk in kuse ]
    lmvbn1 = [ SparseProbVec.lmvbeta( α1_Q[:,kk] + eSt ) for kk in kuse ]

    lw = zeros(Float64, R)

    for ℓ in 1:R
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
    TT::Int, R::Int, K::Int)

  ζ_out = deepcopy(ζ_old)
  N_now = counttrans_mtd(S, TT, ζ_old, R, K) # rows are tos, cols are froms

  for i in 1:(TT-R)  # i indexes ζ, tt indexes S
    tt = i + R
    Slagrev_now = S[range(tt-1, step=-1, length=R)]
    N0 = deepcopy(N_now)
    N0[ S[tt], Slagrev_now[ ζ_out[i] ] ] -= 1
    eSt = [1*(ii==S[tt]) for ii in 1:K]

    kuse = unique(Slagrev_now)
    nkuse = length(kuse)

    lmarg0 = [ SparseProbVec.logMarginal(prior_Q[kk], N0[:,kk]) for kk in kuse ]
    lmarg1 = [ SparseProbVec.logMarginal(prior_Q[kk], N0[:,kk] + eSt) for kk in kuse ]

    lw = zeros(Float64, R)

    for ℓ in 1:R
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
        TT::Int, R::Int, K::Int)

    Independence Metropolis step for λ and ζ.
"""
function MetropIndep_λζ(S::Vector{Int}, lλ_old::Vector{Float64}, ζ_old::Vector{Int},
    prior_λ::Union{Vector{Float64}, SparseDirMix, SBMprior},
    prior_Q::Union{Matrix{Float64}, Vector{SparseDirMix}, Vector{SBMprior}},
    TT::Int, R::Int, K::Int)

  if typeof(prior_λ) == Vector{Float64}
      lλ_cand = SparseProbVec.rDirichlet(prior_λ, logout=true)
  elseif typeof(prior_λ) == SBMprior
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  elseif typeof(prior_λ) == SparseDirMix
      lλ_cand = SparseProbVec.rand(prior_λ, logout=true)
  end

  λ_cand = exp.(lλ_cand)
  ζ_cand = [ StatsBase.sample( Weights(λ_cand) ) for i in 1:(TT-R) ]

  N_cand = counttrans_mtd(S, TT, ζ_cand, R, K) # rows are tos, cols are froms
  N_old = counttrans_mtd(S, TT, ζ_old, R, K) # rows are tos, cols are froms

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
mcmc_mtd!(model::ModMTD, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitorS_indx::Vector{Int}=[1])
"""
function mcmc_mtd!(model::ModMTD, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1, jmpstart_iter::Int=25,
    report_freq::Int=1000;
    monitorS_indx::Vector{Int}=[1])

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    if save
        monitorS_len = length(monitorS_indx)
        sims = PostSimsMTD(  zeros(Float64, n_keep, model.R), # λ
        zeros(Float64, n_keep, model.K, model.K), # Q
        zeros(Int, n_keep, monitorS_len) #= ζ =# )
    end

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            jmpstart = (model.iter % jmpstart_iter == 0)

            if jmpstart

                model.state.lλ, model.state.ζ = MetropIndep_λζ(model.S,
                    model.state.lλ, model.state.ζ, model.prior.λ,
                    model.prior.Q,
                    model.TT, model.R, model.K)

            else

                model.state.ζ = rpost_ζ_mtd_marg(model.S, model.state.ζ,
                    model.prior.Q, model.state.lλ,
                    model.TT, model.R, model.K)

                model.state.lλ = rpost_lλ_mtd(model.prior.λ, model.state.ζ, model.R)

            end

            model.state.lQ = rpost_lQ_mtd(model.S, model.TT, model.prior.Q,
                    model.state.ζ, model.R, model.K)

            model.iter += 1
            if model.iter % report_freq == 0
                write(report_file, "Iter $(model.iter) at $(Dates.now())\n")
            end
        end

        if save
            @inbounds sims.λ[i,:] = exp.( model.state.lλ )
            @inbounds sims.Q[i,:,:] = exp.( model.state.lQ )
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
function timemod!(n::Int64, model::ModMTD, niter::Int, outfilename::String)
    outfile = open(outfilename, "a+")
    write(outfile, "timing for $(niter) iterations each:\n")
    for i in 1:n
        tinfo = @timed mcmc_mtd!(model, niter, false, outfilename)
        write(outfile, "trial $(i), elapsed: $(tinfo[2]) seconds, allocation: $(tinfo[3]/1.0e6) Megabytes\n")
    end
    close(outfile)
end

## estimate time remaining
function etr(timestart::DateTime, n_keep::Int, thin::Int, outfilename::String)
    timeendburn = now()
    durperiter = (timeendburn - timestart).value / 1.0e5 # in milliseconds
    milsecremaining = durperiter * (n_keep * thin)
    estimatedfinish = now() + Dates.Millisecond(Int64(round(milsecremaining)))
    report_file = open(outfilename, "a+")
    write(report_file, "Completed burn-in at $(durperiter/1.0e3*1000.0) seconds per 1000 iterations \n
      $(durperiter/1.0e3/60.0*1000.0) minutes per 1000 iterations \n
      $(durperiter/1.0e3/60.0/60.0*1000.0) hours per 1000 iterations \n
      estimated completion time $(estimatedfinish)")
    close(report_file)
end
