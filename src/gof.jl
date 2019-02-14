# gof.jl

export lossL1, meanForecLoss_MTD, meanForecLoss_MTDg, meanForecLoss_MMTD;

function lossL1(x::Vector{Float64}, y::Vector{Float64})
    @assert length(x)==length(y)
    sum( abs.( x .- y ) )
end

function lossMissedClass(x::Int, y::Vector{Float64})
    Int( x != findmax(y)[2] )
end

function lossSqErr(x::Float64, y::Vector{Float64})
    ( x - collect(Float64, 1:length(y))'y )^2
end


function meanForecLoss(S::Vector{Int}, tprime::Vector{Int},
    lossFn::Function, sims::Union{PostSimsMTD, PostSimsMTDg, PostSimsMMTD},
    TT::Int, R::Int, K::Int,
    simind::Vector{Int}; λ_indx::Union{nothing, λindxMMTD}=nothing)

    nsim = length(simind)
    nprime = length(tprime)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim
        ii = deepcopy( simind[i] )
        if typeof(sims) == PostSimsMMTD
            M = length(sims.λ)
            λ_now = [ deepcopy(sims.λ[m][ii,:]) for m in 1:M ]
            Q_now = [ reshape( sims.Q[m][ii,:], fill(K, m+1)... ) for m in 1:M ]
        end

        for j in 1:nprime
            tt = deepcopy( tprime[j] )
            S_now = deepcopy( S[tt] )
            Slagrev_now = deepcopy( S[range(tt-1, step=-1, length=R)] )

            if typeof(sims) == PostSimsMTD
                forec = forecDist(Slagrev_now, sims.λ[ii,:], sims.Q[ii,:,:])
            elseif typeof(sims) == PostSimsMTDg
                forec = forecDist_MTDg(Slagrev_now, sims.λ[ii,:], sims.Q0[ii,:], [sims.Q[ii,ell,:,:] for ell = 1:R])
            elseif typeof(sims) == PostSimsMMTD
                forec = forecDist_MMTD(Slagrev_now, sims.Λ[ii,:], λ_now,
                                       sims.Q0[ii,:], Q_now, λ_indx)
            end

            St_vec = float([ (k == S_now) for k in 1:K ])

            lossMat[i,j] = lossFn(St_vec, forec)
            missClassMat[i,j] = lossMissedClass(S_now, forec)
            negllMat[i,j] = -log(forec[S_now])
            sqErrMat[i,j] = lossSqErr(Float64(S_now), forec)

            forecMat[i,j,:] = deepcopy(forec)
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat, forecMat)
end
function meanForecLoss(y::Vector{Int}, X::Matrix{Int}, P::Matrix{Float64},
    nprime::Int, lossFn::Function,
    sims::Union{PostSimsMTD, PostSimsMTDg, PostSimsMMTD},
    TT::Int, R::Int, K::Int,
    simind::Vector{Int}; λ_indx::Union{nothing, λindxMMTD}=nothing)

    nsim = length(simind)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    PlossMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim

        ii = deepcopy( simind[i] )

        if typeof(sims) == PostSimsMMTD
            M = length(sims.λ)
            λ_now = [ deepcopy(sims.λ[m][ii,:]) for m in 1:M ]
            Q_now = [ reshape( sims.Q[m][ii,:], fill(K, m+1)... ) for m in 1:M ]
        end

        for tt in 1:nprime
            S_now = deepcopy( y[tt] )
            Slagrev_now = deepcopy( X[tt,:] )

            if typeof(sims) == PostSimsMTD
                forec = forecDist(Slagrev_now, sims.λ[ii,:], sims.Q[ii,:,:])
            elseif typeof(sims) == PostSimsMTDg
                forec = forecDist_MTDg(Slagrev_now, sims.λ[ii,:], sims.Q0[ii,:], [sims.Q[ii,ell,:,:] for ell = 1:R])
            elseif typeof(sims) == PostSimsMMTD
                forec = forecDist_MMTD(Slagrev_now, sims.Λ[ii,:], λ_now,
                                       sims.Q0[ii,:], Q_now, λ_indx)
            end

            St_vec = float([ (k == S_now) for k in 1:K ])

            lossMat[i,tt] = lossFn(St_vec, forec)
            missClassMat[i,tt] = lossMissedClass(S_now, forec)
            negllMat[i,tt] = -log(forec[S_now])
            sqErrMat[i,tt] = lossSqErr(Float64(S_now), forec)

            PlossMat[i,tt] = lossFn(P[tt,:], forec) / float(K)

            forecMat[i,tt,:] = deepcopy(forec)
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat, PlossMat, forecMat)
end
