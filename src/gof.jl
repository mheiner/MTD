# gof.jl

export lossL1, meanForecLoss_MTD, meanForecLoss_MMTD;

function lossL1(x::Vector{Float64}, y::Vector{Float64})
    assert(length(x)==length(y))
    sum( abs.( x .- y ) )
end

function lossMissedClass(x::Int64, y::Vector{Float64})
    Int64( x != indmax(y) )
end

function lossSqErr(x::Float64, y::Vector{Float64})
    ( x - collect(Float64, 1:length(y))'y )^2
end


function meanForecLoss_MTD(S::Vector{Int}, tprime::Vector{Int},
    lossFn::Function, sims::MTD.PostSimsMMTD, TT::Int, R::Int, K::Int,
    simind::Vector{Int})

    nsim = length(simind)
    nprime = length(tprime)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim
        for j in 1:nprime
            ii = copy( simind[i] )
            tt = copy( tprime[j] )
            S_now = copy( S[tt] )
            Slagrev_now = copy( S[range(tt-1, -1, R)] )
            Q_now = reshape( sims.Q[1][ii,:], (K,K) ) # output of sims is MMTD

            forec = forecDist(Slagrev_now, sims.λ[1][ii,:], Q_now)
            St_vec = float([ (k == S_now) for k in 1:K ])

            lossMat[i,j] = lossFn(St_vec, forec)
            missClassMat[i,j] = lossMissedClass(S_now, forec)
            negllMat[i,j] = -log(forec[S_now])
            sqErrMat[i,j] = lossSqErr(Float64(S_now), forec)

            forecMat[i,j,:] = copy(forec)
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat, forecMat)
end
function meanForecLoss_MTD(y::Vector{Int}, X::Matrix{Int}, P::Matrix{Float64},
    nprime::Int,
    lossFn::Function, sims::MTD.PostSimsMMTD, TT::Int, R::Int, K::Int,
    simind::Vector{Int})

    nsim = length(simind)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    PlossMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim
        for tt in 1:nprime
            ii = copy( simind[i] )
            S_now = copy( y[tt] )
            Slagrev_now = copy( X[tt,:] )
            Q_now = reshape( sims.Q[1][ii,:], (K,K) ) # output of sims is MMTD

            forec = forecDist(Slagrev_now, sims.λ[1][ii,:], Q_now)
            St_vec = float([ (k == S_now) for k in 1:K ])

            lossMat[i,tt] = lossFn(St_vec, forec)
            missClassMat[i,tt] = lossMissedClass(S_now, forec)
            negllMat[i,tt] = -log(forec[S_now])
            sqErrMat[i,tt] = lossSqErr(Float64(S_now), forec)

            PlossMat[i,tt] = lossFn(P[tt,:], forec) / float(K)

            forecMat[i,tt,:] = copy(forec)
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat, PlossMat, forecMat)
end


function meanForecLoss_MMTD(S::Vector{Int}, tprime::Vector{Int},
    lossFn::Function, sims::MTD.PostSimsMMTD, TT::Int, R::Int, M::Int, K::Int,
    simind::Vector{Int}, λ_indx::λindxMMTD)

    nsim = length(simind)
    nprime = length(tprime)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim

        ii = copy( simind[i] )

        λ_now = [ copy(sims.λ[m][ii,:]) for m in 1:M ]
        Q_now = [ reshape( sims.Q[m][ii,:], fill(K, m+1)... ) for m in 1:M ]

        for j in 1:nprime
            tt = copy( tprime[j] )
            S_now = copy( S[tt] )
            Slagrev_now = copy( S[range(tt-1, -1, R)] )

            forec = forecDist_MMTD(Slagrev_now, sims.Λ[ii,:],
                                   λ_now, Q_now, λ_indx)
            St_vec = float([ (k == S_now) for k in 1:K ])

            lossMat[i,j] = lossFn(St_vec, forec)
            missClassMat[i,j] = lossMissedClass(S_now, forec)
            negllMat[i,j] = -log(forec[S_now])
            sqErrMat[i,j] = lossSqErr(Float64(S_now), forec)

            forecMat[i,j,:] = copy(forec)
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat, forecMat)
end
function meanForecLoss_MMTD(y::Vector{Int}, X::Matrix{Int}, P::Matrix{Float64},
    nprime::Int,
    lossFn::Function, sims::MTD.PostSimsMMTD, TT::Int, R::Int, M::Int, K::Int,
    simind::Vector{Int}, λ_indx::λindxMMTD)

    nsim = length(simind)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    PlossMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim

        ii = copy( simind[i] )

        λ_now = [ copy(sims.λ[m][ii,:]) for m in 1:M ]
        Q_now = [ reshape( sims.Q[m][ii,:], fill(K, m+1)... ) for m in 1:M ]

        for tt in 1:nprime
            S_now = copy( y[tt] )
            Slagrev_now = copy( X[tt,:] )

            forec = forecDist_MMTD(Slagrev_now, sims.Λ[ii,:],
                                   λ_now, Q_now, λ_indx)
            St_vec = float([ (k == S_now) for k in 1:K ])

            lossMat[i,tt] = lossFn(St_vec, forec)
            missClassMat[i,tt] = lossMissedClass(S_now, forec)
            negllMat[i,tt] = -log(forec[S_now])
            sqErrMat[i,tt] = lossSqErr(Float64(S_now), forec)

            PlossMat[i,tt] = lossFn(P[tt,:], forec) / float(K)

            forecMat[i,tt,:] = copy(forec)
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat, PlossMat, forecMat)
end
