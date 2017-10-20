# gof.jl

export lossL1, meanForecLoss_MTD;

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
        end
    end
    (lossMat, missClassMat, negllMat, sqErrMat)
end
