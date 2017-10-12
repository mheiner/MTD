# gof.jl

export lossL1, meanForecLoss_MTD;

function lossL1(x::Vector{Float64}, y::Vector{Float64})
    assert(length(x)==length(y))
    sum( abs.( x .- y ) )
end

function meanForecLoss_MTD(S::Vector{Int}, tprime::Vector{Int},
    lossFn::Function, sims::MTD.PostSimsMMTD, TT::Int, R::Int, K::Int,
    simind::Vector{Int})

    nsim = length(simind)
    nprime = length(tprime)
    lossMat = zeros(Float64, nsim, nprime)
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
        end
    end
    mean(lossMat)
    # lossMat
end
