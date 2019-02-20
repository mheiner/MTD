# gof.jl

export lossL1, meanForecLoss;

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
    lossFn::Function, sims::Vector{Dict},
    TT::Int, R::Int, K::Int,
    simind::Vector{Int}; λ_indx::Union{Nothing, λindxMMTD}=nothing,
    modeltype::String)

    nsim = length(simind)
    nprime = length(tprime)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim
        ii = deepcopy( simind[i] )
        if modeltype == "MMTD"
            M = length(sims[1].λ)
            λ_now = [ exp.(sims[ii][:lλ][m]) for m in 1:M ]
            Q_now = [ exp.( sims[ii][:lQ][m] ) for m in 1:M ]
        end

        for j in 1:nprime
            tt = deepcopy( tprime[j] )
            S_now = deepcopy( S[tt] )
            Slagrev_now = deepcopy( S[range(tt-1, step=-1, length=R)] )

            if modeltype == "MTD"
                forec = forecDist(Slagrev_now, exp.(sims[ii][:lλ]),
                    exp.(sims[ii][:lQ]))
            elseif modeltype == "MTDg"
                forec = forecDist_MTDg(Slagrev_now, exp.(sims[ii][:lλ]),
                    exp.(sims[ii][:lQ0]), [ exp.(sims[ii][:lQ][ell]) for ell = 1:R ] )
            elseif modeltype == "MMTD"
                forec = forecDist_MMTD(Slagrev_now, exp.(sims[ii][:lΛ]), λ_now,
                                       exp.(sims[ii][:lQ0]), Q_now, λ_indx)
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
    sims::Vector{Dict},
    TT::Int, R::Int, K::Int,
    simind::Vector{Int}; λ_indx::Union{Nothing, λindxMMTD}=nothing,
    modeltype::String)

    nsim = length(simind)
    lossMat = zeros(Float64, nsim, nprime)
    missClassMat = zeros(Int, nsim, nprime)
    negllMat = zeros(Float64, nsim, nprime)
    sqErrMat = zeros(Float64, nsim, nprime)

    PlossMat = zeros(Float64, nsim, nprime)

    forecMat = zeros(Float64, nsim, nprime, K)

    for i in 1:nsim

        ii = deepcopy( simind[i] )

        if modeltype == "MMTD"
            M = length(sims[1].λ)
            λ_now = [ exp.(sims[ii][:lλ][m]) for m in 1:M ]
            Q_now = [ exp.( sims[ii][:lQ][m] ) for m in 1:M ]
        end

        for tt in 1:nprime
            S_now = deepcopy( y[tt] )
            Slagrev_now = deepcopy( X[tt,:] )

            if modeltype == "MTD"
                forec = forecDist(Slagrev_now, exp.(sims[ii][:lλ]),
                exp.(sims[ii][:lQ]))
            elseif modeltype == "MTDg"
                forec = forecDist_MTDg(Slagrev_now, exp.(sims[ii][:lλ]),
                exp.(sims[ii][:lQ0]), [ exp.(sims[ii][:lQ][ell]) for ell = 1:R ] )
            elseif modeltype == "MMTD"
                forec = forecDist_MMTD(Slagrev_now, exp.(sims[ii][:lΛ]), λ_now,
                exp.(sims[ii][:lQ0]), Q_now, λ_indx)
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
