# general.jl

export create_froms, count_trans_L, transTens_MLE, forecDist,
    deepcopyFields, timemod!, etr;

function create_froms(K::UInt16, L::UInt16)
  x = UInt16(1):K
  xx = collect(Iterators.product(Iterators.repeated(x,Int(L))...))
  (vec(xx), length(xx))
end
function create_froms(K::Int, L::Int)
  x = 1:K
  xx = collect(Iterators.product(Iterators.repeated(x,L)...))
  (vec(xx), length(xx))
end

"""
    count_trans_L(S, K, L)

Returns a tensor with counts of observed `L`th order transitions in `S`
  for a chain with `K` states. The first dimension is time now, the next
  dimension is lag 1, then lag 2, and so on.

### Example
```julia
  S = [1,2,1,1,1,2,4,1,3,1,1,3,1,2,1,1,2]
  count_trans_L(S, 4, 2)
```
"""
function count_trans_L(S::Vector{Int}, K::Int, L::Int)
    @assert minimum(S) >= 1 && maximum(S) <= K
    N = zeros(Int, fill(K,L+1)...)

    for tt in (L+1):length(S)
      Srev_now = S[range(tt, step=-1, length=(L+1))]
      N[(Srev_now)...] += 1
    end

    N
end


"""
    transTens_MLE(S, K, L)

Returns a tensor with estimated transition probabilities for `L`th order transitions in `S`
  for a chain with `K` states. The first dimension is time now, the next
  dimension is lag 1, then lag 2, and so on.

### Example
```julia
  S = [1,2,1,1,1,2,4,1,3,1,1,3,1,2,1,1,2]
  N = count_trans_L(S, 4, 2)
  transTens_MLE(S, 4, 2)
  transTens_MLD(N)
```
"""
function transTens_MLE(S::Vector{Int}, K::Int, L::Int)
    @assert minimum(S) >= 1 && maximum(S) <= K
    N = count_trans_L(S, K, L)
    N ./ sum(N, dims=1)
end
function transTens_MLE(N::Union{Array{Float64}, Array{Int64}})
    N ./ sum(N, dims=1)
end



"""
    forecDist(Slag::Vector{Int}, λ::Vector{Float64}, Q::Matrix{Float64})

Computes the forecast distribution given lagged values and parameters.

### Example
```julia
    lam = [0.6, 0.3, 0.1]
    Q = reshape([0.7, 0.3, 0.25, 0.75], (2,2))
    Slagrev = [1,1,2]
    forecDist(Slagrev, lam, Q)
```
"""
function forecDist(Slagrev::Vector{Int}, λ::Vector{Float64}, Q::Matrix{Float64})
    K = size(Q)[1]
    L = size(λ)[1]
    @assert size(Slagrev)[1] == L
    w = zeros(Float64, K)
    for k in 1:K
        for ℓ in 1:L
            w[k] += λ[ℓ] * Q[k, Slagrev[ℓ]]
        end
    end
    w
end


## from Arthur Lui
function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  for field in fields
    substate[field] = deepcopy(getfield(state, field))
  end

  return substate
end


function postSimsInit(monitor::Vector{Symbol}, n_keep::Int, init_state::Union{ParamsMTD, ParamsMTDg, ParamsMMTD})

    state = deepcopyFields(init_state, monitor)
    state[:llik] = 0.0

    sims = [ deepcopy(state) for i = 1:n_keep ]

    return sims
end


## MCMC timing for benchmarks
function timemod!(n::Int64, model::Union{ModMTD, ModMTDg, ModMMTD}, niter::Int, outfilename::String)
    outfile = open(outfilename, "a+")
    write(outfile, "timing for $(niter) iterations each:\n")
    for i in 1:n
        tinfo = @timed mcmc!(model, niter, save=false, report_filename=outfilename)
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
      estimated completion time $(estimatedfinish)\n\n")
    close(report_file)
end
