# reduce.jl

export TransTens, DecompMMTDg, print, reduce_transTens;

struct TransTens
    A::Array
    order::Int
    function TransTens(AA::Array{T}; sum1tol=1.0e-6) where T <: Real
        ord = ndims(AA) - 1
        d = size(AA)
        all( d.==d[1]) || error("All dimensions of transition tensor must have same length.")
        all( AA .>= 0.0 ) || error("Transition tensor must have non-negative entries.")
        maximum(abs.( sum(AA, dims=1) .- 1.0 )) < sum1tol || error("Transition tensor must add to unity along first dimension.")
        return new(AA, ord)
    end
end

struct DecompMMTDg
    Omega::Union{TransTens, Array{T}} where T <: Real
    order::Int
    totalweight_tol::T where T <: Real
    levels::Array{T,2} where T <: Real
    configs::Vector{Vector{Int}}
    Q::Vector{Array{T}} where T <: Real
    involv::Vector{T} where T <: Real

    function DecompMMTDg(Omega::Union{TransTens, Array{T}}, totalweight_tol::T, levels::Array{T,2}, configs::Vector{Vector{Int}}, Q::Vector{Array{T}}, involv::Vector{T}) where T <: Real
        sum(levels[:,2]) <= 1.0 || error("Total mixture weight cannot exceed 1.0.")
        if typeof(Omega) == TransTens
            ord = Omega.order
        else
            ord = ndims(Omega) - 1
        end
        return new(Omega, ord, totalweight_tol, levels, configs, Q, involv)
    end
end

function Base.print(x::DecompMMTDg, io::IO=stdout, digits=5)
    y = "\n======================================================\n\n"
    y = string(y, "MMTDg decomposition of order $(x.order) transition distribution\n\n")
    y = string(y, "------------------------------------------------------\n\n")
    nrow = size(x.levels, 1)
    for i = 1:nrow
        y = string(y, "Sub-order: $(Int(x.levels[i,1])), \t Weight: $(round(x.levels[i,2], digits=digits)), \t Lags: $(x.configs[i])\n")
    end

    y = string(y, "\nTotal weight: $(round(sum(x.levels[:,2]), digits=digits))\n")
    y = string(y, "Weight tolerance: $(x.totalweight_tol)\n\n")

    y = string(y, "------------------------------------------------------\n")

    y = string(y, "\nTotal weight associated with each lag\n\n")
    for i = 1:x.order
    y = string(y, "Lag $(i): $(round(x.involv[i], digits=digits))\n")
    end

    y = string(y, "\n======================================================\n")

    print(io, y)
end

function reduce_intercept(Omeg::TransTens, leftoversum_tol=1.0e-16)

    d = Omeg.order + 1
    d > 1 || error("Argument must be a multidimensional array.")

    a0 = vec(minimum(Omeg.A, dims=2:d))
    lam = sum(a0)
    if lam > 0.0
        a = a0 ./ lam
    else
        a = a0 .* 0.0
    end

    leftover = Omeg.A .- a0
    leftover_sums = vec(sum(leftover, dims=1))
    maximum(abs.( leftover_sums .+ lam .- 1.0 )) < leftoversum_tol || error("All leftover marginal sums should be within the specified tolerance (set with leftoversum_tol).")

    return (lam, a, leftover, leftover_sums[1])
end


function reduce_gen(A::Array{T}, sub_order::Int, leftoversum_tol=1.0e-16) where T <: Real

    order = ndims(A) - 1
    d = size(A)
    K = d[1]

    sub_order > 0 && sub_order < order || order("Sub-order cannot be higher than tensor order.")
    all( d.==d[1]) || error("All dimensions of tensor must have same length.")
    # all( A .>= 0.0 ) || error("Tensor must have non-negative entries.")

    lam_indx = build_λ_indx(order, sub_order)
    n_cand = lam_indx.lens[sub_order]
    configs = lam_indx.indxs[sub_order]

    lam_cand = Vector{T}(undef, n_cand)
    take_cand = Array{T}(undef, vcat(fill(K, sub_order+1), n_cand)...)
    take_cand = Vector{Array{T}}(undef, n_cand)

    for ℓ = 1:n_cand
        Mins = minimum(A, dims= 1 .+ setdiff(1:order, configs[ℓ]))
        sums_Mins = sum(Mins, dims=1)
        lam_cand[ℓ] = minimum(sums_Mins)
        take_cand[ℓ] =  Mins ./ sums_Mins
    end

    lam_sel, lag_sel_indx = findmax(lam_cand)

    if abs(lam_sel) < 1.0e-6
        lam_sel = 0.0
        leftover = deepcopy(A)
        take = zeros(T, 1 + sub_order)
        lag_sel_indx = 0
        config_out = nothing
    else
        leftover = A .- lam_sel .* take_cand[lag_sel_indx] # do not reshpae yet; automatically broadcasts (should set up test in case broadcasting behavior changes someday)
        take = reshape(take_cand[lag_sel_indx], fill(K, 1+sub_order)...)
        config_out = configs[lag_sel_indx]
    end

    leftover_sums = vec(sum(leftover, dims=1))
    maximum(abs.( leftover_sums .- leftover_sums[1] )) < leftoversum_tol || error("All leftover marginal sums should be within specified tolerance; set with leftoversum_tol.")

    return lam_sel, (lag_sel_indx, config_out), take, leftover, leftover_sums[1]
end


function reduce_transTens(Omeg::Union{TransTens, Array{T}}; totalweight_toler=0.001, maxiter=100, leftoversum_tol=1.0e-9) where T <: Real
    if typeof(Omeg) == TransTens
        ord = Omeg.order
    else
        ord = ndims(Omeg) - 1
        d = size(Omeg)
        all( d.==d[1]) || error("All dimensions of transition tensor must have same length.")
        all( Omeg .>= 0.0 ) || error("Transition tensor must have non-negative entries.")
        maximum(abs.( sum(Omeg, dims=1) .- 1.0 )) < leftoversum_tol || error("Transition tensor must add to unity along first dimension.")
    end

    lam = [0.0]
    Q = Vector{Array{Float64}}(undef, 1)
    configs = Vector{Vector{Int}}(undef, 1)
    configs[1] = [0]
    level = [0]
    total_lag_involvement = zeros(ord)

    lam_now, Q_now, A, whatsleft = reduce_intercept(Omeg, leftoversum_tol)

    lam[1] = copy(lam_now)
    Q[1] = deepcopy(Q_now)

    iter = 1
    sub_ord_now = 1

    while sub_ord_now < ord && whatsleft >= totalweight_toler && iter < maxiter

        lam_now, lagsel_now, Q_now, A, whatsleft = reduce_gen(A, sub_ord_now, leftoversum_tol)

        if lam_now > 0
            push!(lam, lam_now)
            push!(configs, lagsel_now[2])
            push!(level, sub_ord_now)
            push!(Q, Q_now)
            total_lag_involvement[lagsel_now[2]] .+= lam_now
        else
            sub_ord_now += 1
        end

        iter += 1

    end

    if sub_ord_now == ord # means there was no more lower orders could do
        push!(lam, whatsleft)
        push!(configs, collect(1:ord))
        push!(level, sub_ord_now)
        push!(Q, A ./ whatsleft)
        total_lag_involvement .+= whatsleft
    end

    return DecompMMTDg( Omeg, totalweight_toler, hcat(level, lam), configs, Q, total_lag_involvement )

end
