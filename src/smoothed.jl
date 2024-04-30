struct SmoothingFunction{F, F2, F3}
    h::Int
    func::F
    deriv_1::F2
    deriv_2::F3
end


const NORMAL_SMOOTHING = SmoothingFunction(2, normcdf, normpdf, z -> -z * normpdf(z))


struct SmoothedMaximumScore
    N::Int64
    K::Int64
    sgn_y::Vector{Int8}  # (2y-1)
    X::Matrix{Float64}
    smoothing::SmoothingFunction

    function SmoothedMaximumScore(X::AbstractMatrix, y::AbstractVector{Bool}, smoothing::SmoothingFunction)
        if size(X, 1) != length(y)
            error("X and y must have the same number of observations")
        end

        if size(X, 2) == 1
            error("X must have more than one covariates")
        end

        N, K = size(X)
        sgn_y = ifelse.(y, Int8(1), Int8(-1))

        new(N, K, sgn_y, X, smoothing)
    end
end



function objective(sms::SmoothedMaximumScore, sigma, b)
    return mean(sms.sgn_y .* sms.smoothing.func.(sms.X * b / sigma))
end


function gradient!(sms::SmoothedMaximumScore, sigma, b, gradient)
    for idx in eachindex(gradient, axes(sms.X, 2))
        gradient[idx] = mean(sms.X[:, idx] .* sms.sgn_y .* sms.smoothing.deriv_1.(sms.X * b / sigma)) / sigma
    end
end


# Fill the upper triangular part of `hessian` with the Hessian.
function hessian_utri!(sms::SmoothedMaximumScore, sigma, b, hessian)
    for j = 1:sms.K
        for i = 1:j
            hessian[i, j] =
                mean(sms.X[:, i] .* sms.X[:, j] .* sms.sgn_y .* sms.smoothing.deriv_2.(sms.X * b / sigma)) / sigma^2
        end
    end
end


function gradient(sms::SmoothedMaximumScore, sigma, b)
    gradient = Vector{Float64}(undef, sms.K)
    gradient!(sms, sigma, b, gradient)
    return gradient
end


function hessian(sms::SmoothedMaximumScore, sigma, b)
    hessian = Matrix{Float64}(undef, sms.K, sms.K)
    hessian_utri!(sms, sigma, b, hessian)
    return Symmetric(hessian)
end


function A_hat(sms::SmoothedMaximumScore, b::AbstractVector, sigma_star)
    @assert length(b) == sms.K
    return sigma_star^(-sms.smoothing.h) * gradient(sms, sigma_star, b)[2:end]
end


# Small sample correction for A_hat defined in Section E
function A_hat(sms::SmoothedMaximumScore, b::AbstractVector, sigma, sigma_star, lambda)
    @assert length(b) == sms.K
    uncorrected = A_hat(sms, b, sigma_star)

    return uncorrected / (1 - (sms.N / lambda * sigma * sigma_star^(2 * sms.smoothing.h))^(-1 / 2))
end


function D_hat(sms::SmoothedMaximumScore, b::AbstractVector, sigma)
    @assert length(b) == sms.K

    t = Matrix{Float64}(undef, sms.K - 1, sms.N)
    for i = 1:sms.N
        t[:, i] = sms.sgn_y[i] * (sms.X[i, 2:end] / sigma) * sms.smoothing.deriv_1(sms.X[i, :]' * b / sigma)
    end

    return sigma / sms.N * t * t'
end
