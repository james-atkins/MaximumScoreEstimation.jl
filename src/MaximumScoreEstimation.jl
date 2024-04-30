module MaximumScoreEstimation

using LinearAlgebra: Symmetric
using StatsBase: mean
using StatsFuns: normcdf, normpdf


include("smoothed.jl")
export SmoothedMaximumScore, SmoothingFunction, NORMAL_SMOOTHING, objective, gradient, hessian

end # module MaximumScoreEstimation
