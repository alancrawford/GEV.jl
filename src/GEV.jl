module GEV

using DataFrames, StatsModels, ForwardDiff, Parameters, LinearAlgebra, Optim, TSVD, ParallelDataTransfer, CategoricalArrays
using StatsBase, Statistics, SparseArrays

using Distributed

VV{T} = Vector{Vector{T}}
ScalarOrVector{T} = Union{T,Vector{T}}
VSpV{T} = Union{Vector{T},SparseVector}
MatSpM{T} = Union{Matrix{T},SparseMatrixCSC}

include("Types.jl")
include("clogit.jl")
include("nlogit.jl")
include("Estimation_clogit.jl")
include("Estimation_nlogit.jl")
include("utils.jl")
include("Elasticities_clogit.jl")
include("Elasticities_nlogit.jl")
include("AggregateDemand_clogit.jl")
include("PostEstimation_clogit.jl")
include("spPostEstimation_clogit.jl")
include("SupplySide_BertrandNash_clogit.jl")
include("spSupplySide_BertrandNash_clogit.jl")

export clogit, clogit_model, clogit_param, clogit_case, 
	   clogit_data, clogit_case_data, make_clogit_data, 
	   clogit_case_output,
	   ll_clogit, ll_clogit_case, 
	   grad_clogit_case, analytic_grad_clogit_case, 
	   fg_clogit_case, analytic_fg_clogit_case, 
	   hessian_clogit_case, pmap_hessian_clogit, 
	   fgh_clogit_case, 
	   estimate_clogit,
	   nlogit, nlogit_model, nlogit_param, nlogit_case,
	   nlogit_nest_data, nlogit_data, make_nlogit_data,
	   ll_nlogit, ll_nlogit_case, 
	   grad_nlogit_case, analytic_grad_nlogit_case, 
	   fg_nlogit_case, analytic_fg_nlogit_case, 
	   hessian_nlogit_case, pmap_hessian_nlogit, 
	   fgh_nlogit_case, 	   
	   estimate_nlogit,
	   logsumexp, fun_RUM, get_vec_dict, passdata, distdata,
	   vec_to_theta!, theta_to_vec!, std_err, @formula, multinomial,
	   clogit_prob, grad_clogit_prob,
	   elas_own_clogit, grad_elas_own_clogit,
	   elas_cross_clogit, grad_elas_cross_clogit,
	   nlogit_prob, grad_nlogit_prob, fg_nlogit_prob,
	   elas_own_nlogit, grad_elas_own_nlogit,
	   elas_within_nlogit, grad_elas_within_nlogit,
	   elas_across_nlogit, grad_elas_across_nlogit,
	   TS1, TS2, 
	   new_clogit_data, DemandOutputs_clogit_case, AggregateDemand, getCW, 
	   getX, getP, getQty, getShares, getdQdX, getdQdP, getDiversionRatioMatrix, getElasticityMatrix, 
	   maxJ, getInsideGoods,
	   spgetX, spgetP, spgetQty, spgetShares, spgetdQdP, spgetDiversionRatioMatrix, spgetElasticityMatrix, 
	   getGroupX, getGroupP, getGroupQty, getGroupShares, getApproxGroupdQdP, getApproxGroupDiversionRatioMatrix, getGroupdQdP, getGroupDiversionRatioMatrix, 
	   spgetGroupX, spgetGroupP, spgetGroupQty, spgetGroupShares, spgetApproxGroupdQdP, spgetApproxGroupDiversionRatioMatrix, spgetGroupdQdP, spgetGroupDiversionRatioMatrix, 
	   make_ownership_matrix, 
	   getMC, getMARGIN, FOC, MS_FOC, FPMS_FOC, 
	   spgetMC, spgetMARGIN, spFOC, spMS_FOC, spFPMS_FOC
	  

end 
