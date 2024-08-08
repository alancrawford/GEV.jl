
# PostEstimation

# ---------------------------------------- #
# Getters for product level demand outputs
# ---------------------------------------- #

getX(AD::Vector{clogit_case_output}) = sum(ad.s .* ad.x for ad in AD) ./ sum(ad.s for ad in AD)

getP(AD::Vector{clogit_case_output}) = sum(ad.s .* ad.p for ad in AD) ./ sum(ad.s for ad in AD)

getQty(AD::Vector{clogit_case_output}, M::Real=1) = sum(ad.s for ad in AD)

function getShares(AD::Vector{clogit_case_output}) 
	qj = getQty(AD)
	return qj ./ sum(qj)
end

getdQdX(AD::Vector{clogit_case_output}) = sum(ad.dsdx for ad in AD)

getdQdP(AD::Vector{clogit_case_output}, PdivY::Bool=false) = PdivY ? sum(ad.dsdx .* ad.z for ad in AD) : sum(ad.dsdx for ad in AD)

function getDiversionRatioMatrix(AD::Vector{clogit_case_output})
	dQdX = getdQdX(AD)
	return -dQdX ./ diag(dQdX)
end 

function getPriceDiversionRatioMatrix(AD::Vector{clogit_case_output}, PdivY::Bool=false)
	dQdP = getdQdP(AD, PdivY)
	return -dQdP ./ diag(dQdP)
end 

getElasticityMatrix(dQdX::Matrix, Q::Vector, X::Vector) = dQdX .* X ./ Q' 

# ------------------------------------------- #
# Getters for grouped aggregate demand outputs
# ------------------------------------------- #

# INDMAT is a G x J matrix whose [g,j]-th entry is 1 if j belongs to group g and 0 otherwise

function getGroupX(AD::Vector{clogit_case_output}, INDMAT::MatSpM)
	grp_q = INDMAT*sum(ad.s for ad in AD)
	grp_qx = INDMAT*sum(ad.s .* ad.x for ad in AD)	
	return grp_qx ./ grp_q	
end

function getGroupP(AD::Vector{clogit_case_output}, INDMAT::MatSpM)
	grp_q = INDMAT*sum(ad.s for ad in AD)
	grp_rev = INDMAT*sum(ad.s .* ad.p for ad in AD)	
	return grp_rev ./ grp_q	
end

getGroupQty(AD::Vector{clogit_case_output}, INDMAT::MatSpM, M::Real=1) = INDMAT*getQty(AD, M)

getGroupShares(AD::Vector{clogit_case_output}, INDMAT::MatSpM) = INDMAT*getShares(AD)

function getGroupdQdX(AD::Vector{clogit_case_output}, INDMAT::MatSpM)  
	(G,J) = size(INDMAT)
	dQdX = getdQdX(AD)
	dQjdXj = diag(dQdX)
	grp_dQdX = zeros(G, G)
	@inbounds for a in 1:G, b in 1:G
		grp_dQdX[a,b] = sum(dQdX.*(INDMAT[a,:]*INDMAT[b,:]'))
	end
	return grp_dQdX
end 

function getGroupdQdP(AD::Vector{clogit_case_output}, INDMAT::MatSpM, PdivY::Bool=false)  
	(G,J) = size(INDMAT)
	dQdP = getdQdP(AD, PdivY)
	dQjdPj = diag(dQdP)
	grp_dQdP = zeros(G, G)
	@inbounds for a in 1:G, b in 1:G
		grp_dQdP[a,b] = sum(dQdP.*(INDMAT[a,:]*INDMAT[b,:]'))
	end
	return grp_dQdP
end 

function getGroupDiversionRatioMatrix(AD::Vector{clogit_case_output}, INDMAT::MatSpM)  
	(G,J) = size(INDMAT)
	dQdX = getdQdX(AD)
	dQjdXj = diag(dQdX)
	DR = zeros(G, G)
	@inbounds for a in 1:G
		dQAdXA = sum(dQdX.*(INDMAT[a,:]*INDMAT[a,:]'))
		for b in 1:G
			if a!==b
				dQBdXA = sum(dQdX.*(INDMAT[a,:]*INDMAT[b,:]'))
				DR[a,b] = - dQBdXA / dQAdXA 
			end
		end 
	end
	return DR
end  


function getGroupPriceDiversionRatioMatrix(AD::Vector{clogit_case_output}, INDMAT::MatSpM, PdivY::Bool=false)  
	(G,J) = size(INDMAT)
	dQdP = getdQdP(AD, PdivY)
	dQjdPj = diag(dQdP)
	DR = zeros(G, G)
	@inbounds for a in 1:G
		dQAdPA = sum(dQdP.*(INDMAT[a,:]*INDMAT[a,:]'))
		for b in 1:G
			if a!==b
				dQBdPA = sum(dQdP.*(INDMAT[a,:]*INDMAT[b,:]'))
				DR[a,b] = - dQBdPA / dQAdPA 
			end
		end 
	end
	return DR
end  


function getGroupdQdP( beta::Vector, df::DataFrame, clm::clogit_model, Q0::Vector, P0::Vector,
		xvar::Symbol, groupvar::Symbol, xvarpos::Int64, INDMAT::MatSpM, parallel::Bool=false)

	# Baseline Volume
	G = length(Q0)
	dQdP = zeros(G, G)
	grouplist = levels(df[:,groupvar])
	
	for (i,b) in enumerate(grouplist)
		
		# 1% price increase for whole brand
		df_TEMP = deepcopy(df)
		b_idx = findall(df_TEMP[!, groupvar] .== b)
		df_TEMP[b_idx, xvar] .*= 1.01

		# New cl data
		cl_TEMP = clogit( clm, make_clogit_data(clm, df_TEMP));

		# New Aggregate Demand
		AD_TEMP = AggregateDemand(beta, df_TEMP, cl_TEMP, xvarpos, parallel)

		# New Qty
		Q1 = getGroupQty(AD_TEMP, INDMAT)
		ΔQ = Q1 - Q0
		ΔP = 0.01*P0 

		# Diversion Ratio for increase in b
		dQdP[i, :] = ΔQ ./ ΔP

	end
	
	return dQdP
end

function getGroupdQdP( beta::Vector, df::DataFrame, clm::clogit_model, Q0::Vector, P0::Vector,
		xvar::Symbol, groupvar::Symbol, xvarpos::Int64, xzvarpos::ScalarOrVector, INDMAT::Union{Matrix, SparseMatrixCSC}, parallel::Bool=false)

	# Baseline Volume
	G = length(Q0)
	dQdP = zeros(G, G)
	grouplist = levels(df[:,groupvar])
	
	for (i,b) in enumerate(grouplist)
		
		# 1% price increase for whole brand
		df_TEMP = deepcopy(df)
		b_idx = findall(df_TEMP[!, groupvar] .== b)
		df_TEMP[b_idx, xvar] .*= 1.01

		# New cl data
		cl_TEMP = clogit( clm, make_clogit_data(clm, df_TEMP));

		# New Aggregate Demand
		AD_TEMP = AggregateDemand(beta, df_TEMP, cl_TEMP, xvarpos, xzvarpos, parallel)

		# New Qty
		Q1 = getGroupQty(AD_TEMP, INDMAT)
		ΔQ = Q1 - Q0
		ΔP = 0.01*P0 

		# Diversion Ratio for increase in b
		dQdP[i, :] = ΔQ ./ ΔP

	end
	
	return dQdP
end



function getGroupDiversionRatioMatrix(  beta::Vector, df::DataFrame, clm::clogit_model, Q0::Vector, 
		xvar::Symbol, groupvar::Symbol, xvarpos::Int64, INDMAT::MatSpM, parallel::Bool=false)

	# Baseline Volume
	G = length(Q0)
	DR = zeros(G, G)
	grouplist = levels(df[:,groupvar])
	
	for (i,b) in enumerate(grouplist)
		
		# 1% price increase for whole brand
		df_TEMP = deepcopy(df)
		b_idx = findall(df_TEMP[!, groupvar] .== b)
		df_TEMP[b_idx, xvar] .*= 1.01

		# New cl data
		cl_TEMP = clogit( clm, make_clogit_data(clm, df_TEMP));

		# New Aggregate Demand
		AD_TEMP = AggregateDemand(beta, df_TEMP, cl_TEMP, xvarpos, parallel)

		# New Qty
		Q1 = getGroupQty(AD_TEMP, INDMAT)
		ΔQ = Q1 - Q0

		# Diversion Ratio for increase in b
		DR[i, :] = -ΔQ / ΔQ[i]

	end

	return DR
end

function getGroupDiversionRatioMatrix(  beta::Vector, df::DataFrame, clm::clogit_model, Q0::Vector, 
		xvar::Symbol, groupvar::Symbol, xvarpos::Int64, xzvarpos::ScalarOrVector, INDMAT::MatSpM, parallel::Bool=false)

	# Baseline Volume
	G = length(Q0)
	DR = zeros(G, G)
	grouplist = levels(df[:,groupvar])
	
	for (i,b) in enumerate(grouplist)
		
		# 1% price increase for whole brand
		df_TEMP = deepcopy(df)
		b_idx = findall(df_TEMP[!, groupvar] .== b)
		df_TEMP[b_idx, xvar] .*= 1.01

		# New cl data
		cl_TEMP = clogit( clm, make_clogit_data(clm, df_TEMP));

		# New Aggregate Demand
		AD_TEMP = AggregateDemand(beta, df_TEMP, cl_TEMP, xvarpos, parallel)

		# New Qty
		Q1 = getGroupQty(AD_TEMP, INDMAT)
		ΔQ = Q1 - Q0

		# Diversion Ratio for increase in b
		DR[i, :] = -ΔQ / ΔQ[i]

	end

	return DR
end

