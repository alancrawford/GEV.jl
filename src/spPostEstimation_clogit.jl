# ---------------------------------------- #
# Sparse Getters for product level demand outputs
# ---------------------------------------- #

maxJ(AD::Vector{clogit_case_output}) = maximum(maximum(ad.jid) for ad in AD)
getInsideGoods(nzind::Vector{Int64}, J::Int64) = setdiff(nzind, J)
getInsideGoods(Q::SparseVector, J::Int64) = setdiff(Q.nzind, J)
getInsideGoods(AD::Vector{clogit_case_output}, J::Int64) = setdiff(spgetQty(AD, J).nzind, J)

# --------------------------------------------------------- #

function spgetX(AD::Vector{clogit_case_output}, J::Int64)
	num = sum(sparsevec(ad.jid, ad.s .* ad.x, J) for ad in AD) 
	denom = sum(sparsevec(ad.jid,ad.s, J) for ad in AD)
	return sparsevec(denom.nzind, num[denom.nzind] ./ denom[denom.nzind])
end
spgetX(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}) = Vector(spgetX(AD, J)[inside_good_idx])

# --------------------------------------------------------- #

function spgetP(AD::Vector{clogit_case_output}, J::Int64)
	if length(AD[1].p) .== 0
		return spgetX(AD, J)
	else 
		num = sum(sparsevec(ad.jid, ad.s .* ad.p, J) for ad in AD) 
		denom = sum(sparsevec(ad.jid,ad.s, J) for ad in AD)
		return sparsevec(denom.nzind, num[denom.nzind] ./ denom[denom.nzind])
	end
end
spgetP(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}) = Vector(spgetP(AD, J)[inside_good_idx])

# --------------------------------------------------------- #

spgetQty(AD::Vector{clogit_case_output}, J::Int64) = sum(sparsevec(ad.jid, ad.s, J) for ad in AD)
spgetQty(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}) = Vector(spgetQty(AD, J)[inside_good_idx])

# --------------------------------------------------------- #

function spgetShares(AD::Vector{clogit_case_output}, J::Int64) 
	qj = spgetQty(AD, J)
	return qj ./ sum(qj)
end
spgetShares(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}) = Vector(spgetShares(AD, J)[inside_good_idx])

# --------------------------------------------------------- #

spgetdQdX(AD::Vector{clogit_case_output}, J::Int64) =
	sum(sparse(repeat(ad.jid, 1, ad.J)[:], repeat(ad.jid, 1, ad.J)'[:], ad.dsdx[:], J, J) for ad in AD)
spgetdQdX(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}) =
	Matrix(sum(sparse(repeat(ad.jid, 1, ad.J)[:], repeat(ad.jid, 1, ad.J)'[:], ad.dsdx[:], J, J) for ad in AD)[inside_good_idx, inside_good_idx])

# --------------------------------------------------------- #

function spgetdQdP(AD::Vector{clogit_case_output}, J::Int64, PdivY::Bool=false)
	if PdivY 
		dQdP = sum( sparse( repeat(ad.jid, 1, ad.J)[:], repeat(ad.jid, 1, ad.J)'[:] , (ad.z .* ad.dsdx)[:], J, J) for ad in AD)
	else 
		dQdP = spgetdQdX(AD, J)
	end		
	return dQdP	
end

function spgetdQdP(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}, PdivY::Bool=false)
	if PdivY 
		dQdP = sum( sparse( repeat(ad.jid, 1, ad.J)[:], repeat(ad.jid, 1, ad.J)'[:] , (ad.z .* ad.dsdx)[:], J, J) for ad in AD)[inside_good_idx, inside_good_idx]
	else 
		dQdP = spgetdQdX(AD, J, inside_good_idx)
	end	
	return Matrix(dQdP)	
end

function spgetdQdP_PdivY(ad::clogit_case_output, J::Int64)
    JID = repeat(ad.jid, 1, ad.J)
    return sparse( JID[:], JID'[:] , vec(ad.z .* ad.dsdx), J, J)
end

function spgetdQdP(ad::clogit_case_output, J::Int64)
    JID = repeat(ad.jid, 1, ad.J)
    return sparse( JID[:], JID'[:] , ad.dsdx[:], J, J)
end

# --------------------------------------------------------- #

function spgetDiversionRatioMatrix(AD::Vector{clogit_case_output}, J::Int64)
	dQdX = spgetdQdX(AD, J)
	DR = spzeros(J,J)
	rows = rowvals(dQdX)
	for j in unique(rows)
		dQdXj = dQdX[j,:]
		DR[j, dQdXj.nzind] = (- dQdXj / dQdX[j,j]).nzval
	end
	return DR
end 

function spgetPriceDiversionRatioMatrix(AD::Vector{clogit_case_output}, J::Int64, PdivY::Bool=false)
	dQdP = spgetdQdP(AD, J, PdivY)
	DR = spzeros(J,J)
	rows = rowvals(dQdP)
	for j in unique(rows)
		dQdpj = dQdP[j,:]
		DR[j, dQdpj.nzind] = (- dQdpj / dQdP[j,j]).nzval
	end
	return DR
end

function spgetDiversionRatioMatrix(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64})
	dQdX = Matrix(spgetdQdX(AD, J, inside_good_idx))
	return -dQdX ./ diag(dQdX)
end  

function spgetPriceDiversionRatioMatrix(AD::Vector{clogit_case_output}, J::Int64, inside_good_idx::Vector{Int64}, PdivY::Bool=false)
	dQdP = Matrix(spgetdQdP(AD, J, inside_good_idx, PdivY))
	return -dQdP ./ diag(dQdP)
end 

spgetElasticityMatrix(dQdX::SparseMatrixCSC, Q::SparseVector, X::SparseVector) = dQdX .* X ./ Q' 

# ------------------------------------------- #
# Sparse Getters for grouped aggregate demand outputs
# ------------------------------------------- #

# INDMAT is a G x J matrix whose [g,j]-th entry is 1 if j belongs to group g and 0 otherwise

function spgetGroupX(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM)
	num = sum(sparsevec(ad.jid, ad.s .* ad.x, J) for ad in AD) 
	denom = sum(sparsevec(ad.jid, ad.s, J) for ad in AD)
	grp_q = INDMAT*denom
	grp_qx = INDMAT*num	
	return Vector(grp_qx ./ grp_q)
end

function spgetGroupP(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM)
	if length(AD[1].p) .== 0
		return spgetGroupX(AD, J, INDMAT)
	else
		num = sum(sparsevec(ad.jid, ad.s .* ad.p, J) for ad in AD) 
		denom = sum(sparsevec(ad.jid, ad.s, J) for ad in AD)
		grp_q = INDMAT*denom
		grp_rev = INDMAT*num	
		return Vector(grp_rev ./ grp_q)
	end
end

spgetGroupQty(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM) = Vector(INDMAT*spgetQty(AD, J))

spgetGroupShares(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM) = Vector(INDMAT*spgetShares(AD, J))

function spgetGroupdQdX(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM)  
	G = size(INDMAT, 1)
	sp_dQdX = spgetdQdX(AD, J)
	igidx = getInsideGoods(AD, J)
	dQdX = zeros(G, G)
	for a in 1:G, b in 1:G
		dQdX[a,b] = sum(sp_dQdX[igidx, igidx].*(INDMAT[a,igidx]*INDMAT[b,igidx]'))
	end
	return dQdX
end 

function spgetGroupdQdP(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM, PdivY::Bool=false)  
	G = size(INDMAT, 1)
	sp_dQdP = spgetdQdP(AD, J, PdivY)
	igidx = getInsideGoods(AD, J)
	dQdP = zeros(G, G)
	for a in 1:G, b in 1:G
		dQdP[a,b] = sum(sp_dQdP[igidx, igidx].*(INDMAT[a,igidx]*INDMAT[b,igidx]'))
	end
	return dQdP
end 

function spgetGroupDiversionRatioMatrix(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM)  
	G = size(INDMAT, 1)
	dQdX= spgetdQdX(AD, J)
	igidx = getInsideGoods(AD, J)
	DR = zeros(G, G)
	@inbounds for a in 1:G
		dQAdXA = sum(dQdX[igidx, igidx].*(INDMAT[a,igidx]*INDMAT[a,igidx]'))
		for b in 1:G
			if a!==b
				dQBdXA = sum(dQdX[igidx, igidx].*(INDMAT[a,igidx]*INDMAT[b,igidx]'))
				DR[a,b] = - dQBdXA / dQAdXA 
			end
		end 
	end
	return DR
end 

function spgetGroupPriceDiversionRatioMatrix(AD::Vector{clogit_case_output}, J::Int64, INDMAT::MatSpM, PdivY::Bool=false)  
	G = size(INDMAT, 1)
	dQdP = spgetdQdP(AD, J, PdivY)
	igidx = getInsideGoods(AD, J)
	DR = zeros(G, G)
	@inbounds for a in 1:G
		dQAdPA = sum(dQdP[igidx, igidx].*(INDMAT[a,igidx]*INDMAT[a,igidx]'))
		for b in 1:G
			if a!==b
				dQBdPA = sum(dQdP[igidx, igidx].*(INDMAT[a,igidx]*INDMAT[b,igidx]'))
				DR[a,b] = - dQBdPA / dQAdPA 
			end
		end 
	end
	return DR
end 

# SIMULATION METHODS

function spgetGroupdQdX( beta::Vector, df::DataFrame, clm::clogit_model, J::Int64, Q0::Vector, X0::Vector,
		xvar::Symbol, groupvar::Symbol, xvarpos::Int64, INDMAT::MatSpM, parallel::Bool=false)

	# Baseline Volume
	G = length(Q0)
	dQdX = zeros(G, G)
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
		Q1 = spgetGroupQty(AD_TEMP, J, INDMAT)
		ΔQ = Q1 - Q0
		ΔX = 0.01*X0


		# Diversion Ratio for increase in b
		dQdX[i, :] = ΔQ ./ ΔX

	end
	
	return dQdX
end

function spgetGroupdQdX( beta::Vector, df::DataFrame, clm::clogit_model, J::Int64, Q0::Vector, X0::Vector,
		xvar::Symbol, groupvar::Symbol, xvarpos::Int64, xzvarpos::ScalarOrVector, INDMAT::MatSpM, parallel::Bool=false)

	# Baseline Volume
	G = length(Q0)
	dQdX = zeros(G, G)
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
		Q1 = spgetGroupQty(AD_TEMP, J, INDMAT)
		ΔQ = Q1 - Q0
		ΔX = 0.01*X0

		# Diversion Ratio for increase in b
		dQdX[i, :] = ΔQ ./ ΔX

	end
	
	return dQdX
end


function spgetGroupDiversionRatioMatrix(  beta::Vector, df::DataFrame, clm::clogit_model, J::Int64, Q0::Vector,
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
		Q1 = spgetGroupQty(AD_TEMP, J, INDMAT)
		ΔQ = Q1 - Q0

		# Diversion Ratio for increase in b
		DR[i, :] = -ΔQ / ΔQ[i]

	end

	return DR
end

function spgetGroupDiversionRatioMatrix(  beta::Vector, df::DataFrame, clm::clogit_model, J::Int64, Q0::Vector,
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
		AD_TEMP = AggregateDemand(beta, df_TEMP, cl_TEMP, xvarpos, xzvarpos, parallel)

		# New Qty
		Q1 = spgetGroupQty(AD_TEMP, J, INDMAT)
		ΔQ = Q1 - Q0

		# Diversion Ratio for increase in b
		DR[i, :] = -ΔQ / ΔQ[i]

	end

	return DR
end

