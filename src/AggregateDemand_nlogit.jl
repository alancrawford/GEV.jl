#=
1. Create code computing demand derivative matrix with an existing set of choice sets and estimates

2. Add one that allows for a change of price
=#


# ADD IN DDQi here if hand code....

# New cl.data from previous commands
function DemandOutputs_clogit_case(beta::AbstractVector, clcd::clogit_case_data, 
						xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[]) where T<:Real
	
	# new_xvar is already in Xj -> pass it for ForwardDiff
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z= clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	for inds in all_xvars
		alpha_x_xvar .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	PROBi = multinomial(V)

	# Step 3: Get individual outputs [No Hessian Yet]
	DQi = zeros(Float64,J,J)
	for (nj,j) in enumerate(jid), (nk,k) in enumerate(jid)
		if j==k
			DQi[j,k] = alpha_x_xvar[nj]*(1 - PROBi[nj]) * PROBi[nj] / Xj[nj,xvarpos]
		else 
			DQi[j,k] = -alpha_x_xvar[nj] * PROBi[nj] * PROBi[nk] / Xj[nj,xvarpos]
		end
	end 

	# Allow for choice set hetergeneity so add indicator functions
	PRODSi = zeros(Int64,J)
	PRODSi[jid] .= 1

	return (PRODS = PRODSi, CTR = PRODSi*PRODSi', PROB=PROBi, DQ = DQi , CW = logsumexp(V))

end

# clogit input
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, 
							xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[])

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	J = maximum(df[!, cl.model.choice_id])
	CW = 0.
	PRODS = zeros(Int64, J)
	CTR = zeros(Int64, J, J)
	PROB = zeros(Float64,J)
	DQ = zeros(Float64, J, J)
	for clcd in cl.data
		Di = DemandOutputs_clogit_case(beta, clcd, xvarpos, interacted_xvarpos)
		PRODS += Di.PRODS
		CTR += Di.CTR
		PROB += Di.PROB
		DQ += Di.DQ
		CW += Di.CW
	end

	return (PROB = PROB ./ PRODS, DQ = DQ ./ CTR, CW = CW) 

end 

# Loop over all choice sets
function AggregateDemand(	beta::Vector, df::DataFrame, clm::clogit_model, 
				new_xvar::AbstractVector{T}, xvarname::Symbol, xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[]) where T

	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Write old xvar
	df[!, Symbol(:old_,xvarname)] = df[!, xvarname]

	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,clm.choice_id]]

	# New clogit data (but same model)
	cl_new = clogit( clm, make_clogit_data(clm, df))

	return AggregateDemand(beta, df, cl_new, xvarpos, interacted_xvarpos)

end 

# Aggregate Elasticities and Diversion Ratios
function AggregateDiversionRatioMatrix( DQ::Matrix , INDMAT::Matrix)
	(N,J) = size(INDMAT)
	dsj_dpj = diag(DQ)
	DR = zeros(N, N)
	for a in 1:N
		dsAdpA = sum(dsj_dpj .* INDMAT[a, :])
		for b in 1:N 
			if a!==b
				dsBdpA = sum(DQ.*(INDMAT[a,:]*INDMAT[b,:]'))
				DR[a,b] = - dsBdpA / dsAdpA 
			end
		end 
	end
	return DR
end 

function AggregateElasticityMatrix( DQ::Matrix, prob::Vector, xvar::Vector, INDMAT::Matrix)

	(N,J) = size(INDMAT)
	dsj_dxj = diag(DQ)

	E = zeros(N, N)
	for a in 1:N # xA up
		sj_in_A = prob .* INDMAT[a,:]
		sA = sum(sj_in_A) 										
		xA = sum(xvar .* (sj_in_A / sA)) 						
		E[a,a] = sum(dsj_dxj .* INDMAT[a, :]) * xA / sA
		for b in 1:N  # sB responds
			if a!==b 
				sj_in_B = prob .* INDMAT[b, :] 
				sB = sum(sj_in_B)
				E[a,b] = sum(DQ.*(INDMAT[a,:]*INDMAT[b,:]')) * xA / sB
			end
		end 
	end

	return E

end 
