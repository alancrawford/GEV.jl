
# New clogit data with updated X inputs
function new_clogit_data(df::DataFrame, clm::clogit_model, x::Vector{Float64}, xvar::Symbol)
	
	# Copy old xvar - dataframe modified in place
	df[!, Symbol(:old_,xvar)] = df[!, xvar]
	# Insert new xvar - dataframe modified in place
	df[!, xvar] = x[df[!,clm.choice_id]]
	# New clogit data (but same model)
	return clogit( clm, make_clogit_data(clm, df))

end 

# New point special point
function new_clogit_data(df::DataFrame, clm::clogit_model, p::Vector{Float64}, xvar::Symbol, pvar::Symbol, zvar::Symbol)
	
	# Old Price
	df[!, Symbol(:old_,pvar)] = df[!, pvar]
	# Old xvar
	df[!, Symbol(:old_,xvar)] = deepcopy(df[!, xvar])
	# New x 
	df[!, pvar] = p[df[!,clm.choice_id]]
	# New Level with interaction
	df[!, xvar] = df[!, pvar] .* df[!, zvar];
	return clogit( clm, make_clogit_data(clm, df))

end 

# Sparse methods

# New clogit data with updated X inputs
function new_clogit_data(df::DataFrame, clm::clogit_model, x::SparseVector, xvar::Symbol)
	
	# Copy old xvar - dataframe modified in place
	df[!, Symbol(:old_,xvar)] = df[!, xvar]
	# Insert new xvar - dataframe modified in place
	df[!, xvar] = x[df[!,clm.choice_id]]
	# New clogit data (but same model)
	return clogit( clm, make_clogit_data(clm, df))

end 

# New point special point
function new_clogit_data(df::DataFrame, clm::clogit_model, p::SparseVector, xvar::Symbol, pvar::Symbol, zvar::Symbol)
	
	# Old Price
	df[!, Symbol(:old_,pvar)] = df[!, pvar]
	# Old xvar
	df[!, Symbol(:old_,xvar)] = deepcopy(df[!, xvar])
	# New x 
	df[!, pvar] = p[df[!,clm.choice_id]]
	# New Level with interaction
	df[!, xvar] = df[!, pvar] .* df[!, zvar];
	return clogit( clm, make_clogit_data(clm, df))

end 

# -------------------------------- #
# Demand Outputs for an individual
# -------------------------------- #

# Get individual demand output from their choice set
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, xvarpos::Int64)
	
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z = clcd
	
	jid = convert.(Int64, jid)

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	alpha_x_xvar .+= Xj[:,xvarpos]*beta[xvarpos]

	# Step 2: Get prob purchase
	V = Xj*beta
	s = multinomial(V)
	cw = logsumexp(V)

	temp_own = alpha_x_xvar .* (1 .- s) .* s  ./ Xj[:, xvarpos]
	temp_cross = -alpha_x_xvar .* s ./ Xj[:, xvarpos]
	dsdx = (repeat(temp_cross, 1, J) .* repeat(s, 1, J)') .* (1 .- I(J)) .+ diagm(temp_own)

	if pvar == zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, Float64[], Float64[])
	elseif  zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, Float64[])
	else 
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, z)
	end 
end


# Get individual demand output from their choice set with interactions in xvar passed in xvarpos
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, xvarpos::Int64, xzvarpos::Int64)
	
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z = clcd
	
	jid = convert.(Int64, jid)

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	allvars = vcat(xvarpos, xzvarpos)
	for inds in allvars
		alpha_x_xvar .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	s = multinomial(V)
	cw = logsumexp(V)

	temp_own = alpha_x_xvar .* (1 .- s) .* s  ./ Xj[:, xvarpos]
	temp_cross = -alpha_x_xvar .* s ./ Xj[:, xvarpos]
	dsdx = (repeat(temp_cross, 1, J) .* repeat(s, 1, J)') .* (1 .- I(J)) .+ diagm(temp_own)

	if pvar == zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, Float64[], Float64[])
	elseif  zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, Float64[])
	else 
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, z)
	end 
end

# Get individual demand output from their choice set with interactions in xvar passed in xvarpos
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, xvarpos::Int64, xzvarpos::Vector{Int64})
	
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z = clcd
	
	jid = convert.(Int64, jid)

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	allvars = vcat(xvarpos, xzvarpos)
	for inds in allvars
		alpha_x_xvar .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	s = multinomial(V)
	cw = logsumexp(V)

	temp_own = alpha_x_xvar .* (1 .- s) .* s  ./ Xj[:, xvarpos]
	temp_cross = -alpha_x_xvar .* s ./ Xj[:, xvarpos]
	dsdx = (repeat(temp_cross, 1, J) .* repeat(s, 1, J)') .* (1 .- I(J)) .+ diagm(temp_own)

	if pvar == zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, Float64[], Float64[])
	elseif  zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, Float64[])
	else 
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, z)
	end 
end

# ------------------------------------------- #
# Aggregate Demand Outputs for all individuals
# ------------------------------------------- #

# Loop over all individuals to create aggregate demand compononts
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, xvarpos::Int64, parallel::Bool=false)

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	AD = parallel ?
		pmap(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos), cl.data) : 
		map(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos), cl.data);

	return AD

end

# Loop over all individuals to create aggregate demand compononts
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, xvarpos::Int64, xzvarpos::Int64, parallel::Bool=false)

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	AD = parallel ?
		pmap(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data) : 
		map(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data);

	return AD

end

# Loop over all individuals to create aggregate demand compononts
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, xvarpos::Int64, xzvarpos::Vector{Int64}, parallel::Bool=false)

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	AD = parallel ?
		pmap(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data) : 
		map(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data);

	return AD

end

# Get aggregate consumer welfare
getCW(AD::Vector{clogit_case_output}) = sum(ad.cw for ad in AD)
