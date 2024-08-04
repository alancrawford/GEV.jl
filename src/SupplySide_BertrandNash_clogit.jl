
# ----------- OWNERSHIP MATRICES --------------- #

function make_ownership_matrix(df::DataFrame, groupvar::Symbol)
	indmat = StatsBase.indicatormat(df[!, groupvar])
	return ( IND = indmat, MAT = indmat'*indmat)
end

# ------------------- MC & MARGIN --------------- #

getMC(P::Vector, Q::Vector, dQdP::Matrix, OWN::Matrix) = P + (OWN .* dQdP)\Q

function getMARGIN( P::Vector, Q::Vector, dQdP::Matrix, IND::Matrix, OWN::Matrix)
	FIRM_QTY = IND .* Q'
	PROFIT = -FIRM_QTY*((OWN.*dQdP)\Q) 
	REVENUE =  FIRM_QTY*P
	return PROFIT ./ REVENUE;
end 

# --------------- FOC: IN CS SETS ----------------- #

# No interactions
function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix, P,
				 Pvarname::Symbol, Pvarpos::Int64, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, parallel)

	Q = getQty(AD)
	dQdP = getdQdP(AD)

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

# Mask to allow interactions
function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix, P,
				 Pvarname::Symbol, Pvarpos::Int64, PZvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, PZvarpos, parallel)

	Q = getQty(AD)
	dQdP = getdQdP(AD)

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

# Masks to allow for P/Y FOC 
function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix, P, 
				  xvar::Symbol, pvar::Symbol, zvar::Symbol, xvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, xvar, pvar, zvar)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, xvarpos, parallel)

	Q = getQty(AD)

	dQdP = !haskey(clm.opts, :PdivY) ? getdQdP(AD, true) : getdQdP(AD, clm.opts[:PdivY])

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

# Only implement MC/ Skrainka contraction for sparse call

function getMScomponents(AD::Vector{clogit_case_output}, PdivY::Bool=false) 
	
	if PdivY 
		dQdP_vec = [ad.z .* ad.dsdx for ad in AD]
		Λ_vec = [ diag(dQdP_vec[i]) ./ (1 .- ad.s) for (i,ad) in enumerate(AD)]
		Γ_vec = [ dQdP_vec[i] .* ( 1 .- I(ad.J) )  .+ diagm( diag(dQdP_vec[i]) .- Λ_vec[i]) for (i,ad) in enumerate(AD)]
	else 
		dQdP_vec = [ad.dsdx for ad in AD]
		Λ_vec = [ diag(dQdP_vec[i]) ./ (1 .- ad.s) for (i,ad) in enumerate(AD)]
		Γ_vec = [ dQdP_vec[i] .* ( 1 .- I(ad.J) )  .+ diagm( diag(dQdP_vec[i]) .- Λ_vec[i]) for (i,ad) in enumerate(AD)]
	end 

	Δ = sum(dQdP_vec)
	Λ =  diagm(sum(Λ_vec))
	Γ = -sum(Γ_vec)

	return ( Delta = Δ, Lambda = Λ , Gamma = Γ)

end

# No interactions
function FPMS_FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix, P,
				 Pvarname::Symbol, Pvarpos::Int64, parallel::Bool=false)
	
	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, parallel)

	Q = getQty(AD)
	MS = getMScomponents(AD) 

	invL =  diagm(1 ./ diag(MS.Lambda))
	G =  OMEGA .* MS.Gamma

	# FOC
	F =  MC + invL * G * (P .- MC) - invL * Q

end

# Mask to allow interactions
function FPMS_FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix, P,
				 Pvarname::Symbol, Pvarpos::Int64, PZvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, PZvarpos, parallel)

	Q = getQty(AD)
	MS = getMScomponents(AD) 

	invL =  diagm(1 ./ diag(MS.Lambda))
	G =  OMEGA .* MS.Gamma

	# FOC
	F =  MC + invL * G * (P .- MC) - invL * Q
	
end

# Masks to allow for P/Y FOC with sparse price vector
function FPMS_FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix, P, 
				  xvar::Symbol, pvar::Symbol, zvar::Symbol, xvarpos::ScalarOrVector{Int64}, parallel::Bool=false)


	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, xvar, pvar, zvar)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, xvarpos, parallel)

	Q = getQty(AD)
	MS = !haskey(clm.opts, :PdivY) ? getMScomponents(AD) : getMScomponents(AD, clm.opts[:PdivY])

	invL =  diagm(1 ./ diag(MS.Lambda))
	G =  OMEGA .* MS.Gamma

	# FOC
	F =  MC + invL * G * (P .- MC) - invL * Q
end

