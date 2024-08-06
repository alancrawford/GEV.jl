# ---------------------------------------------------- #
# 					 Sparse Methods					   #
# ---------------------------------------------------- #

# Make Sparse Ownership Matrix
function make_ownership_matrix(df::DataFrame, groupvar::Symbol, pidvar::Symbol)
	N = maximum(df[!, groupvar])
	J = maximum(df[!, pidvar])
	indmat = spzeros(N,J)
	for row in eachrow(df)
		indmat[row[groupvar], row[pidvar]] = 1
	end
	return ( IND = indmat, MAT = indmat'*indmat)
end

#------------------------ Sparse MC & Margin ------------------------ #

function spgetMC(P::SparseVector, Q::SparseVector, dQdP::SparseMatrixCSC, OMAT::SparseMatrixCSC, inside_good_idx::Vector)
	Δ = Matrix(OMAT .* dQdP)[inside_good_idx, inside_good_idx] 
	return Vector(P[inside_good_idx]) .+ Δ \ Vector(Q[inside_good_idx])
end 

function spgetMARGIN(P::SparseVector, Q::SparseVector, dQdP::SparseMatrixCSC, IMAT::SparseMatrixCSC, OMAT::SparseMatrixCSC, inside_good_idx::Vector)
	Δ = Matrix(OMAT .* dQdP)[inside_good_idx, inside_good_idx] 
	FIRM_QTY = Matrix(IMAT .* Q')[:, inside_good_idx]
	PROFIT = -FIRM_QTY*(Δ\ Vector(Q[inside_good_idx])) 
	REVENUE =  FIRM_QTY*Vector(P[inside_good_idx])
	MARGIN = PROFIT ./ REVENUE
	return MARGIN
end 

# --------------- FOC: Sparse Price Inputs ----------------- #

# No interactions
function spFOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::MatSpM, P, J::Int64, ig::Vector,
				 Pvarname::Symbol, Pvarpos::Int64, parallel::Bool=false)

	Pinput = sparsevec([ig..., J], [P..., 0])
	
	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, Pinput, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, parallel)

	Q = spgetQty(AD, J, ig)

	if !haskey(clm.opts, :PdivY)
		clm.opts[:PdivY] = false 
	end 
	if parallel
		sparse_dQdP = clm.opts[:PdivY] ? 
			pmapreduce(x->spgetdQdP_PdivY(x, J), +, AD) : pmapreduce(x->spgetdQdP(x, J), +, AD)	
	else 
		sparse_dQdP = clm.opts[:PdivY] ? 
			mapreduce(x->spgetdQdP_PdivY(x, J), +, AD) : mapreduce(x->spgetdQdP(x, J), +, AD)	
	end 
	dQdP = sparse_dQdP[ig,ig]

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

# Mask to allow interactions
function spFOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::MatSpM, P, J::Int64, ig::Vector,
				 Pvarname::Symbol, Pvarpos::Int64, PZvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	Pinput = sparsevec([ig..., J], [P..., 0])

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, Pinput, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, PZvarpos, parallel)

	Q = spgetQty(AD, J, ig)
	
	if !haskey(clm.opts, :PdivY)
		clm.opts[:PdivY] = false 
	end 
	if parallel
		sparse_dQdP = clm.opts[:PdivY] ? 
			pmapreduce(x->spgetdQdP_PdivY(x, J), +, AD) : pmapreduce(x->spgetdQdP(x, J), +, AD)	
	else 
		sparse_dQdP = clm.opts[:PdivY] ? 
			mapreduce(x->spgetdQdP_PdivY(x, J), +, AD) : mapreduce(x->spgetdQdP(x, J), +, AD)	
	end 
	dQdP = sparse_dQdP[ig,ig]

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

# Masks to allow for P/Y FOC with sparse price vector
function spFOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::MatSpM, P, J::Int64, ig::Vector,
				  xvar::Symbol, pvar::Symbol, zvar::Symbol, xvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	Pinput = sparsevec([ig..., J], [P..., 0])

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, Pinput, xvar, pvar, zvar )

	# Aggregate Demand	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, xvarpos, parallel)


	Q = spgetQty(AD, J, ig)

	if !haskey(clm.opts, :PdivY)
		clm.opts[:PdivY] = false 
	end 
	if parallel
		sparse_dQdP = clm.opts[:PdivY] ? 
			pmapreduce(x->spgetdQdP_PdivY(x, J), +, AD) : pmapreduce(x->spgetdQdP(x, J), +, AD)	
	else 
		sparse_dQdP = clm.opts[:PdivY] ? 
			mapreduce(x->spgetdQdP_PdivY(x, J), +, AD) : mapreduce(x->spgetdQdP(x, J), +, AD)	
	end 
	dQdP = sparse_dQdP[ig,ig]

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end


# --------------- BEN SKRAINKA'S / MS CONTRACTION MAPPING WTIH SAME FIXED POINT AS FOC: SPARSE PRICE IN CS SETS ----------------- #

struct mscomp 
	dQdP :: SparseMatrixCSC
	Lambda :: SparseMatrixCSC
	Gamma :: SparseMatrixCSC
end

function spgetMScomponents_PdivY(ad::clogit_case_output, J::Int64)

	JID = repeat(ad.jid, 1, ad.J)
	dQdP = sparse( JID[:], JID'[:] , (ad.z .* ad.dsdx)[:], J, J) 
	Λ = sparsevec(ad.jid, diag(dQdP[ad.jid,ad.jid]).nzval ./ ( 1 .- ad.s) , J)
	Γ = dQdP .* ( 1 .- I(J) )  .+ spdiagm( diag(dQdP) .- Λ)

	return mscomp(dQdP, Λ, Γ)
end

function spgetMScomponents(ad::clogit_case_output, J::Int64)

	JID = repeat(ad.jid, 1, ad.J)
	dQdP = sparse( JID[:], JID'[:] , ad.dsdx[:], J, J) 
	Λ = sparsevec(ad.jid, diag(dQdP[ad.jid,ad.jid]).nzval ./ ( 1 .- ad.s) , J)
	Γ = dQdP .* ( 1 .- I(J) )  .+ spdiagm( diag(dQdP) .- Λ)

	return mscomp(dQdP, Λ, Γ)
	
end

function spgetMScomponents_PdivY(AD::Vector{clogit_case_output}, J::Int64, ig::Vector{Int64}, parallel::Bool=false) 

	ms = parallel ? pmap(x->spgetMScomponents_PdivY(x, J), AD) : map(x->spgetMScomponents_PdivY(x, J), AD)

	Δ = mapreduce(x->x.dQdP, +, ms)[ig, ig]
	Λ = spdiagm(mapreduce(x->x.Lambda[ig], +, ms))
	Γ = -mapreduce(x->x.Gamma, +, ms)[ig, ig]

	return ( Delta = Δ, Lambda = Λ , Gamma = Γ)

end

function spgetMScomponents(AD::Vector{clogit_case_output}, J::Int64, ig::Vector{Int64}, parallel::Bool=false) 

	ms = parallel ? pmap(x->spgetMScomponents(x, J), AD) : map(x->spgetMScomponents(x, J), AD)

	Δ = mapreduce(x->x.dQdP, +, ms)[ig, ig]
	Λ = spdiagm(mapreduce(x->x.Lambda[ig], +, ms))
	Γ = -mapreduce(x->x.Gamma, +, ms)[ig, ig]

	return ( Delta = Δ, Lambda = Λ , Gamma = Γ)

end



# No interactions
function spFPMS_FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::MatSpM, P,  J::Int64, ig::Vector,
				 Pvarname::Symbol, Pvarpos::Int64, parallel::Bool=false)

	Pinput = sparsevec([ig..., J], [P..., 0])
	
	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, Pinput, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, parallel)

	Q = spgetQty(AD, J, ig)

	if !haskey(clm.opts, :PdivY)
		clm.opts[:PdivY] = false 
	end 
	
	MS = clm.opts[:PdivY] ? spgetMScomponents_PdivY(AD, J, ig, parallel) : spgetMScomponents(AD, J, ig, parallel) 

	invL =  spdiagm(1 ./ diag(MS.Lambda))
	G =  OMEGA .* MS.Gamma

	# FOC
	F =  MC + invL * G * (P .- MC) - invL * Q

end

# Mask to allow interactions
function spFPMS_FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::MatSpM, P, J::Int64, ig::Vector,
				 Pvarname::Symbol, Pvarpos::Int64, PZvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	Pinput = sparsevec([ig..., J], [P..., 0])

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, Pinput, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, PZvarpos, parallel)

	Q = spgetQty(AD, J, ig)

	if !haskey(clm.opts, :PdivY)
		clm.opts[:PdivY] = false 
	end 
	
	MS = clm.opts[:PdivY] ? spgetMScomponents_PdivY(AD, J, ig, parallel) : spgetMScomponents(AD, J, ig, parallel) 

	invL =  spdiagm(1 ./ diag(MS.Lambda))
	G =  OMEGA .* MS.Gamma

	# FOC
	F =  MC + invL * G * (P .- MC) - invL * Q
	
end


# Masks to allow for P/Y FOC with sparse price vector
function spFPMS_FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::MatSpM, P, J::Int64, ig::Vector,
				  xvar::Symbol, pvar::Symbol, zvar::Symbol, xvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	Pinput = sparsevec([ig..., J], [P..., 0])

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, Pinput, xvar, pvar, zvar )

	# Aggregate Demand	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, xvarpos, parallel)

	Q = spgetQty(AD, J, ig)

	if !haskey(clm.opts, :PdivY)
		clm.opts[:PdivY] = false 
	end 
	
	MS = clm.opts[:PdivY] ? spgetMScomponents_PdivY(AD, J, ig, parallel) : spgetMScomponents(AD, J, ig, parallel) 

	invL =  spdiagm(1 ./ diag(MS.Lambda))
	G =  OMEGA .* MS.Gamma

	# FOC
	F =  MC + invL * G * (P .- MC) - invL * Q
end

	