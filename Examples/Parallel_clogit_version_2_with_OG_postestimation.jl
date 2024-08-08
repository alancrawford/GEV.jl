# EXAMPLE -- clogit demand estimation with merger simulation

using Distributed

addprocs(3)

@everywhere begin
	using Pkg
	Pkg.activate("./Git/GEV.jl")
	Pkg.instantiate()
end

@everywhere using GEV
using CSV, DataFramesMeta, StatsModels, Optim, LinearAlgebra, Statistics , SparseArrays

df = CSV.read(joinpath(@__DIR__,"./Git/GEV.jl/Examples/Data/restaurant.csv"), DataFrame);

df[!, :include] .= 1
for row in eachrow(df)
	if row.chosen .== 0 && rand() > 0.85
		row.include = 0 
	end 
end

dftemp = unique(select(df, :family_id, :income, :kids));
dftemp[!, :restaurant] .= "zzzzzz";
dftemp[!, :cost] .= 0;
dftemp[!, :chosen] .= 0;
dftemp[!, :rating] .= 0;
dftemp[!, :distance] .= 0;
dftemp[!, :include] .= 1;

append!(df, dftemp)
sort!(df, [:family_id, :restaurant])

df = @subset(df, :include .== 1)

prods = sort!(unique(df.restaurant));
J = length(prods);
prod_df = DataFrame(:restaurant => prods, :pid => Int(3) .*Int.(collect(1:length(prods))), :owner=>[1,1,2,2,3,3,4,5]);
df = leftjoin(df, prod_df, on=:restaurant);

# This gives interaction, which then enters as an interaction 
#df.cost0 = deepcopy(df.cost);
df.invY = 1 ./ df.income;
df.cost_div_Y = df.cost .* df.invY;

# ----- #
# Setup #
# ----- #

# clogit formula
f1 = @formula( chosen ~ cost_div_Y + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:pid)
clm.opts[:PdivY] = true
clm.opts[:pvar] = :cost
clm.opts[:zvar] = :invY

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Option 1. Estimate clogit model with LBFGS() or other algorithm only requiring gradients
result = estimate_clogit(cl; opt_mode = :parallel, 	# <- Need to call :parallel here
							 opt_method = :grad,  	# <- :grad or :hess , linked to algorithm
							x_initial = randn(cl.model.nx),
							algorithm = GEV.Optim.LBFGS(), 	# <- algorithm
							optim_opts = GEV.Optim.Options(show_trace=true), # <- optim options
							batch_size = 50);   # <- Can put subset of workers i.e. [2]

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(xstar, cl)
#se = sqrt.(diag(inv(pmap_hessian_clogit(xstar, cl.data))));
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Print out results - this is working and checks out versus stata!
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"],[cl.model.coefnames xstar se])

# ----------- POST-ESTIMATION DEMAND SIDE OUTPUTS AT A NEW PRICE POINT ------------- #

# Make record of DataFrame
df0 = deepcopy(df); 

# No interactions with individual characteristics
pos_PdivY = 1 # Must be an Int()

# Aggregate Demand from raw data
AD = AggregateDemand(xstar, df0, cl, pos_PdivY);

# Number of Products Incl OG
J = maxJ(AD);

# Prices								
P0 = spgetP(AD, J);

# clogit with new price to test code
cl0 = new_clogit_data(df0, clm, Vector(P0), :cost_div_Y, :cost, :invY); 

# Aggregate at new price
AD0 = AggregateDemand(xstar, df0, cl0, pos_PdivY); 	

# ----------------------------------------------------------------------------------- #

# Product level outputs in sparse form
# -------------------------------------

Q = spgetQty(AD0, J)
s = spgetShares(AD0, J)
X = spgetX(AD0, J)
P = spgetP(AD0, J)
dQdP = spgetdQdP(AD0, J,  clm.opts[:PdivY])
dsdP = dQdP ./ length(AD)
DR = spgetPriceDiversionRatioMatrix(AD0, J,  clm.opts[:PdivY])
E = spgetElasticityMatrix(dQdP , Q, P)

# Call Inside Goods as dense Vector, Matrices with inside_good_idx as pid ref 
inside_good_idx = getInsideGoods(AD0, J) 

Q_ig = spgetQty(AD0, J, inside_good_idx)
s_ig = spgetShares(AD0, J, inside_good_idx)
X_ig = spgetX(AD0, J, inside_good_idx)
P_ig = spgetP(AD0, J, inside_good_idx)
dQdP_ig = spgetdQdP(AD0, J, inside_good_idx, clm.opts[:PdivY])
dsdP_ig = dQdP_ig ./ length(AD)
DR_ig = spgetPriceDiversionRatioMatrix(AD0, J, inside_good_idx,  clm.opts[:PdivY])
E_ig = getElasticityMatrix(dQdP_ig , Q_ig , P_ig)

# Putting OG subs on main diagonal
DR_ig .+ I(length(inside_good_idx)).*( 1 .- sum(DR_ig, dims=2))

# ----------- POST-ESTIMATION GROUPED DEMAND SIDE OUTPUTS ------------- #


# Grouped level outputs - INCL OUTSIDE GOOD
# --------------------------------------------

# Get group product selector matrix - INCLUDING OUTSIDE GOOD AS J+1 PRODUCT ID AND OUTSIDE GOOD AS N+1 OWNER ID
firm_df = combine(groupby(df0, :pid), :restaurant => unique => :brand, :owner => unique => :owner)
OWN = make_ownership_matrix(firm_df, :owner, :pid)

# FIRM LEVEL
X_g = spgetGroupX(AD0, J, OWN.IND)
P_g = spgetGroupP(AD0, J, OWN.IND)
Q_g = spgetGroupQty(AD0, J, OWN.IND)
s_g = spgetGroupShares(AD0, J, OWN.IND)

dQdP_g = spgetGroupdQdX( xstar, df0, clm, J, Q_g, P_g,  :cost_div_Y, :owner, pos_PdivY, OWN.IND, clm.opts[:PdivY]) 
AdQdP_g = spgetGroupdQdP(AD0, J, OWN.IND, clm.opts[:PdivY])

DR_g = spgetGroupDiversionRatioMatrix( xstar, df0, clm, J, Q_g, :cost_div_Y, :owner, pos_PdivY, OWN.IND, clm.opts[:PdivY]) 
ADR_g = spgetGroupPriceDiversionRatioMatrix( AD0 , J, OWN.IND, clm.opts[:PdivY])

E_g = getElasticityMatrix(dQdP_g, Q_g, P_g )
AE_g = getElasticityMatrix(AdQdP_g, Q_g, P_g )

# ----------- POST-ESTIMATION SUPPLY-SIDE OUTPUTS ------------- #

# Get Marginal Costs
# ------------------------------------------- 

# Option 1: call only inside goods
INDMAT_ig = OWN.IND[1:end-1, inside_good_idx]
OMEGA_ig = OWN.MAT[inside_good_idx, inside_good_idx]
MC_ig = getMC(P_ig, Q_ig, dQdP_ig, Matrix(OMEGA_ig))

# Check MC_ig set FOC to 0
isapprox.(Q_ig .+ (OMEGA_ig .* dQdP_ig) * (P_ig .- MC_ig), 0; atol=1e-6)

# Option 2: Call sparse inputs with index with inside good product id's and get a sparse MC output
MC_ig = spgetMC(P,Q,dQdP,OWN.MAT,inside_good_idx) # MC = P .+ (OWN.MAT .* dQdP) \ Q

# Margins under different market structure
# ------------------------------------------- 

NumProds = length(inside_good_idx)
MARGIN_SPN = getMARGIN(P_ig, Q_ig, dQdP_ig, Matrix(I(NumProds)), Matrix(I(NumProds)))
[MARGIN_SPN -1 ./ diag(E_ig) isapprox.(MARGIN_SPN .- -1 ./ diag(E_ig), 0; atol=1e-6)] # Check

MARGIN_MPN = getMARGIN(P_ig, Q_ig, dQdP_ig, Matrix(I(NumProds)), Matrix(OMEGA_ig))
[MARGIN_MPN  (P_ig .-MC_ig)./P_ig isapprox.(MARGIN_MPN .- (P_ig .-MC_ig)./P_ig, 0.; atol=1e-6)] # Check

FIRM_MARGIN = getMARGIN(P_ig, Q_ig, dQdP_ig, Matrix(INDMAT_ig), Matrix(OMEGA_ig))

# Sparse Margin Call 

MARGIN_SPN = spgetMARGIN(P, Q, dQdP, sparse(I(J)), sparse(I(J)), inside_good_idx)
[MARGIN_SPN -1 ./ diag(E_ig) isapprox.(MARGIN_SPN .- -1 ./ diag(E_ig), 0; atol=1e-6)] # Check

MARGIN_MPN = spgetMARGIN(P, Q, dQdP, sparse(I(J)), OWN.MAT, inside_good_idx)
[MARGIN_MPN  (P_ig .-MC_ig)./P_ig isapprox.(MARGIN_MPN .- (P_ig .-MC_ig)./P_ig, 0.; atol=1e-6)] # Check

# ------------------ #
# FOC
# ------------------ #

# Check FOC first at pre-merger values
df1 = deepcopy(df0);
NumInsideProds = length(inside_good_idx)
PRE_OMEGA = OWN.MAT[inside_good_idx,inside_good_idx]
PARALLEL_FLAG = false; # Faciltates distribution of demand output calculations

# First Order Condition - FOC
FOC0(x) = spFOC(zeros(NumInsideProds), xstar, df1, clm, MC_ig, PRE_OMEGA, x, J, inside_good_idx,
									:cost_div_Y, :cost, :invY, pos_PdivY, PARALLEL_FLAG)

# Check FOC condition
P0 = deepcopy(P_ig)
isapprox.(FOC0(P0) , 0; atol=1e-6) 

using NLsolve

# Pre Merger FOC solver Checks
P_init = P0 .+ randn(7)

# Use FOC
pre_res = nlsolve(FOC0, P_init)
[pre_res.zero  P0]
 
# Ben's method: MC + zeta(p) where zeta(p) = invL * (OMEGA .* Γ) * (P - MC) - invL * Q 
FPMS_FOC0(x) = spFPMS_FOC(zeros(NumInsideProds), xstar, df1, clm, MC_ig, PRE_OMEGA, x, J, inside_good_idx,
									:cost_div_Y, :cost, :invY, pos_PdivY, PARALLEL_FLAG)

pre_FPMSres = fixedpoint(FPMS_FOC0, P_init; show_trace=true, ftol=1e-2)
[pre_res.zero  pre_FPMSres.zero P0]

# ------------------ #
# MERGER SIMULATION
# ------------------ #

# Merger of 3 & 4
firm_df.post_owner = firm_df.owner 
firm_df.post_owner[firm_df.owner.==3] .= 4
POST_OWN= make_ownership_matrix(firm_df, :post_owner, :pid)
POST_OMEGA = Matrix(POST_OWN.MAT[inside_good_idx,inside_good_idx])

# FOC under new merger under static Bertrand-nash competition
PARALLEL_FLAG = false
FOC1(x) = spFOC(zeros(NumInsideProds), xstar, df1, clm, MC_ig, POST_OMEGA, x, J, inside_good_idx,
									 :cost_div_Y, :cost, :invY, pos_PdivY, PARALLEL_FLAG)

# Solve for post-merger prices (start from pre-merger)
post_res = nlsolve(FOC1, P_ig)

# Ben's method: MC + zeta(p) where zeta(p) = invL * (OMEGA .* Γ) * (P - MC) - invL * Q 
FPMS_FOC1(x) = spFPMS_FOC(zeros(NumInsideProds), xstar, df1, clm, MC_ig, POST_OMEGA, x, J, inside_good_idx,
									:cost_div_Y, :cost, :invY, pos_PdivY, PARALLEL_FLAG)

post_FPMSres = fixedpoint(FPMS_FOC1, P_ig)

[post_res.zero post_FPMSres.zero]

# Price Rise 
P1 = post_FPMSres.zero
PriceIncrease = (P1 .- P0 ) ./ P0

# ------------------ #
# CONSUMER WELFARE
# ------------------ #

P1_CWinput = sparsevec([inside_good_idx..., J], [P1..., 0])
cl1 = new_clogit_data(df1, clm, P1_CWinput, :cost_div_Y, :cost, :invY)

CW0 = getCW(AggregateDemand(xstar, df0, cl0, pos_PdivY))
CW1 = getCW(AggregateDemand(xstar, df1, cl1, pos_PdivY))
CW_CHANGE = CW1/CW0 - 1
