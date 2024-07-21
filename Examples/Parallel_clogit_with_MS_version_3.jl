# EXAMPLE -- clogit demand estimation with merger simulation

#=
  The difference with this simulation is to include a price level and 
  interaction
 =#

using Distributed

addprocs(3)

@everywhere begin
	using Pkg
	Pkg.activate("./Git/GEV.jl")
	Pkg.instantiate()
end

@everywhere using GEV
using CSV, DataFrames, StatsModels, Optim, LinearAlgebra, Statistics

df = CSV.read(joinpath(@__DIR__,"./Git/GEV.jl/Examples/Data/restaurant.csv"), DataFrame);

prods = unique(df.restaurant);
J = length(prods);
prod_df = DataFrame(:restaurant => prods, :pid => Int.(collect(1:length(prods))), :owner=>[1,1,2,2,3,3,4]);
df = leftjoin(df, prod_df, on=:restaurant);

# This gives interaction, which then enters as an interaction 
df.cost0 = deepcopy(df.cost);
df.invY = 1 ./ df.income;
df.cost_div_Y = df.cost .* df.invY;

# Setup

# clogit formula
f1 = @formula( chosen ~ cost + cost&invY +distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:pid)

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

# Impose common price to check code for clogit
df0 = deepcopy(df); # Data to add a new common price to

# No interactions with individual characteristics
pos_price = 1
pos_invY = 4
P0 = getX(AggregateDemand(xstar, df0, cl, pos_price, pos_invY));

cl0 = new_clogit_data(df0, clm, P0, :cost);

# Product level output
AD = AggregateDemand(xstar, df0, cl0, pos_price, pos_invY);
Q = getQty(AD)
s = getShares(AD)
P = getX(AD)
dQdP = getdQdP(AD)
dsdP = dQdP ./ length(AD)
DR = getDiversionRatioMatrix(AD)
E = getElasticityMatrix(AD)

# ----------- POST-ESTIMATION GROUPED DEMAND SIDE OUTPUTS ------------- #

# Get group product selector matrix
firm_df = combine(groupby(df0, :pid), 
					:restaurant => unique => :brand, 
							:owner => unique => :owner)
OWN = make_ownership_matrix(firm_df, :owner)

# FIRM LEVEL
P_g = getGroupX(AD, OWN.IND)
Q_g = getGroupQty(AD, OWN.IND)
s_g = getGroupShares(AD, OWN.IND)
dQdP_g = getGroupdQdP(AD, OWN.IND)
dsdP_g = dQdP ./ length(AD)
DR_g = getGroupDiversionRatioMatrix( AD , OWN.IND)
E_g = getGroupElasticityMatrix(AD , OWN.IND)

# ----------- POST-ESTIMATION SUPPLY-SIDE OUTPUTS ------------- #

# Get Marginal Costs
MC = getMC(P,Q,dQdP,OWN.MAT) # MC = P .+ (OWN.MAT .* dQdP) \ Q

#= OR can use shares
	MC = getMC(P,s,dsdP,OWN.MAT) # MC = P .+ (OWN.MAT .* dsdP) \ s
=#

# Margins under different market structure
MARGIN_SPN = getMARGIN(Q, P, Matrix(I(J)), Matrix(I(J)), dQdP)
[MARGIN_SPN -1 ./ diag(E) isapprox.(MARGIN_SPN .- -1 ./ diag(E), 0; atol=1e-6)] # Check

# Product margin under multiproduct industry structure
MARGIN_MPN = getMARGIN(Q, P, Matrix(I(J)), OWN.MAT, dQdP)
[MARGIN_MPN  (P .-MC)./P isapprox.(MARGIN_MPN .- (P .-MC)./P , 0.; atol=1e-6)] # Check

# Check aggregate elasticity vs lerner at the age
FIRM_MARGIN = getMARGIN(Q, P, OWN.IND, OWN.MAT, dQdP)
FIRM_AS_SPN_LERNER = - 1 ./ FIRM_MARGIN # Note lerner won't match diag(E_g), aggregation across products in elasticities removes within product cross partials (they are not held fixed)
[FIRM_AS_SPN_LERNER diag(E_g)] # LERNER >= Egg because of multiproduct effect 
[FIRM_MARGIN -1 ./ diag(E_g)] # Same reason FIRM_MARGIN >= -1/Egg

# ------------------ #
# MERGER SIMULATION
# ------------------ #

# Check FOC first at pre-merger values
df1 = deepcopy(df0);

PARALLEL_FLAG = false; # Faciltates distribution of demand output calculations
FOC0(x) = FOC(zeros(J), xstar, df1, clm, MC, OWN.MAT, x, :cost, pos_price, pos_invY, PARALLEL_FLAG)

# Check
@time FOC0(P0)
isapprox.(FOC0(P0) , 0; atol=1e-6) 

using NLsolve

# Merger of 3 & 4
firm_df.post_owner = firm_df.owner 
firm_df.post_owner[firm_df.owner.==3] .= 4
POST_OWN= make_ownership_matrix(firm_df, :post_owner)

# FOC under new merger under static Bertrand-nash competition
PARALLEL_FLAG = false
FOC1(x) = FOC(zeros(J), xstar, df1, clm, 
			MC, POST_OWN.MAT, x, :cost, pos_price, pos_invY, PARALLEL_FLAG)

# Solve for post-merger prices (start from pre-merger)
post_res = nlsolve(FOC1, P0)

# Price Rise 
P1 = post_res.zero
PriceIncrease = (P1 .- P0 ) ./ P0

# Consumer Welfare Change
CW0 = getCW(AggregateDemand(xstar, df0, cl0, pos_price, pos_invY))
CW1 = getCW(AggregateDemand(xstar, df1, new_clogit_data(df1, clm, P1, :cost), pos_price, pos_invY))
CW_CHANGE = CW1/CW0 - 1
