function pmapreduce(mapfun, redfun, iters...)
       @distributed redfun for arg in collect(zip(iters...))
              mapfun(arg...)
       end
end 

function categorical!(df::DataFrame, x::Symbol)
	df[!, x] = categorical(df.x)
	return nothing
end 

function logsumexp(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return maxV + log(sum(exp.(V .- maxV)))
end 

function multinomial(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return  exp.(V .- maxV) ./sum(exp.(V .- maxV))
end

std_err(f::Function, x::Vector) = sqrt.(LinearAlgebra.diag(inv(ForwardDiff.hessian(f, x))))

function std_err(beta::Vector{T}, cl::clogit) where T<:Real
	J = length(beta)
	OG = zeros(J,J)
	for cld in cl.data 
		g = analytic_grad_clogit_case(beta, cld) 
		OG .+= g*g'
	end
	try
		u,s,v = svd(OG)
		return sqrt.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u'))
	catch 
		sv = svdvals(OG)
		nsv = J - sum(isapprox.(sv./sv[1],0.; atol= 1e-6))
		println("WARNING: used tsvd to approx std err with nsv=$(nsv)")
		u,s,v = tsvd(OG, nsv)
		return sqrt.(abs.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u')))		
	end
end

function std_err(theta::Vector{T}, nl::nlogit) where T<:Real
	J = length(theta)
	OG = zeros(J,J)
	for id in 1:length(nl.data)
		g = analytic_grad_nlogit_case(theta, nl, id) 
		OG .+= g*g'
	end
	try
		u,s,v = svd(OG)
		return sqrt.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u'))
	catch 
		sv = svdvals(OG)
		nsv = J - sum(isapprox.(sv./sv[1],0.; atol= 1e-6))
		println("WARNING: used tsvd to approx std err with nsv=$(nsv)")
		u,s,v = tsvd(OG, nsv)
		return sqrt.(abs.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u')))		
	end
end

TS1(Qk) = 1.0 / (1 - Qk)
TS2(Qk) = 4.0 / (3.0*(1-Qk) + sqrt( (1.0 + 7*Qk)*(1.0 - Qk) ) )
