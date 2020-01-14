""" # The main citation here is:
Averbuch, A., Shabat, G., & Shkolnisky, Y. (2016).
Direct Inversion of the Three-Dimensional Pseudo-polar Fourier Transform.
SIAM Journal on Scientific Computing, 38(2), A1100–A1120.
https://doi.org/10.1137/15m1031916
"""

using ToeplitzMatrices, OffsetArrays, FFTW

function DirectInversion3DPPFT(ÎΩppx, ÎΩppy, ÎΩppz)
    # This is Algorithm 1 of Averbuch et al. (2016)
    qnp1,np1,foo = size(ÎΩppx)
    @assert foo == np1
    q = qnp1 / np1
    n = np1 - 1
    qn = qnp1 - 1
    # Algorithm line 3
    # FIXME: Check that Complex is needed...
    ÎD = zeros(::Complex,(np1,np1,np1))
    # Algorithm line 4
    for j = 1:(n/2)
        # The x direction
        # Algorithm line 5
        # FIXME: make sure that .= is correct semantics for <- in the algorithm...
        ÎD[j,:,:] .= recover2d(ÎΩppx[q*(j-1)+1,:,:],ÎD[j,:,:],j-1,q=q)
        # Algorithm line 6
        ÎD[n+2-j,:,:] .= recover2d(ÎΩppx[qn - q*(j-1) + 1,np1:-1:1,np1:-1:1],ÎD[n+2-j,:,:],j-1,q=q)

        # The y direction
        # Algorithm line 7
        ÎD[:,j,:] .= recover2d(ÎΩppy[q*(j-1)+1,:,:],ÎD[:,j,:],j-1,q=q)
        # Algorithm line 8
        ÎD[:,n+2-j,:] .= recover2d(ÎΩppy[qn-q*(j-1)+1,np1:-1:1,np1:-1:1],ÎD[:,n+2-j,j-1],q=q)

        # The z direction
        # Algorithm line 9
        # FIXME: Check Julia precedence of + and *...
        ÎD[:,:,j] .= recover2d(ÎΩppz[q*(j-1)+1,:,:],ÎD[:,:,j],j-1,q=q)
        # Algorithm line 10
        ÎD[:,:,n+2-j] .= recover2d(ÎΩppz[qn-q*(j-1)+1,np1:-1:1,np1:-1:1],ÎD[:,:,n+2-j],j-1,q=q)
    # Algorithm line 11
    end

    # Center point
    # Algorithm line 12
    ÎD[(n/2)+1,(n/2)+1,(n/2)+1] .= ÎΩppx[(3*n/2)+1,(n/2)+1,(n/2)+1]

    # Algorithm line 13; the return value...
    return InvDecimatedFreq(ÎD,q=q,algo="JuliaToeplitz")
end

function InvDecimatedFreq(Ĩ;q,algo="JuliaToeplitz")
	np1,foo,bar = size(Ĩ)
	@assert np1 == foo
	@assert np1 == bar
	n = np1 - 1
	m = q*n + 1
	n_by_2 = n ÷ 2
	# Build the FD_star operator, a n-1 × n matrix that eventually multiplies a n×1 column vector
	FD_star = OffsetArray{ComplexF32}(undef, -n_by_2:n_by_2-1, -n_by_2:n_by_2) .= 0.0
	for j = -n_by_2:n_by_2-1
		for k = -n_by_2:n_by_2
			FD_star[j,k] = exp(2*π*1im*j*q*k/m)
		end
	end
	fd_star_fd_col_one = OffsetArray{ComplexF32}(undef, -n_by_2:n_by_2-1) .= 0.0
	l = -n_by_2
	for k = -n_by_2:n_by_2-1
		for j = -n_by_2:n_by_2
			fd_star_fd_col_one[k] .+= exp(2*π*1im*q*j*(k-l)/m)
		end
	end
	FD_star_FD = Toeplitz(parent(fd_star_fd_col_one),parent(fd_star_fd_col_one))
	# Use Julia's backslah operator to calculate the inverse.
	# As opposed to the paper's application of the Gohberg-Semencul formula.
	# It may be slightly slower, but it is well tested, standard Julia code, and
	# we don't have to write, debug, worry about it...
	inv_operator = FD_star_FD \ parent(FD_star)
	I₁ = zeros(np1,np1,n)
	I₂ = zeros(np1,n,n)
	I = zeros(n,n,n)
	for k=1:np1
		for l=1:np1
			v = Ĩ[k,l,:]
			v = AdjFDecimated(v,q=q)
			u = inv_operator * v
			I₁[k,l,:] .= u[:]
		end
	end
	for k=1:np1
		for l=1:n
			v = I₁[k,:,l]
			v = AdjFDecimated(v,q=q)
			u = inv_operator * v
			I₂[k,:,l] .= u[:]
		end
	end

	for k=1:n
		for l=1:n
			v = I₂[:,k,l]
			v = AdjFDecimated(v,q=q)
			u = inv_operator * v
			I[:,k,l] .= u[:]
		end
	end
	return I
end

function hvect(left,middle,right)
	reshape( vcat([left;], [middle;],  [right;]),(1,:))
end

# This is Algorithm 2 of Averbuch et al. (2016)
function recover2d(ÛΩpp, ÛD, j; q)
    np1,foo = size(ÛΩpp)
    @assert foo == np1
    n = np1 -1
    # Algorithm line 3
    α = (n/2 - j)/(n/2)
    # Algorithm line 4
    m = (q*n)+1
    # Algorithm line 5
    if j == 0
        # Algorithm line 6
        result = ÛΩpp
        return result
    # Algorithm line 7
    end
    # Algorithm Line 8
    # In matlab, the code is an instantiated array from a range of values.
    # Match those semantics with a Julia list comprehension...
    y₁ = [ y * (-2 * q * π / m) for y in (-(n/2):(n/2)) ]
    # Algorithm Line 9; again check Matlab code
    x₁ = [ x * (-2 * q * α * π / m) for x in (-(n/2):(n/2)) ]
    # Algorithm Line 10
    C = zeros(np1,np1)
    # Algorithm line 11
    for k = 1:j
        # Algorithm line 12
        C[k,:] .= TrigResample(ÛΩpp[k, 1:np1],y₁,x₁, algo="LS")
        # Algorithm line 13
        C[n +2 -k, :] .= TrigResample(ÛΩpp[n + 2 - k, 1:np1],y₁,x₁, algo="LS")
    # Algorithm line 14
    end
    # Algorithm line 15
    # FIXME: consider keeping this as ranges for efficiency...
    # Or, alternatively, recode using the [1:10;] syntax to instantiate the array, rather
    # than a list comprehension...
    x = [xi*(-2)*(q*π/m) for xi in -(n/2 - j):(n/2 - j)]
    # Algorithm line 16
    # WARNING! The matlab code for y produces a row-vector. I am jumping through
    # hoops here to make sure that the julia code produces the same thing.
    # IF that horizontality turns out to be an artifact of the matlab semantics,
    # this could be the site of either a thrown error, or subtle bugs.
    # DANGER!!!!
	#y = reshape( vcat([-n/2:-n/2+j-1;], [-n/2:n/2;].*α,  [n/2-j+1:n/2;]),(1,:)) .* (-2) * (q * π / m)
    y = hvect([-n/2:-n/2+j-1;], [-n/2:n/2;].*α, [n/2-j+1:n/2;]) .* (-2) * (q * π / m)
    # Algorithm line 17
    R₁ = zeros(np1 - 2*j, np1)
    # Algorithm line 18
    for k=1:np1
        # Algorithm lines 19,20
        # FIXME: (Maybe?)
        # My helper hvect might need to be reworked for these slices
        # as opposed to the instantiated arrays...
        R₁ .= TrigResample(hvect(C[1:j, k], ÛΩpp[:,k], C[n-j+2:np1,k]),y,x, algo="LS")
    # Algorithm line 21
    end
    # Algorithm line 22
    R₂ = zeros(np1 - 2*j, np1 - 2*j)
    # Algorithm line 23
    for k=1:(np1-(2*j))
        # Algorithm lines 24,25
        R₂ .= TrigResample(hvect(ÛD[k+j,1:j],R₁[k,:],ÛD[k+j,n-j+2:np1]),y,x, algo="LS")
    # Algorithm line 26
    end
    # Algorithm line 27
    result = ÛD
    # Algorithm line 28
    result[1+j:np1-j,1+j:np1-j] .= R₂[:,:]
    return result
end

# This is Algorithm 3 in Averbuch et al. (2016)

function AdjFDecimated(y; q)
	# y is a vector of odd length.
	np1, = size(y)
	if ~isodd(np1)
		throw(DomainError("Length of y is not odd."))
	end
	# Algorithm line 3
	n = np1 - 1
	# Algorithm line 4
	m = q*n + 1
	# Algorithm line 5
	z = zeros(m,1)
	# Algorithm line 6
	z[1:q:m] .= y[:]
	# Algorithm line 7
	# FIXME (Maybe?) Why didn't they use m here? Should we check for equality?
	# Checked/verified(?) with their matlab code. Modulo the matlab length() semantics
	# with an N dimensional array, it is the length of the vector.
	# In our case, they built z as a 1D column vector a few lines above, so
	# the interpretation matches.
	l, = size(z)
	# Algorithm line 8
	pf = floor(Int,l/2)
	# Algorithm line 9
	pc = ceil(Int, l/2)
	# Algorithm line 10
	x = z[pf+1:l, 1:pf]
	# Algorithm line 11
	# OK; after staring at the paper, and the matlab implementation, the actual sign convention
	# being used in their code is: Î = I exp(-2πi). So, assuming Julia's FFT sign convention matches matlab's,
	# we should be good.
	# Just checked: the Julia and Matlab sign conventions match. Julia's IFFT should give the same results as matlab's.
	xIFFT = IFFT(x)
	# Algorithm line 12
	# FIXME (Maybe) I think this Averbuch's way of dealing with
	# the complex valued return from the IFFT, and pulling out
	# the real part.
	x = m .* xIFFT[pc+1:l, 1:pc]
	# Algorithm line 13
	# Again, I think this might be pulling the real part out of
	# a complex valued vector...
	# Probably should do this the Julia way
	x = x[n+1:2*n]
	return x
end

# # This is Algorithm 4 of Averbuch et al., 2016
# function InvDecimatedFreq(Ĩ;q,algo="Averbuch")
# 	np1,foo,bar = size(Ĩ)
# 	@assert np1 == foo
# 	@assert np1 == bar
# 	n = np1 - 1
# 	# Algorithm line 3
# 	c = zeros(n,1)
# 	# Algorithm line 4
# 	m = q*n + 1
# 	# Algorithm line 5
# 	for k = -n/2:n/2-1
# 		# Algorithm line 6
# 		for l = -n/2:n/2
# 			# Algorithm line 7
# 			c[k+n/2+1] .+= exp(π * 1im * q * l *(-n/2 - k)/m)
# 		# Algorithm line 8
# 		end
# 	# Algorithm line 9
# 	end
# 	# Algorithm line 10
# 	# FIXME (Maybe?) See how/if we can use the Julia Toeplitz code...
# 	(M₁,M₂,M₃,M₄) = ToeplitzInv(c)
# 	# Algorithm line 11; implicitly loop unrolled
# 	D₁ = ToeplitzDiag(M₁)
# 	D₂ = ToeplitzDiag(M₂)
# 	D₃ = ToeplitzDiag(M₃)
# 	D₄ = ToeplitzDiag(M₄)
# 	# Algorithm line 12
# 	I₁ = zeros(np1,np1,n)
# 	# Algorithm line 13
# 	I₂ = zeros(np1,n,n)
# 	# Algorithm line 14
# 	I = zeros(n,n,n)
# 	# Algorithm line 15
# 	for k=1:np1
# 		# Algorithm line 16
# 		for l=1:np1
# 			# Algorithm line 17
# 			v = Ĩ[k,l,:]
# 			# Algorithm line 18
# 			v = AdjFDecimated(v,q=q)
# 			# Algorithm line 19
# 			u = ToeplitzInvMul(D₁,D₂,D₃,D₄,v)
# 			# Algorithm line 20
# 			I₁[k,l,:] .= u[:]
# 		# Algorithm line 21
# 		end
# 	# Algorithm line 22
# 	end
# 	# Algorithm line 23
# 	for k=1:np1
# 		# Algorithm line 24
# 		for l=1:n
# 			# Algorithm line 25
# 			v = I₁[k,:,l]
# 			# Algorithm line 26
# 			v = AdjFDecimated(v,q=q)
# 			# Algorithm line 27
# 			u = ToeplitzInvMul(D₁,D₂,D₃,D₄,v)
# 			# Algorithm line 28
# 			I₂[k,l,:] .= u[:]
# 		# Algorithm line 29
# 		end
# 	# Algorithm line 30
# 	end
# 	# Algorithm line 31
# 	for k=1:n
# 		# Algorithm line 32
# 		for l=1:n
# 			# Algorithm line 33
# 			v = I₂[:,k,l]
# 			# Algorithm line 34
# 			v = AdjFDecimated(v,q=q)
# 			# Algorithm line 35
# 			u = ToeplitzInvMul(D₁,D₂,D₃,D₄,v)
# 			# Algorithm line 36
# 			I[:,k,l] .= u[:]
# 		# Algorithm line 37
# 		end
# 	# Algorithm line 38
# 	end
# 	return I
# # End of Algorithm 4
# end

# This is a shim whose only job is to dispatch between the fast Toeplitz
# Algorithm of Averbuch et al. (2016) and their suggested alternative
# of simply formulating a least squares problem for the tigonometric interpolations.


function TrigResample(y, f, x; algo = "LS")
	N, = size(y)
	foo, = size(f)
	@assert N == foo
	M, = size(x)
	#FIXME consider doing this calculation in 32 bits for speed.
	A = zeros(::Complex64,(N,N))

	for j in 1:N
		for k in 1:N
			a[j,k] .= exp(1im * k * y[k])
		end
	end
	# Solve the bloody thing using least squares or whatever \ chooses
	α = A \ f
	# Now evaluate the resulting polynomial...
	result = zeros(::Complex64, M)
	for k in -N/2:(N/2)-1
		for i in 1:M
			result[i] .+= α[k] * exp(1im * k * x[i])
		end
	end
	return result
end

# # WARNING! Need to establish which "type" of NUFFT this routine
# # is calling by looking up references 15 and 16 in Averbuch et al. (2016)
#
# # Algorithm 5
# function TrigResample(y, f, x; algo = "Toeplitz")
# 	throw(ArgumentError("The fast Toeplitz algorithm is not fully implemented (yet)."))
#
#     # This assert is incorrect. If I ever get back to this algorithm, fix it!
# 	n, = size(y)
# 	foo, = size(x)
# 	@assert n == foo
#
# 	# Algorithm line 3
# 	# FIXME: HUH??? Why are we defining a full row vector
# 	# when it appears that we are only using the first element of it
# 	# in the rest of this routine???
# 	k = [-n/2:n/2-1;]
#
# 	# Algorithm line 4
# 	# FIXME (Maybe?) Code smell; what is going on with Length(y) instead
# 	# of simply "n" in their algorithm??? Check the matlab code...
# 	c = n * NUFFT₁(y,exp(1im * k[1] .* y ),-1,n)
# 	# Algorithm line 5
# 	(M₁,M₂,M₃,M₄) = ToeplitzInv(c)
# 	# Algorithm line 6:8 ; loop unrolled by hand here...
# 	# FIXME (Maybe?) The algorithm suggests computing these in the preprocessing...
# 	D₁ = ToeplitzDiag(M₁[:,1],M₁[1,:])
# 	D₂ = ToeplitzDiag(M₂[:,1],M₂[1,:])
# 	D₃ = ToeplitzDiag(M₃[:,1],M₃[1,:])
# 	D₄ = ToeplitzDiag(M₄[:,1],M₄[1,:])
# 	# Algorithm line 9
# 	v = n * NUFFT₁(y,f,-1,n)
# 	# Algorithm line 10
# 	α = ToeplitzInvMul(D₁,D₂,D₃,D₄,v)
# 	# Algorithm line 11
# 	g = NUFFT₂(x,1,n,α)
# 	# Return value
# 	return g
# # End of Algorithm 5
# end
#
# # Algorithm 6
# function ToeplitzInv(c)
#
# return
# # End of Algorithm 6
#
#
# # Algorithm 7
# function ToeplitzDiag(c,r)
# 	# Algorithm line 3
# 	n, = size(c)
# 	# Algorithm line 4
# 	C = [c;0;r[n:-1:2]]
# 	# Algorithm line 5
# 	D = FFT(C)
# 	return D
# # End of Algorithm 5
# end
