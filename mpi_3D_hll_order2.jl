using MPI
using HDF5
# using PyPlot
push!(LOAD_PATH, "/home/bruno/Desktop/Dropbox/Developer/hydro")
using tools

MPI.Init()
comm = MPI.COMM_WORLD
const pId = MPI.Comm_rank(comm)
const nProc = MPI.Comm_size(comm)
const root = 0


if pId == root
  print( "\nHydrodynamics 3D\n\nUsing MPI\nnProcess: $nProc\n" )
end

##################################################################
const nProc_x = round( Int, nProc^(1/3))
const nProc_y = round( Int, nProc^(1/3))
const nProc_z = round( Int, nProc^(1/3))
if nProc_x*nProc_x*nProc_x !=  nProc
  error("ERROR: Number of processes must be an integer cubed (8, 27, 64, 125, 216")
  exit()
end
const pId_z = divrem( pId, nProc_x*nProc_y)[1] #pId // (nProc_x*nProc_y)
const pId_x = divrem( pId, nProc_x )[2]  #pId % nProc_x
const pId_y = divrem( pId - pId_z*nProc_y*nProc_z, nProc_x )[1]  #(pId-pId_z*nProc_y*nProc_x) // nProc_x
const pParity = mod( pId_x + pId_y + pId_z, 2 )
#Boundery processes and periodic Boundery processes
const pDownId   = pId_y == 0         ? pId_z*nProc_x*nProc_y + (nProc_y-1)*nProc_x + pId_x : pId_z*nProc_x*nProc_y + (pId_y-1)*nProc_x + pId_x
const pUpId     = pId_y == nProc_y-1 ? pId_z*nProc_x*nProc_y + pId_x                       : pId_z*nProc_x*nProc_y + (pId_y+1)*nProc_x + pId_x
const pLeftId   = pId_x == 0         ? pId_z*nProc_x*nProc_y + pId_y*nProc_x + nProc_x-1   : pId_z*nProc_x*nProc_y + pId_y*nProc_x + (pId_x-1)
const pRightId  = pId_x == nProc_x-1 ? pId_z*nProc_x*nProc_y + pId_y*nProc_x               : pId_z*nProc_x*nProc_y + pId_y*nProc_x + (pId_x+1)
const pBottomId = pId_z == 0         ? (nProc_z-1)*nProc_x*nProc_y + pId_y*nProc_x + pId_x : (pId_z-1)*nProc_x*nProc_y + pId_y*nProc_x + pId_x
const pTopId    = pId_z == nProc_z-1 ? pId_y*nProc_x + pId_x                               : (pId_z+1)*nProc_x*nProc_y + pId_y*nProc_x + pId_x
##################################################################
#Global size of simulation

nPoints = 100

Lx = 1.0
Ly = 1.0
Lz = 1.0
xMax_g, xMin_g = Lx/2, -Lx/2
yMax_g, yMin_g = Ly/2, -Ly/2
zMax_g, zMin_g = Lz/2, -Lz/2


nDepth, nHeight, nWidth = nPoints, nPoints, nPoints
dx_g, dy_g, dz_g = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
xPoints = dx_g * collect( 0 : nWidth-1 )  + xMin_g
yPoints = dy_g * collect( 0 : nHeight-1 ) + yMin_g
zPoints = dz_g * collect( 0 : nDepth-1 )  + zMin_g

#Local size of simulation
nDepth_p, nHeight_p, nWidth_p = round( Int, nDepth/nProc_z ), round( Int, nHeight/nProc_y ), round( Int, nWidth/nProc_x )
xPoints_p = xPoints[pId_x*nWidth_p+1: (pId_x+1)*nWidth_p ]
yPoints_p = yPoints[pId_y*nHeight_p+1: (pId_y+1)*nHeight_p ]
zPoints_p = zPoints[pId_z*nDepth_p+1: (pId_z+1)*nDepth_p ]


const dx = dx_g
const dy = dy_g
const dz = dz_g
const xMin = xPoints_p[1]
const yMin = yPoints_p[1]
const zMin = zPoints_p[1]
const xMax = xPoints_p[end]
const yMax = yPoints_p[end]
const zMax = zPoints_p[end]


##################################################################
#Create output
outDir = "/home/bruno/Desktop/data/hydro/"
outFileName = outDir * "sphere_$(pId_z)_$(pId_y)_$(pId_x).h5"
outFile = h5open( outFileName, "w")

outFile[ "ponitsX" ] = map( Float32, xPoints_p )
outFile[ "ponitsY" ] = map( Float32, yPoints_p )
outFile[ "ponitsZ" ] = map( Float32, zPoints_p )

if pId == root
  print( "\n[ pID: $pId ] Blocks: $(nProc_x) x $(nProc_y) x $(nProc_z)\n" )
  print(   "[ pID: $pId ] Points: $(nProc_x*nWidth_p) x $(nProc_y*nHeight_p) x $(nProc_z*nDepth_p)\n" )
  print(   "[ pID: $pId ] PpB:    $(nWidth_p) x $(nHeight_p) x $(nDepth_p)\n\n" )
  println( "[ pID: $pId ] Out file: $outFileName\n" )
end
##################################################################
#Initialize data

const gamma = 7/5
const c0 = 0.4
vals_1 = zeros( Float64, (5, nWidth_p +4, nHeight_p + 4, nDepth_p + 4) ) #[ rho, rho*v_x, rho*v_y, rho*v_z, Energy = rho*vel**2/2 + p/(gamma-1)]
vals_half = zeros( Float64, (5, nWidth_p +4, nHeight_p + 4, nDepth_p + 4) ) #For half time step
vals_2 = zeros( Float64, (5, nWidth_p +4, nHeight_p + 4, nDepth_p + 4) ) #For next timestep
flx_x = zeros( Float64, (5, nWidth_p +4, nHeight_p + 4, nDepth_p + 4) ) #[rho*vel_x, rho*vel_x*vel_x + p, rho*vel_y*vel_x, rho*vel_z*vel_x, vel_x*(E+p)]
flx_y = zeros( Float64, (5, nWidth_p +4, nHeight_p + 4, nDepth_p + 4) ) #[rho*vel_y, rho*vel_x*vel_y, rho*vel_y*vel_y + p, rho*vel_z*vel_y, vel_y*(E+p)]
flx_z = zeros( Float64, (5, nWidth_p +4, nHeight_p + 4, nDepth_p + 4) ) #[rho*vel_z, rho*vel_x*vel_z, rho*vel_y*vel_z, rho*vel_z*vel_z + p, vel_z*(E+p)]

for k in 3:nDepth_p+2
  z = zMin + (k-3)*dz
  for j in 3:nHeight_p+2
    y = yMin + (j-3)*dy
    for i in 3:nWidth_p+2
      x = xMin + (i-3)*dx
#       if (i < nx/4) && (j > ny/4 && j < ny*3/4) && (k > nz/4 && k < nz*3/4)
#       if sqrt( (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2 ) < 0.2
      if sqrt( (x)^2 + (y)^2 + (z)^2 ) < 0.2
	rho = 1
        v_x = 0
        v_y = 0
        v_z = 0
        p = 5
      else
        rho = 0.7
        v_x = 0
        v_y = 0
        v_z = 0
        p = 1
      end
      v2 =  v_x*v_x + v_y*v_y * v_z*v_z
      vals_1[ :, i, j, k ] = [ rho, rho*v_x, rho*v_y, rho*v_z, rho*v2/2 + p/(gamma-1) ]
    end
  end
end

function saveVals32( outFile, vals, nSnap )
  outFile["vals/$nSnap"] = map( Float32, vals[:, 3:end-2, 3:end-2, 3:end-2])
end

##################################################################
bounderyLeft_new   = zeros( Float64, ( 5, 2, nHeight_p+4, nDepth_p+4 ) )
bounderyRight_new  = zeros( Float64, ( 5, 2, nHeight_p+4, nDepth_p+4 ) )
bounderyUp_new     = zeros( Float64, ( 5, nWidth_p+4, 2, nDepth_p+4 ) )
bounderyDown_new   = zeros( Float64, ( 5, nWidth_p+4, 2, nDepth_p+4 ) )
bounderyTop_new    = zeros( Float64, ( 5, nWidth_p+4, nHeight_p+4, 2 ) )
bounderyBottom_new = zeros( Float64, ( 5, nWidth_p+4, nHeight_p+4, 2 ) )
##################################################################
function transferBounderies( valsIn::Array{Float64} )
  bounderyRight  = valsIn[ :, [end-3, end-2],          :,          : ]
  bounderyLeft   = valsIn[ :,       [ 3, 4 ],          :,          : ]
  bounderyUp     = valsIn[ :,         :, [end-3, end-2],           : ]
  bounderyDown   = valsIn[ :,         :,        [ 3, 4],           : ]
  bounderyTop    = valsIn[ :,         :,           :, [end-3, end-2] ]
  bounderyBottom = valsIn[ :,         :,           :,        [ 3, 4] ]

#   MPI.Barrier(comm)
  if pParity == 0

    MPI.Send( bounderyLeft, pLeftId, 3, comm )
    MPI.Recv!( bounderyLeft_new, pRightId, 3, comm )
    valsIn[ :, [ end-1, end ], :, : ] = bounderyLeft_new

    MPI.Send( bounderyRight, pRightId, 4, comm )
    MPI.Recv!( bounderyRight_new, pLeftId, 4, comm )
    valsIn[ :, [ 1, 2] , :, : ] = bounderyRight_new

    MPI.Send( bounderyUp, pUpId, 1, comm )
    MPI.Recv!( bounderyUp_new, pDownId, 1, comm )
    valsIn[ :, :, [ 1, 2], :] = bounderyUp_new

    MPI.Send( bounderyDown, pDownId, 2, comm )
    MPI.Recv!( bounderyDown_new, pUpId, 2, comm )
    valsIn[ :, :, [ end-1, end ], :] = bounderyDown_new

    MPI.Send( bounderyTop, pTopId, 5, comm )
    MPI.Recv!( bounderyTop_new, pBottomId, 5, comm )
    valsIn[ :, :, :, [ 1, 2] ] = bounderyTop_new

    MPI.Send( bounderyBottom, pBottomId, 6, comm )
    MPI.Recv!( bounderyBottom_new, pTopId, 6, comm )
    valsIn[ :, :, :, [ end-1, end ] ] = bounderyBottom_new
  else

    MPI.Recv!( bounderyLeft_new, pRightId, 3, comm )
    MPI.Send( bounderyLeft, pLeftId, 3, comm )
    valsIn[ :, [ end-1, end ], :, : ] = bounderyLeft_new

    MPI.Recv!( bounderyRight_new, pLeftId, 4, comm )
    MPI.Send( bounderyRight, pRightId, 4, comm )
    valsIn[ :, [ 1, 2], :, : ] = bounderyRight_new

    MPI.Recv!( bounderyUp_new, pDownId, 1, comm )
    MPI.Send( bounderyUp, pUpId, 1, comm )
    valsIn[ :, :, [ 1, 2], :] = bounderyUp_new

    MPI.Recv!( bounderyDown_new, pUpId, 2, comm )
    MPI.Send( bounderyDown, pDownId, 2, comm )
    valsIn[ :, :, [ end-1, end ], :] = bounderyDown_new

    MPI.Recv!( bounderyTop_new, pBottomId, 5, comm )
    MPI.Send( bounderyTop, pTopId, 5, comm )
    valsIn[ :, :, :, [ 1, 2] ] = bounderyTop_new

    MPI.Recv!( bounderyBottom_new, pTopId, 6, comm )
    MPI.Send( bounderyBottom, pBottomId, 6, comm )
    valsIn[ :, :, :, [ end-1, end ] ] = bounderyBottom_new
  end
  MPI.Barrier(comm)
end


function setGlobalBounderies( vals )
  #Open boundaries
  #X boundery
  if pId_x == 0
    vals[ :,     1, :, : ] = vals[ :,     3, :, : ]
    vals[ :,     2, :, : ] = vals[ :,     3, :, : ]
  elseif pId_x == nProc_x-1
    vals[ :,   end, :, : ] = vals[ :, end-2, :, : ]
    vals[ :, end-1, :, : ] = vals[ :, end-2, :, : ]
  end
#   if bounce vals[ 2,   1, :, : ] = -vals[ 2,     2, :, : ] end #bounce left wall
#   vals[ 2, end, :, : ] = -vals[ 2, end-1, :, : ] #bounce right wall
  #Y boundery
  if pId_y == 0
    vals[ :, :,     1, : ] = vals[ :, :,     3, : ]
    vals[ :, :,     2, : ] = vals[ :, :,     3, : ]
  elseif pId_y == nProc_y-1
    vals[ :, :,   end, : ] = vals[ :, :, end-2, : ]
    vals[ :, :, end-1, : ] = vals[ :, :, end-2, : ]
  end
#   if bounce vals[ 3, :,   1, : ] = -vals[ 3, :,     2, : ] end #bounce down wall
#   vals[ 3, :, end, : ] = -vals[ 3, :, end-1, : ] #bounce up wall
  #Z boundery
  if pId_z == 0
    vals[ :, :, :,     1 ] = vals[:, :, :,     3 ]
    vals[ :, :, :,     2 ] = vals[:, :, :,     3 ]
  elseif pId_y == nProc_y-1
    vals[ :, :, :,   end ] = vals[:, :, :, end-2 ]
    vals[ :, :, :, end-1 ] = vals[:, :, :, end-2 ]
  end
#   if bounce vals[ 4, :, :,   1 ] = -vals[ 4, :, :,     2 ] end #bounce bottom wall
#   println( "Fin: setBounderies" )
end

getPresure( rho, vel, E ) = ( E  - rho*vel*vel/2 ) * (gamma-1)

function getPrim( vals, i, j, k ) #Get primmitives from conservatives
  rho = vals[ 1, i, j, k ]
  vx = vals[ 2, i, j, k ] / rho
  vy = vals[ 3, i, j, k ] / rho
  vz = vals[ 4, i, j, k ] / rho
  E  = vals[ 5, i, j, k ]
  v2 = vx*vx + vy*vy + vz*vz
  p = ( E - rho*v2/2 ) * (gamma-1)
  return [ rho, vx, vy, vz, p, E ]
end

function getConserv( prim ) #get density, momentum and energy
  rho, vx, vy, vz, p, E = prim
  return [ rho, rho*vx, rho*vy, rho*vz, E  ]
end

getSoundVel( rho, p ) = sqrt( gamma * p / rho )

function getWavesSpeed( speed_l, speed_r, sound_l, sound_r )
  wSpeed_l = min( speed_l - sound_l, speed_r - sound_r )
  wSpeed_r = max( speed_l + sound_l, speed_r + sound_r )
  return wSpeed_l, wSpeed_r
end



function getFlux( axis, prim )
  rho, vx, vy, vz, p, E = prim
  if   axis == 'x'
    flux = [ rho*vx, rho*vx*vx + p, rho*vy*vx, rho*vz*vx, vx*( E + p ) ]
  elseif axis == 'y'
    flux = [ rho*vy, rho*vx*vy, rho*vy*vy + p, rho*vz*vy, vy*( E + p ) ]
  elseif axis == 'z'
    flux = [ rho*vz, rho*vx*vz, rho*vy*vz, rho*vz*vz + p, vz*( E + p ) ]
  end
  return flux
end

function getTimeMin( vals, flx_x, flx_y, flx_z )
  dt = 1e40
  for k in 1:nDepth_p+4 #ALSO BOUNDERIES
    for j in 1:nHeight_p+4
      for i in 1:nWidth_p+4
        rho, vx, vy, vz, p, E = getPrim( vals, i, j, k )
        soundVel = getSoundVel( rho, p )
#         setFluxes( 'x', flx_x, i, j, k, rho, vx, vy, vz, p, E  )
#         setFluxes( 'y', flx_y, i, j, k, rho, vx, vy, vz, p, E  )
#         setFluxes( 'z', flx_z, i, j, k, rho, vx, vy, vz, p, E  )
        dt = min( dt, dx / (  abs( vx ) + soundVel ) )
        dt = min( dt, dy / (  abs( vy ) + soundVel ) )
        dt = min( dt, dz / (  abs( vz ) + soundVel ) )
      end
    end
  end
#   println( "Fin: getTimeMin, $dt" )
  return c0*dt
end

function getTimeMin_global( vals, flx_x, flx_y, flx_z )
  dt = getTimeMin( vals, flx_x, flx_y, flx_z )
  dt_global = -1.0
  dt_global =  MPI.Reduce( dt, MPI.MIN, root, comm)
  dt_global = pId == root ? [dt_global] : [-1.]
  MPI.Bcast!(dt_global, root, comm)
  return dt_global[1]
end

function doHalfTimeStep( dt, vals, vals_half, flx_x, flx_y, flx_z, )
  for k in 2:nDepth_p+3
    for j in 2:nHeight_p+3
      for i in 2:nWidth_p+3
	iFlux_l_x = flx_x[ :, i-1, j, k  ]
	iFlux_r_x = flx_x[ :,   i, j, k  ]
	iFlux_l_y = flx_y[ :, i, j-1, k  ]
	iFlux_r_y = flx_y[ :,   i, j, k  ]
	iFlux_l_z = flx_z[ :, i, j, k-1  ]
	iFlux_r_z = flx_z[ :,   i, j, k  ]
	term_x = ( iFlux_r_x - iFlux_l_x ) / dx
        term_y = ( iFlux_r_y - iFlux_l_y ) / dy
        term_z = ( iFlux_r_z - iFlux_l_z ) / dz
        vals_half[ :, i, j, k ] = vals[ :, i, j, k ] - dt/2*( term_x + term_y + term_z )
      end
    end
  end
#   println( "Fin: doHalfTimeStep" )
end

function riemann_hll( axis, prim_l, prim_r, prim_l_c, prim_r_c )
  rho_l, vx_l, vy_l, vz_l, p_l, E_l = prim_l
  rho_r, vx_r, vy_r, vz_r, p_r, E_r = prim_r
#   rho_l_c, vx_l_c, vy_l_c, vz_l_c, p_l_c, E_l_c = prim_l_c
#   rho_r_c, vx_r_c, vy_r_c, vz_r_c, p_r_c, E_r_c = prim_r_c
  cs_l = getSoundVel( rho_l, p_l )
  cs_r = getSoundVel( rho_r, p_r )
  if     axis == 'x' s_l, s_r = getWavesSpeed( vx_l, vx_r, cs_l, cs_r )
  elseif axis == 'y' s_l, s_r = getWavesSpeed( vy_l, vy_r, cs_l, cs_r )
  elseif axis == 'z' s_l, s_r = getWavesSpeed( vz_l, vz_r, cs_l, cs_r )
  end
  F_l = getFlux( axis, prim_l )
  F_r = getFlux( axis, prim_r )
  if     s_l > 0
    return F_l
  elseif s_r < 0
    return F_r
  else
    U_l = getConserv( prim_l )
    U_r = getConserv( prim_r )
    return ( s_r*F_l - s_l*F_r + s_l*s_r*( U_r - U_l ) ) / ( s_r - s_l )
  end
end

function hll_o2_step( dt, vals, vals_half, vals_new )
  for k in 3:nDepth_p+2
    for j in 3:nHeight_p+2
      for i in 3:nWidth_p+2
  # 	println( " [ $i , $j , $k ]" )
	prim_c = getPrim( vals_half, i, j, k )
	prim_l_x = getPrim( vals_half, i-1, j, k )
	prim_r_x = getPrim( vals_half, i+1, j, k )
	prim_l_y = getPrim( vals_half, i, j-1, k )
	prim_r_y = getPrim( vals_half, i, j+1, k )
	prim_l_z = getPrim( vals_half, i, j, k-1 )
	prim_r_z = getPrim( vals_half, i, j, k+1 )
	prim_ll_x = getPrim( vals_half, i-2, j, k )
	prim_rr_x = getPrim( vals_half, i+2, j, k )
	prim_ll_y = getPrim( vals_half, i, j-2, k )
	prim_rr_y = getPrim( vals_half, i, j+2, k )
	prim_ll_z = getPrim( vals_half, i, j, k-2 )
	prim_rr_z = getPrim( vals_half, i, j, k+2 )
  ################################################################
  #QUESTION: in notes double division by 2 for deltas
  ################################################################
	delta_c_x = ( prim_r_x - prim_l_x ) / 2
	delta_c_y = ( prim_r_y - prim_l_y ) / 2
	delta_c_z = ( prim_r_z - prim_l_z ) / 2

  # 	delta_l_x = ( prim_c - prim_ll_x ) / 2
  # 	delta_l_y = ( prim_c - prim_ll_y ) / 2
  # 	delta_l_z = ( prim_c - prim_ll_z ) / 2
  #
  # 	delta_r_x = ( prim_rr_x - prim_c ) / 2
  # 	delta_r_y = ( prim_rr_y - prim_c ) / 2
  # 	delta_r_z = ( prim_rr_z - prim_c ) / 2

	prim_cl_x = prim_c - delta_c_x /2
	prim_cl_y = prim_c - delta_c_y /2
	prim_cl_z = prim_c - delta_c_z /2

	prim_cr_x = prim_c + delta_c_x /2
	prim_cr_y = prim_c + delta_c_y /2
	prim_cr_z = prim_c + delta_c_z /2

	prim_lr_x = prim_l_x + delta_c_x /2
	prim_lr_y = prim_l_y + delta_c_y /2
	prim_lr_z = prim_l_z + delta_c_z /2

	prim_rl_x = prim_r_x - delta_c_x /2
	prim_rl_y = prim_r_y - delta_c_y /2
	prim_rl_z = prim_r_z - delta_c_z /2

	iFlux_l_x = riemann_hll( 'x', prim_lr_x, prim_cl_x, prim_l_x, prim_c  )
	iFlux_r_x = riemann_hll( 'x', prim_cr_x, prim_rl_x, prim_c, prim_r_x  )
	iFlux_l_y = riemann_hll( 'y', prim_lr_y, prim_cl_y, prim_l_y, prim_c  )
	iFlux_r_y = riemann_hll( 'y', prim_cr_y, prim_rl_y, prim_c, prim_r_y  )
	iFlux_l_z = riemann_hll( 'z', prim_lr_z, prim_cl_z, prim_l_z, prim_c  )
	iFlux_r_z = riemann_hll( 'z', prim_cr_z, prim_rl_z, prim_c, prim_r_z  )
	term_x = ( iFlux_r_x - iFlux_l_x ) / dx
	term_y = ( iFlux_r_y - iFlux_l_y ) / dy
	term_z = ( iFlux_r_z - iFlux_l_z ) / dz
	vals_new[ :, i, j, k ] = vals[ :, i, j, k ] - dt/2*( term_x + term_y + term_z )
      end
    end
  end
end






function doTimeStep( vals, vals_half, vals_new, flx_x, flx_y, flx_z,  )
  computeTime, transferTime = 0, 0
  transferTime += @elapsed transferBounderies( vals )
  computeTime += @elapsed setGlobalBounderies( vals )
  dt = getTimeMin( vals, flx_x, flx_y, flx_z )
#   print( "[ pID: $pId ] dt: $dt\n" )
  computeTime += @elapsed doHalfTimeStep( 0, vals, vals_half, flx_x, flx_y, flx_z, )
  transferTime += @elapsed transferBounderies( vals_half )
  computeTime += @elapsed setGlobalBounderies( vals_half )
  computeTime += @elapsed hll_o2_step( dt, vals, vals_half, vals_new )
#   println( "Fin: doTimeStep" )
  return [ computeTime, transferTime ]
end
###############################################################

const nIter = 100
const nSnaps = 50
const iterPerSave = divrem( nIter, nSnaps )[1]
saveCount = 1

# println( "\nnPoints: $nx x $ny x $nz " )
if pId == root
  println( "\nnSnapshots: $nSnaps" )
  println( "\nStarting $nIter iterations\n" )
end

saveVals32( outFile, vals_1, 0)
time_compute = 0 
time_transfer =  0
time_save = 0 
printProgress( 0, nIter, 0 )
for i in 1:nIter
#   println( i )
  if i == saveCount*iterPerSave
    if pId == root printProgress( i, nIter, time_compute+time_transfer+time_save ) end
    time_save += @elapsed saveVals32( outFile, vals_1, saveCount)
    saveCount += 1
  end
  time_step = doTimeStep( vals_1, vals_half, vals_2, flx_x, flx_y, flx_z )
  time_compute += time_step[1]
  time_transfer += time_step[2]
  vals_1, vals_2 = vals_2, vals_1
  MPI.Barrier(comm)
end

saveVals32( outFile, vals_1, saveCount)

if pId == root
  time_total = time_compute + time_save + time_transfer
  println( "\n\nTotal time: $(time_total)  " )
  println( " Compute time: $time_compute   $(time_compute/ time_total) )  " )
  println( " Transfer time: $time_transfer   $(time_transfer/ time_total) )  " )
  println( " Save time: $time_save    $(time_save/ time_total)  " )

  println("")
end

close( outFile )
MPI.Finalize()
