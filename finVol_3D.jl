#nucleares.unam.mx/astroplasmas
#jrechy@astro.unam.mx
# using PyPlot
using HDF5
push!(LOAD_PATH, "/home/bruno/Desktop/Dropbox/Developer/hydro")
using tools

println( "̣\nHydrodynamics 3D" )

outDir = "/home/bruno/Desktop/data/hydro/"
outFileName = outDir * "finVol3D_01.h5"
println( "\nOut file: $outFileName" )
outFile = h5open( outFileName, "w")

nPoints = 50

const nx = nPoints
const ny = nPoints
const nz = nPoints
const xMin = 0
const xMax = 1
const yMin = 0
const yMax = 1
const zMin = 0
const zMax = 1
const dx = (xMax - xMin ) / (nx - 1 )
const dy = (yMax - yMin ) / (ny - 1 )
const dz = (zMax - yMin ) / (nz - 1 )
box_x = linspace( xMin, xMax, nx )
box_y = linspace( yMin, yMax, ny )
box_z = linspace( zMin, zMax, nz )

outFile["box/x"] = Array(box_x)
outFile["box/y"] = Array(box_y)
outFile["box/z"] = Array(box_z)


###############################################################
const gamma = 7/5
const c0 = 0.4
vals_1 = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[ rho, rho*v_x, rho*v_y, rho*v_z, Energia = rho*vel**2/2 + p/(gamma-1)]
vals_2 = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #For next timestep
flx_x = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[rho*vel_x, rho*vel_x*vel_x + p, rho*vel_y*vel_x, rho*vel_z*vel_x, vel_x*(E+p)]
flx_y = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[rho*vel_y, rho*vel_x*vel_y, rho*vel_y*vel_y + p, rho*vel_z*vel_y, vel_y*(E+p)]
flx_z = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[rho*vel_z, rho*vel_x*vel_z, rho*vel_y*vel_z, rho*vel_z*vel_z + p, vel_z*(E+p)]
solid = zeros( Int8, ( nx + 2, ny + 2, nz + 2) ) 
function saveVals( outFile, vals, nSnap )
  outFile["vals/$nSnap"] = map( Float32, vals[:, 2:end-1, 2:end-1, 2:end-1])
end

###############################################################
#Initial conditions
for k in 2:nz+1
  z = zMin + (k-2)*dz
  for j in 2:ny+1
    y = yMin + (j-2)*dy
    for i in 2:nx+1
      x = xMin + (i-2)*dx
      if ( sqrt( (x)^2 + (y)^2 + (z)^2   ) <= 0.2 )
        rho = 1
        v_x = 0
        v_y = 0
        v_z = 0
        p = 10
      else
        rho = 0.7
        v_x = 0
        v_y = 0
        v_z = 0
        p = 1
      end
      v2 =  v_x*v_x + v_y*v_y * v_z*v_z
      vals_1[ :, i, j, k ] = [ rho, rho*v_x, rho*v_y, rho*v_z, rho*v2/2 + p/(gamma-1) ]
      #Set solid X boundery walls
      if i == 2 
	solid[ 1, j, k ] = 1
      end
      #Set solid Y boundery walls
      if j == 2
	solid[ i, 1, k ] = 3
      end
      #Set solid Z boundery walls
      if k == 2
	solid[ i, j, 1 ] = 5
      end
    end
  end
end

function setBounderies( vals )
  #Open boundaries
  #X boundery
  vals[:, end, :, :] = vals[:, end-1, :, :]
  #Y boundery
  vals[:, :, end, :] = vals[:, :, end-1, :]
  #Z boundery
  vals[:, :, :, end] = vals[:, :, :, end-1]
  #Bounce in solid walls
  for k in 1:nz+2
    for j in 1:ny+2
      for i in 1:nx+2
	s = solid[ i, j, k ]
	if     (s == 1) 
	  vals[ :, i, j, k ] =  vals[ :, i+1, j, k ] 
	  vals[ 2, i, j, k ] = -vals[ 2, i+1, j, k ] #bounce right     
	elseif (s == 2) 
	  vals[ :, i, j, k ] =  vals[ :, i-1, j, k ] 
	  vals[ 2, i, j, k ] = -vals[ 2, i-1, j, k ] #bounce left
	elseif (s == 3) 
	  vals[ :, i, j, k ] =  vals[ :, i, j+1, k ]
	  vals[ 3, i, j, k ] = -vals[ 3, i, j+1, k ] #bounce up     
	elseif (s == 4) 
	  vals[ :, i, j, k ] =  vals[ :, i, j-1, k ]
	  vals[ 3, i, j, k ] = -vals[ 3, i, j-1, k ] #bounce down
	elseif (s == 5) 
	  vals[ :, i, j, k ] =  vals[ :, i, j, k+1 ]
	  vals[ 4, i, j, k ] = -vals[ 4, i, j, k+1 ] #bounce top    
	elseif (s == 6) 
	  vals[ :, i, j, k ] =  vals[ :, i, j, k-1 ]
	  vals[ 4, i, j, k ] = -vals[ 4, i, j, k-1 ] #bounce bottom
	end
      end
    end
  end
end

getPresure( rho, vel, E ) = ( E  - rho*vel*vel/2 ) * (gamma-1)

function getPrim( vals, i, j, k )
  rho = vals[ 1, i, j, k ]
  vx = vals[ 2, i, j, k ] / rho
  vy = vals[ 3, i, j, k ] / rho
  vz = vals[ 4, i, j, k ] / rho
  E  = vals[ 5, i, j, k ]
  v2 = vx*vx + vy*vy + vz*vz
  p = ( E - rho*v2/2 ) * (gamma-1)
  return [ rho, vx, vy, vz, p, E ]
end

getSoundVel( rho, p ) = sqrt( gamma * p / rho )

function getWavesSpeed( speed_l, speed_r, sound_l, sound_r )
  wSpeed_l = min( speed_l - sound_l, speed_r - sound_r )
  wSpeed_r = max( speed_l + sound_l, speed_r + sound_r )
  return wSpeed_l, wSpeed_r
end

function setFluxes( coord, fluxes, i, j, k, rho, vx, vy, vz, p, E  ) #coord: 1->x, 2->y, 3->z
#   E = vals[ 5, i, j, k ]
  if coord == 1
    fluxes_local = [ rho*vx, rho*vx*vx + p, rho*vy*vx, rho*vz*vx, vx*( E + p ) ]
  elseif coord == 2
    fluxes_local = [ rho*vy, rho*vx*vy, rho*vy*vy + p, rho*vz*vy, vy*( E + p ) ]
  elseif coord == 3
    fluxes_local = [ rho*vz, rho*vx*vz, rho*vy*vz, rho*vz*vz + p, vz*( E + p ) ]
  end
  fluxes[ :, i, j, k ] = fluxes_local
end

function getTimeMin( vals, flx_x, flx_y, flx_z )
  dt = 1e40
  for k in 1:nz+2 #ALSO BOUNDERIES
    for j in 1:ny+2
      for i in 1:nx+2
        rho, vx, vy, vz, p, E = getPrim( vals, i, j, k )
        soundVel = getSoundVel( rho, p )
        setFluxes( 1, flx_x, i, j, k, rho, vx, vy, vz, p, E  )
        setFluxes( 2, flx_y, i, j, k, rho, vx, vy, vz, p, E  )
        setFluxes( 3, flx_z, i, j, k, rho, vx, vy, vz, p, E  )
        dt = min( dt, dx / (  abs( vx ) + soundVel ) )
        dt = min( dt, dy / (  abs( vy ) + soundVel ) )
        dt = min( dt, dz / (  abs( vz ) + soundVel ) )
      end
    end
  end
  return c0*dt
end

function getInterCellFlux_hll_x( values, fluxes, id_l, id_r, id_1, id_2 )
  rho_l, vx_l, vy_l, vz_l, p_l, E_l = getPrim( values, id_l, id_1, id_2 )
  rho_r, vx_r, vy_r, vz_r, p_r, E_r = getPrim( values, id_r, id_1, id_2 )
  cs_l = getSoundVel( rho_l, p_l )
  cs_r = getSoundVel( rho_r, p_r ) 
#   vel_l = sqrt( vx_l*vx_l + vy_l*vy_l + vz_l*vz_l )
#   vel_r = sqrt( vx_r*vx_r + vy_r*vy_r + vz_r*vz_r )
#   s_l, s_r = getWavesSpeed( vel_l, vel_r, cs_l, cs_r )
  s_l, s_r = getWavesSpeed( vx_l, vx_r, cs_l, cs_r )
  if s_l > 0
    return fluxes[ :, id_l, id_1, id_2 ]
  elseif s_r < 0
    return fluxes[ :, id_r, id_1, id_2 ]
  else
    F_l = fluxes[ :, id_l, id_1, id_2 ]
    F_r = fluxes[ :, id_r, id_1, id_2 ]
    U_l = values[ :, id_l, id_1, id_2 ]
    U_r = values[ :, id_r, id_1, id_2 ]
    return ( s_r*F_l - s_l*F_r + s_l*s_r*( U_r - U_l ) ) / ( s_r - s_l )
  end
end

function getInterCellFlux_hll_y( values, fluxes, id_l, id_r, id_1, id_2 )
  rho_l, vx_l, vy_l, vz_l, p_l, E_l = getPrim( values, id_1, id_l, id_2 )
  rho_r, vx_r, vy_r, vz_r, p_r, E_r = getPrim( values, id_1, id_r, id_2 )
  cs_l = getSoundVel( rho_l, p_l )
  cs_r = getSoundVel( rho_r, p_r ) 
#   vel_l = sqrt( vx_l*vx_l + vy_l*vy_l + vz_l*vz_l )
#   vel_r = sqrt( vx_r*vx_r + vy_r*vy_r + vz_r*vz_r )
#   s_l, s_r = getWavesSpeed( vel_l, vel_r, cs_l, cs_r )
  s_l, s_r = getWavesSpeed( vy_l, vy_r, cs_l, cs_r )
  if s_l > 0
    return fluxes[ :, id_1, id_l, id_2 ]
  elseif s_r < 0
    return fluxes[ :, id_1, id_r, id_2 ]
  else
    F_l = fluxes[ :, id_1, id_l, id_2 ]
    F_r = fluxes[ :, id_1, id_r, id_2 ]
    U_l = values[ :, id_1, id_l, id_2 ]
    U_r = values[ :, id_1, id_r, id_2 ]
    return ( s_r*F_l - s_l*F_r + s_l*s_r*( U_r - U_l ) ) / ( s_r - s_l )
  end
end

function getInterCellFlux_hll_z( values, fluxes, id_l, id_r, id_1, id_2 )
  rho_l, vx_l, vy_l, vz_l, p_l, E_l = getPrim( values, id_1, id_2, id_l )
  rho_r, vx_r, vy_r, vz_r, p_r, E_r = getPrim( values, id_1, id_2, id_r )
  cs_l = getSoundVel( rho_l, p_l )
  cs_r = getSoundVel( rho_r, p_r ) 
#   vel_l = sqrt( vx_l*vx_l + vy_l*vy_l + vz_l*vz_l )
#   vel_r = sqrt( vx_r*vx_r + vy_r*vy_r + vz_r*vz_r )
#   s_l, s_r = getWavesSpeed( vel_l, vel_r, cs_l, cs_r )
  s_l, s_r = getWavesSpeed( vz_l, vz_r, cs_l, cs_r )
  if s_l > 0
    return fluxes[ :, id_1, id_2, id_l ]
  elseif s_r < 0
    return fluxes[ :, id_1, id_2, id_r ]
  else
    F_l = fluxes[ :, id_1, id_2, id_l ]
    F_r = fluxes[ :, id_1, id_2, id_r ]
    U_l = values[ :, id_1, id_2, id_l ]
    U_r = values[ :, id_1, id_2, id_r ]
    return ( s_r*F_l - s_l*F_r + s_l*s_r*( U_r - U_l ) ) / ( s_r - s_l )
  end
end



function doTimeStep( vals, vals_new, flx_x, flx_y, flx_z )
  setBounderies( vals )
#   println( minimum(vals[1,:,:,:]) )
  dt = getTimeMin( vals, flx_x, flx_y, flx_z )
  for k in 2:nz+1
    for j in 2:ny+1
      for i in 2:nx+1
	iFlux_l_x = getInterCellFlux_hll_x( vals, flx_x, i-1, i, j, k  )
	iFlux_r_x = getInterCellFlux_hll_x( vals, flx_x, i, i+1, j, k  )
	iFlux_l_y = getInterCellFlux_hll_y( vals, flx_y, j-1, j, i, k  )
	iFlux_r_y = getInterCellFlux_hll_y( vals, flx_y, j, j+1, i, k  )
	iFlux_l_z = getInterCellFlux_hll_z( vals, flx_z, k-1, k, i, j  )
	iFlux_r_z = getInterCellFlux_hll_z( vals, flx_z, k, k+1, i, j  )
	term_x = ( iFlux_r_x - iFlux_l_x ) / dx
        term_y = ( iFlux_r_y - iFlux_l_y ) / dy
        term_z = ( iFlux_r_z - iFlux_l_z ) / dz
        vals_new[ :, i, j, k ] = vals[ :, i, j, k ] - dt*( term_x + term_y + term_z )
      end
    end
  end
end



# ###############################################################



const nIter = 100
const nSnaps = 50
const iterPerSave = divrem( nIter, nSnaps )[1]
saveCount = 1

println( "\nnPoints: $nx x $ny x $nz " )
println( "nSnapshots: $nSnaps" )
println( "\nStarting $nIter iterations\n" ) 
  
saveVals( outFile, vals_1, 0)
time_compute, time_save = 0, 0
printProgress( 0, nIter, time_compute )
for i in 1:nIter
  
  if i == saveCount*iterPerSave
    printProgress( i, nIter, time_compute+time_save )
    time_save += @elapsed saveVals( outFile, vals_1, saveCount)
    saveCount += 1
  end
  time_compute += @elapsed doTimeStep( vals_1, vals_2, flx_x, flx_y, flx_z )
  vals_1, vals_2 = vals_2, vals_1
end
saveVals( outFile, vals_1, saveCount)

time_total = time_compute + time_save
println( "\n\nTotal time: $(time_total)  " )
println( " Compute time: $time_compute   $(time_compute/ time_total) )  " )
println( " Save time: $time_save    $(time_save/ time_total)  " )

println("")
close( outFile )

