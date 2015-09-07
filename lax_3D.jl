#nucleares.unam.mx/astroplasmas
# using PyPlot
using HDF5
push!(LOAD_PATH, "/home/bruno/Desktop/Dropbox/Developer/hydro")
using tools

println( "̣\nHydrodynamics 3D" )

outDir = "/home/bruno/Desktop/data/hydro/"
outFileName = outDir * "hydro3D_01.h5"
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
outFile["box/x"] = box_x
outFile["box/y"] = box_y
outFile["box/z"] = box_z

###############################################################
const gamma = 7/5
const c0 = 0.6
vals_1 = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[ rho, rho*v_x, rho*v_y, rho*v_z, Energia = rho*vel**2/2 + p/(gamma-1)]
vals_2 = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #For next timestep
flx_x = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[rho*vel_x, rho*vel_x*vel_x + p, rho*vel_y*vel_x, rho*vel_z*vel_x, vel_x*(E+p)]
flx_y = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[rho*vel_y, rho*vel_x*vel_y, rho*vel_y*vel_y + p, rho*vel_z*vel_y, vel_y*(E+p)]
flx_z = zeros( Float64, (5, nx + 2, ny + 2, nz + 2) ) #[rho*vel_z, rho*vel_x*vel_z, rho*vel_y*vel_z, rho*vel_z*vel_z + p, vel_z*(E+p)]


function saveVals( outFile, vals, nSnap )
  outFile["vals/$nSnap"] = float32(vals[:, 2:end-1, 2:end-1, 2:end-1])
end

###############################################################
#Initial conditions
for k in 2:nz+1
  z = zMin + (k-2)*dz
  for j in 2:ny+1
    y = yMin + (j-2)*dy
    for i in 2:nx+1
      x = xMin + (i-2)*dx
      if ( sqrt( (x)^2 + (y)^2 + (z)^2 ) <= 0.5 )
        rho = 0.9
        v_x = 0
        v_y = 0
        v_z = 0
        p = 10
      else
        rho = 0.5
        v_x = 0
        v_y = 0
        v_z = 0
        p = 1
      end
#       if i == nx/2
#         rho = 0.45
#         vel = 0
#         p = (1 + 0.125 )
#       end
      v2 =  v_x*v_x + v_y*v_y * v_z*v_z
      vals_1[ :, i, j, k ] = [ rho, rho*v_x, rho*v_y, rho*v_z, rho*v2/2 + p/(gamma-1) ]
    end
  end
end

function setBounderies( vals )
  #Open boundaries
  #X boundery
  vals[:, 1,   :, :] = vals[:, 2,     :, :]
  vals[:, end, :, :] = vals[:, end-1, :, :]
  #Y boundery
  vals[:, :, 1,   :] = vals[:, :, 2,     :]
  vals[:, :, end, :] = vals[:, :, end-1, :]
  #Z boundery
  vals[:, :, :,   1] = vals[:, :, :,     2]
  vals[:, :, :, end] = vals[:, :, :, end-1]
end

function setBounderies( vals )
  #Open boundaries
  #X boundery
#   vals[:, 1,   :, :] = vals[:, 2,     :, :]  #For open boundery
  vals[1, 1,   :, :] = vals[1, 2,     :, :]
  vals[2, 1,   :, :] = -vals[2, 2,     :, :]
  vals[3, 1,   :, :] = vals[3, 2,     :, :]
  vals[4, 1,   :, :] = vals[4, 2,     :, :]
  vals[5, 1,   :, :] = vals[5, 2,     :, :]
  vals[:, end, :, :] = vals[:, end-1, :, :]
  #Y boundery
#   vals[:, :, 1,   :] = vals[:, :, 2,     :] #For open boundery
  vals[1, 1,   :, :] = vals[1, 2,     :, :]
  vals[2, 1,   :, :] = vals[2, 2,     :, :]
  vals[3, 1,   :, :] = -vals[3, 2,     :, :]
  vals[4, 1,   :, :] = vals[4, 2,     :, :]
  vals[5, 1,   :, :] = vals[5, 2,     :, :]
  vals[:, :, end, :] = vals[:, :, end-1, :]
  #Z boundery
#   vals[:, :, :,   1] = vals[:, :, :,     2] #For open boundery
  vals[1, 1,   :, :] = vals[1, 2,     :, :]
  vals[2, 1,   :, :] = vals[2, 2,     :, :]
  vals[3, 1,   :, :] = vals[3, 2,     :, :]
  vals[4, 1,   :, :] = -vals[4, 2,     :, :]
  vals[5, 1,   :, :] = vals[5, 2,     :, :]
  vals[:, :, :, end] = vals[:, :, :, end-1]
end

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

function doTimeStep( vals, vals_new, flx_x, flx_y, flx_z )
  setBounderies( vals )
  dt = getTimeMin( vals, flx_x, flx_y, flx_z )
  for k in 2:nz+1
    for j in 2:ny+1
      for i in 2:nx+1
        vals_r = vals[ :, i+1, j, k ] #Right
        vals_l = vals[ :, i-1, j, k ] #Left
        vals_u = vals[ :, i, j+1, k ] #Up
        vals_d = vals[ :, i, j-1, k ] #Down
        vals_t = vals[ :, i, j, k+1 ] #Top
        vals_b = vals[ :, i, j, k-1 ] #Bottom
        #Fluxes
        flx_r = flx_x[ :, i+1, j, k ] #Right
        flx_l = flx_x[ :, i-1, j, k ] #Left
        flx_u = flx_y[ :, i, j+1, k ] #Up
        flx_d = flx_y[ :, i, j-1, k ] #Down
        flx_t = flx_z[ :, i, j, k+1 ] #Top
        flx_b = flx_z[ :, i, j, k-1 ] #Bottom
        #First term
        t1 = ( vals_r + vals_l + vals_u + vals_d + vals_t + vals_b )/6
        #Seconf term
        t2 =  ( flx_r - flx_l )/(2*dx) + ( flx_u - flx_d )/(2*dy) + ( flx_t - flx_b )/(2*dz)
        vals_new[ :, i, j, k ] = t1 - dt*t2
      end
    end
  end
end



# ###############################################################
println( "\nStarting simulation" )


const nIter = 100
const nSnaps = 50
const iterPerSave = divrem( nIter, nSnaps )[1]
saveCount = 1

saveVals( outFile, vals_1, 0)
time_compute, time_save = 0, 0
printProgress( 0, nIter, time_compute )
for i in 1:nIter
#   println(i, )
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

