import sys, time, os
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
#import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import h5py as h5

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
dataDir = "/home/bruno/Desktop/data/qTurbulence/"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )


from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, kernelMemoryInfo
from tools import ensureDirectory, printProgressTime

cudaP = "double"
nPoints = 128
useDevice = None
usingAnimation = False
showKernelMemInfo = False

for option in sys.argv:
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1]) 
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) } 
cudaPre, cudaPreComplex = precision[cudaP]


#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

Lx = 1.
Ly = 1.
Lz = 1.
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]
R = np.sqrt( X*X + Y*Y + Z*Z )

gamma = 7./5.
c0 = 0.5

#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
#Initialize openGL
if usingAnimation:
  import volumeRender
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.windowTitle = "Hydro 3D  nPoints={0}".format(nPoints)
  volumeRender.initGL()
  
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=usingAnimation)

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,4   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]

print "\nCompiling CUDA code"
cudaCodeFile = open("cuda_hydro_3D.cu","r")
cudaCodeString_raw = cudaCodeFile.read().replace( "cudaP", cudaP ) 
#cudaCodeString = cudaCodeString_raw % { 
  #"THREADS_PER_BLOCK":block3D[0]*block3D[1]*block3D[2], 
  #"B_WIDTH":block3D[0], "B_HEIGHT":block3D[1], "B_DEPTH":block3D[2],
  #'blockDim.x': block3D[0], 'blockDim.y': block3D[1], 'blockDim.z': block3D[2],
  #'gridDim.x': grid3D[0], 'gridDim.y': grid3D[1], 'gridDim.z': grid3D[2] }
cudaCodeString = cudaCodeString_raw
cudaCode = SourceModule(cudaCodeString)
#setFlux_kernel = cudaCode.get_function('setFlux') 
setInterFlux_hll_kernel = cudaCode.get_function('setInterFlux_hll')
getInterFlux_hll_kernel = cudaCode.get_function('getInterFlux_hll')
tex_1 = cudaCode.get_texref("tex_1")
tex_2 = cudaCode.get_texref("tex_2")
tex_3 = cudaCode.get_texref("tex_3")
tex_4 = cudaCode.get_texref("tex_4")
tex_5 = cudaCode.get_texref("tex_5")
surf_flx_1 = cudaCode.get_surfref("surf_flx_1")
surf_flx_2 = cudaCode.get_surfref("surf_flx_2")
surf_flx_3 = cudaCode.get_surfref("surf_flx_3")
surf_flx_4 = cudaCode.get_surfref("surf_flx_4")
surf_flx_5 = cudaCode.get_surfref("surf_flx_5")
########################################################################
convertToUCHAR = ElementwiseKernel(arguments="cudaP normaliztion, cudaP *values, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*( values[i]*normaliztion -1 ) );",
			      name = "sendModuloToUCHAR_kernel")
########################################################################
getTimeMin_kernel = ReductionKernel( np.dtype( cudaPre ),
			    neutral = "1e6",
			    arguments=" float delta, cudaP* cnsv_rho, cudaP* cnsv_vel, float* soundVel".replace("cudaP", cudaP),
			    map_expr = " delta / ( abs( cnsv_vel[i]/ cnsv_rho[i] ) +  soundVel[i]   )    ",
			    reduce_expr = "min(a,b)",
			    name = "getTimeMin_kernel")
################################################### 
def timeStep():
  for coord in [ 1, 2, 3]:
    #Bind textures to read conserved
    tex_1.set_array( cnsv1_array )
    tex_2.set_array( cnsv2_array )
    tex_3.set_array( cnsv3_array )
    tex_4.set_array( cnsv4_array )
    tex_5.set_array( cnsv5_array )
    #Bind surfaces to write inter-cell fluxes
    surf_flx_1.set_array( flx1_array )
    surf_flx_2.set_array( flx2_array )
    surf_flx_3.set_array( flx3_array )
    surf_flx_4.set_array( flx4_array )
    surf_flx_5.set_array( flx5_array )
    setInterFlux_hll_kernel( np.int32( coord ), cudaPre( gamma ), cudaPre(dx), cudaPre(dy), cudaPre(dz), cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, cnsv5_d, times_d,  grid=grid3D, block=block3D ) 
    if coord == 1: dt = c0 * gpuarray.min( times_d ).get()
    #Bind textures to read inter-cell fluxes
    tex_1.set_array( flx1_array )
    tex_2.set_array( flx2_array )
    tex_3.set_array( flx3_array )
    tex_4.set_array( flx4_array )
    tex_5.set_array( flx5_array ) 
    getInterFlux_hll_kernel( np.int32( coord ), cudaPre( dt ), cudaPre( gamma ), cudaPre(dx), cudaPre(dy), cudaPre(dz),
                          cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, cnsv5_d,  grid=grid3D, block=block3D ) 
    copy3D_cnsv1()
    copy3D_cnsv2()
    copy3D_cnsv3()
    copy3D_cnsv4()
    copy3D_cnsv5()
    
def stepFuntion():
  maxVal = ( gpuarray.max( cnsv1_d ) ).get()
  convertToUCHAR( cudaPre( 0.95/maxVal ), cnsv1_d, plotData_d)
  copyToScreenArray()
  
  timeStep()
  
########################################################################
if showKernelMemInfo: 
  #kernelMemoryInfo( setFlux_kernel, 'setFlux_kernel')
  #print ""
  kernelMemoryInfo( setInterFlux_hll_kernel, 'setInterFlux_hll_kernel')
  print ""
  kernelMemoryInfo( getInterFlux_hll_kernel, 'getInterFlux_hll_kernel')
  print ""
########################################################################
########################################################################
print "\nInitializing Data"  
initialMemory = getFreeMemory( show=True )
rho = np.zeros( X.shape, dtype=cudaPre )  #density
vx  = np.zeros( X.shape, dtype=cudaPre )
vy  = np.zeros( X.shape, dtype=cudaPre )
vz  = np.zeros( X.shape, dtype=cudaPre )
p   = np.zeros( X.shape, dtype=cudaPre )  #pressure 
#####################################################
#Initialize a centerd sphere
rho[ R <= 0.2 ] = 1.
rho[ R >  0.2 ] = 0.7
p[ R <= 0.2 ] = 10
p[ R >  0.2 ] = 1
v2 = vx*vx + vy*vy + vz*vz 
#####################################################
#Initialize conserved values 
cnsv1_h = rho
cnsv2_h = rho * vx
cnsv3_h = rho * vy
cnsv4_h = rho * vz
cnsv5_h = rho*v2/2. + p/(gamma-1)

#####################################################
#Initialize device global data 
cnsv1_d = gpuarray.to_gpu( cnsv1_h )
cnsv2_d = gpuarray.to_gpu( cnsv2_h )
cnsv3_d = gpuarray.to_gpu( cnsv3_h )
cnsv4_d = gpuarray.to_gpu( cnsv4_h )
cnsv5_d = gpuarray.to_gpu( cnsv5_h )
times_d = gpuarray.to_gpu( np.zeros( X.shape, dtype=np.float32 ) )
#Texture and surface arrays
cnsv1_array, copy3D_cnsv1 = gpuArray3DtocudaArray( cnsv1_d, allowSurfaceBind=True, precision=cudaP )
cnsv2_array, copy3D_cnsv2 = gpuArray3DtocudaArray( cnsv2_d, allowSurfaceBind=True, precision=cudaP )
cnsv3_array, copy3D_cnsv3 = gpuArray3DtocudaArray( cnsv3_d, allowSurfaceBind=True, precision=cudaP )
cnsv4_array, copy3D_cnsv4 = gpuArray3DtocudaArray( cnsv4_d, allowSurfaceBind=True, precision=cudaP )
cnsv5_array, copy3D_cnsv5 = gpuArray3DtocudaArray( cnsv5_d, allowSurfaceBind=True, precision=cudaP )

flx1_array, copy3D_flx1_1 = gpuArray3DtocudaArray( cnsv1_d, allowSurfaceBind=True, precision=cudaP )
flx2_array, copy3D_flx2_1 = gpuArray3DtocudaArray( cnsv2_d, allowSurfaceBind=True, precision=cudaP )
flx3_array, copy3D_flx3_1 = gpuArray3DtocudaArray( cnsv3_d, allowSurfaceBind=True, precision=cudaP )
flx4_array, copy3D_flx4_1 = gpuArray3DtocudaArray( cnsv4_d, allowSurfaceBind=True, precision=cudaP )
flx5_array, copy3D_flx5_1 = gpuArray3DtocudaArray( cnsv5_d, allowSurfaceBind=True, precision=cudaP )
if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6) 




start, end = cuda.Event(), cuda.Event()

start.record() # start timing
 

end.record(), end.synchronize()

secs = start.time_till( end )*1e-3
print 'Time: {0:0.4f}'.format( secs )

#configure volumeRender functions  
if usingAnimation: 
  #volumeRender.viewTranslation[2] = -2
  volumeRender.transferScale = np.float32( 2.8 )
  #volumeRender.keyboard = keyboard
  #volumeRender.specialKeys = specialKeyboardFunc
  volumeRender.stepFunc = stepFuntion
  #run volumeRender animation
  volumeRender.animate()










