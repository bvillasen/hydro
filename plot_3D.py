import numpy as np
import time, sys, os
import matplotlib.pyplot as plt
import h5py as h5

currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
sys.path.extend( [parentDirectory] )



nSnap = 0
changeSnapshot = False

dataDir = "/home/bruno/Desktop/data/hydro/"
dataFileName = dataDir + "finVol3D_01.h5"
dataFile = h5.File( dataFileName, 'r' )

box_x, box_y, box_z = dataFile['box']['x'][...], dataFile['box']['y'][...], dataFile['box']['z'][...] 
nSnapshots = len( dataFile['vals'].keys() )
data = dataFile['vals'][str(nSnap)][...]
dens = data[:,:,:, 0]
vel  = data[:,:,:, 1]
nWidth, nHeight, nDepth = dens.shape
dataToPlot = dens

print 'nSnapshots: ', nSnapshots

##############################################################
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#Add Modules from other directories
devDir = '/home/bruno/Desktop/Dropbox/Developer/'
toolsDir = devDir + "pyCUDA/tools"
volumeRenderDir = devDir + 'pyCUDA/volumeRender'
sys.path.extend( [ toolsDir, volumeRenderDir] )
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray
import volumeRender
#Initialize openGL
volumeRender.volRenderDirectory = volumeRenderDir
nWidth, nHeight, nDepth = dataToPlot.shape
volumeRender.nWidth = nWidth
volumeRender.nHeight = nHeight
volumeRender.nDepth = nDepth
volumeRender.initGL()
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=0, usingAnimation=True )

##Read and compile CUDA code
print "\nCompiling CUDA code"
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));")
########################################################################
def preparePlotData( inputData ):
  plotData = inputData.astype(np.float32)
  plotData = np.abs( plotData )
  #plotData = np.log10( 100*plotData + 1 )
  plotData = plotData * 0.85
  return plotData
########################################################################
def sendToScreen( plotData ):
  floatToUchar( plotData, plotData_d )
  copyToScreenArray()
########################################################################
def stepFunction():
  global changeSnapshot
  if changeSnapshot: 
    changeData( nSnap )
    changeSnapshot = False
    
  sendToScreen( data_d )
########################################################################
  
  
#Initialize all gpu data
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )  
data_h = preparePlotData( dataToPlot )
data_d = gpuarray.to_gpu( data_h )
#memory for plotting
plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 

def changeData( nSnap ):
  global dataToPlot, data_d, changeSnapshot, ax1, ax2
  print "\nPlotting snapshot: {0}\n".format(nSnap)
  #fileBase = dataDir + 'snapshots/psi_{0}_'.format( snapshot )
  data = dataFile['vals'][str(nSnap)][...]
  dens = data[:,:,:, 0]
  vel  = data[:,:,:, 1] / dens
  data_h = dens 
  data_h = preparePlotData( data_h )
  data_d = gpuarray.to_gpu( data_h )
  #For 1D cut
  dens_x = dens[0, 0, :]
  vel_x  = vel[0, 0, :]/dens_x
  ax1.clear(), ax2.clear()
  ax1.plot( box_x, dens_x )
  ax2.plot( box_x, vel_x )
  ax1.set_ylim( 0, 1.5 )
  ax2.set_ylim( -5, 5 )
  plt.draw()
  changeSnapshot = False


def specialKeyboardFunc( key, x, y ):
  global nSnap, changeSnapshot
  if key== volumeRender.GLUT_KEY_RIGHT:
    nSnap += 1
    if nSnap == nSnapshots: nSnap = 0
    changeSnapshot = True
  if key== volumeRender.GLUT_KEY_LEFT:
    nSnap -= 1
    if nSnap == -1: nSnap = nSnapshots-1
    changeSnapshot = True 


plt.ion(), plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
dens_x = data[0,0,:, 0]
vel_x  = data[0,0,:, 1]/dens_x
ax1.clear(), ax2.clear()

ax1.plot( box_x, dens_x )
ax2.plot( box_x, vel_x )
ax1.set_ylim( 0, 1.5 )
ax2.set_ylim( 0, 10 )
ax1.set_xlim(box_x.min(), box_x.max())
plt.draw()
  


#change volumeRender default step function 
volumeRender.stepFunc = stepFunction
volumeRender.specialKeys = specialKeyboardFunc
#run volumeRender animation
volumeRender.animate()









#dataFile  = 'snapshots/regularGrid_{1}_dim{0}_nRef{2}.h5'.format(dim, snapshot, nRef)

#print "\nLoading data..."   
#start = time.time()
#h5Data = h5.File( dataDir + dataFile, 'r' )
##Gas Data
#gasGrp = h5Data['gas']
#gasTemp = gasGrp['temp'][...]
#gasDens = gasGrp['dens'][...]
#gasDens_smt = gasGrp['dens_smt'][...]
#h5Data.close()
#print " Data Loaded: {0}".format( dataDir + dataFile )
#print " Time: ", time.time() - start

#dataToPlot = gasDens
#nWidth, nHeight, nDepth = dataToPlot.shape

#rndData = np.random.rand( nWidth, nHeight, nDepth )
#dataToPlot = rndData

