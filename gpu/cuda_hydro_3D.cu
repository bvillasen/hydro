#include <pycuda-helpers.hpp>

//Textures for conserv
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_1;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_2;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_3;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_4;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_5;


//Surfaces for Fluxes
surface< void, cudaSurfaceType3D> surf_flx_1;
surface< void, cudaSurfaceType3D> surf_flx_2;
surface< void, cudaSurfaceType3D> surf_flx_3;
surface< void, cudaSurfaceType3D> surf_flx_4;
surface< void, cudaSurfaceType3D> surf_flx_5;

// __device__ cudaP getPresure( cudaP gamma, cudaP rho, cudaP vel, cudaP E ){
//   return ( E  - rho*vel*vel/2 ) * (gamma-1);
// }

// __device__ float getSoundVel( cudaP gamma, cudaP rho, cudaP p ){
//   return float( sqrt( gamma * p / rho ) );
// }

// __global__ void setFlux( int coord, cudaP gamma,
// 			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5, 
// 			 float* soundVel2 ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
// 
//   cudaP rho, vx, vy, vz, p, E, v2;
//   rho = cnsv_1[ tid ];
//   vx  = cnsv_2[ tid ] / rho;
//   vy  = cnsv_3[ tid ] / rho;
//   vz  = cnsv_4[ tid ] / rho;
//   E   = cnsv_5[ tid ];
//   v2  = vx*vx + vy*vy + vz*vz;
//   p   = ( E - rho*v2/2 ) * (gamma-1);
//   
//   soundVel2[ tid ] = float(  p * gamma / rho );
//   
// //   //Get the fluxes
// //   cudaP f1, f2, f3, f4, f5;
// //   if ( coord == 1 ){
// //     f1 = rho * vx;
// //     f2 = rho * vx * vx + p;
// //     f3 = rho * vy * vx;
// //     f4 = rho * vz * vz;
// //     f5 = vx * ( E + p );    
// //   } 
// //   else if ( coord == 2){ 
// //     f1 = rho * vy;
// //     f2 = rho * vx * vy;
// //     f3 = rho * vy * vy + p;
// //     f4 = rho * vz * vy;
// //     f5 = vy * ( E + p );    
// //   } 
// //   else if ( coord == 3){ 
// //     f1 = rho * vz;
// //     f2 = rho * vx * vz;
// //     f3 = rho * vy * vz;
// //     f4 = rho * vz * vz + p;
// //     f5 = vz * ( E + p );
// //   }
// // 
// //   //Write fluxes to surfaces 
// //   surf3Dwrite(  f1, surf_flx_1,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
// //   surf3Dwrite(  f2, surf_flx_2,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);    
// //   surf3Dwrite(  f3, surf_flx_3,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
// //   surf3Dwrite(  f4, surf_flx_4,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp); 
// //   surf3Dwrite(  f5, surf_flx_5,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
// }
  
__global__ void setInterFlux_hll( const int coord, const cudaP gamma, const cudaP dx, const cudaP dy, const cudaP dz,
			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5, 
			 float* times ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP v2;
  cudaP rho_l, vx_l, vy_l, vz_l, E_l, p_l;
  cudaP rho_c, vx_c, vy_c, vz_c, E_c, p_c;
//   float time;
  //Read adjacent conserv
  if ( coord == 1 ){
    rho_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
    rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);
    
    vx_l  = fp_tex3D( tex_2, t_j-1, t_i, t_k) / rho_l;
    vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;
    
    vy_l  = fp_tex3D( tex_3, t_j-1, t_i, t_k) / rho_l;
    vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;
    
    vz_l  = fp_tex3D( tex_4, t_j-1, t_i, t_k) / rho_l;
    vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;
    
    E_l   = fp_tex3D( tex_5, t_j-1, t_i, t_k);
    E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);
    
    //Boundary bounce condition
    if ( t_j == 0 ) vx_l = -vx_c;
  }
  else if ( coord == 2 ){
    rho_l = fp_tex3D( tex_1, t_j, t_i-1, t_k);
    rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);
    
    vx_l  = fp_tex3D( tex_2, t_j, t_i-1, t_k) / rho_l;
    vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;
    
    vy_l  = fp_tex3D( tex_3, t_j, t_i-1, t_k) / rho_l;
    vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;
    
    vz_l  = fp_tex3D( tex_4, t_j, t_i-1, t_k) / rho_l;
    vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;
    
    E_l   = fp_tex3D( tex_5, t_j, t_i-1, t_k);
    E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);
    
    //Boundary bounce condition
    if ( t_i == 0 ) vy_l = -vy_c;  
  }
  else if ( coord == 3 ){
    rho_l = fp_tex3D( tex_1, t_j, t_i, t_k-1);
    rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);
    
    vx_l  = fp_tex3D( tex_2, t_j, t_i, t_k-1) / rho_l;
    vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;
    
    vy_l  = fp_tex3D( tex_3, t_j, t_i, t_k-1) / rho_l;
    vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;
    
    vz_l  = fp_tex3D( tex_4, t_j, t_i, t_k-1) / rho_l;
    vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;
    
    E_l   = fp_tex3D( tex_5, t_j, t_i, t_k-1);
    E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);
    
    //Boundary bounce condition
    if ( t_k == 0 ) vz_l = -vz_c;
  }
  

  
  v2    = vx_l*vx_l + vy_l*vy_l + vz_l*vz_l;
  p_l   = ( E_l - rho_l*v2/2 ) * (gamma-1);
  
  v2    = vx_c*vx_c + vy_c*vy_c + vz_c*vz_c;
  p_c   = ( E_c - rho_c*v2/2 ) * (gamma-1);
  
 
  cudaP cs_l, cs_c, s_l, s_c;
  cs_l = sqrt( p_l * gamma / rho_l );
  cs_c = sqrt( p_c * gamma / rho_c );
  
  

  if ( coord == 1 ){ 
    s_l = min( vx_l - cs_l, vx_c - cs_c );
    s_c = max( vx_l + cs_l, vx_c + cs_c );
    //Use v2 to save time minimum
    v2 = dx / ( abs( vx_c ) + cs_c );
    v2 = min( v2, dy / ( abs( vy_c ) + cs_c ) );
    v2 = min( v2, dz / ( abs( vz_c ) + cs_c ) );
    times[ tid ] = v2;
  }
  else if ( coord == 2 ){ 
    s_l = min( vy_l - cs_l, vy_c - cs_c );
    s_c = max( vy_l + cs_l, vy_c + cs_c );
  }  
  else if ( coord == 3 ){ 
    s_l = min( vz_l - cs_l, vz_c - cs_c );
    s_c = max( vz_l + cs_l, vz_c + cs_c );
  }
  
  // Adjacent fluxes from left and center cell
  cudaP F_l, F_c, iFlx;
  
  //iFlx rho
  if ( coord == 1 ){
    F_l = rho_l * vx_l;
    F_c = rho_c * vx_c;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vy_l;
    F_c = rho_c * vy_c;
  }  
  else if ( coord == 3 ){
    F_l = rho_l * vz_l;
    F_c = rho_c * vz_c;
  }  
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c - rho_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_flx_1,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
  
  //iFlx rho * vx
  if ( coord == 1 ){
    F_l = rho_l * vx_l * vx_l + p_l;
    F_c = rho_c * vx_c * vx_c + p_c;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vx_l * vy_l;
    F_c = rho_c * vx_c * vy_c;
  }  
  else if ( coord == 3 ){
    F_l = rho_l * vx_l * vz_l;
    F_c = rho_c * vx_c * vz_c;
  } 
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vx_c - rho_l*vx_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_flx_2,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
  
  //iFlx rho * vy 
  if ( coord == 1 ){
    F_l = rho_l * vy_l * vx_l ;
    F_c = rho_c * vy_c * vx_c ;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vy_l * vy_l + p_l;
    F_c = rho_c * vy_c * vy_c + p_c;
  }  
  else if ( coord == 3 ){
    F_l = rho_l * vy_l * vz_l;
    F_c = rho_c * vy_c * vz_c;
  } 
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vy_c - rho_l*vy_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_flx_3,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
  
  //iFlx rho * vz
  if ( coord == 1 ){
    F_l = rho_l * vz_l * vx_l ;
    F_c = rho_c * vz_c * vx_c ;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vz_l * vy_l ;
    F_c = rho_c * vz_c * vy_c ;
  }  
  else if ( coord == 3 ){
    F_l = rho_l * vz_l * vz_l + p_l ;
    F_c = rho_c * vz_c * vz_c + p_c ;
  } 
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vz_c - rho_l*vz_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_flx_4,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
    
  //iFlx E
  if ( coord == 1 ){
    F_l = vx_l * ( E_l + p_l ) ;
    F_c = vx_c * ( E_c + p_c ) ;
  }
  else if ( coord == 2 ){
    F_l = vy_l * ( E_l + p_l ) ;
    F_c = vy_c * ( E_c + p_c ) ;
  }  
  else if ( coord == 3 ){
    F_l = vz_l * ( E_l + p_l ) ;
    F_c = vz_c * ( E_c + p_c ) ;
  }
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( E_c - E_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_flx_5,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
 
}

__global__ void getInterFlux_hll( const int coord, const cudaP dt,  const cudaP gamma, const cudaP dx, const cudaP dy, const cudaP dz, 
			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5 ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  //Read inter-cell fluxes from textures
  
  cudaP iFlx1_l, iFlx2_l, iFlx3_l, iFlx4_l, iFlx5_l; 
  cudaP iFlx1_r, iFlx2_r, iFlx3_r, iFlx4_r, iFlx5_r; 
  cudaP delta;
  if ( coord == 1 ){
    delta = dt / dx;
    iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
    iFlx1_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);
    
    iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
    iFlx2_r = fp_tex3D( tex_2, t_j+1, t_i, t_k);
    
    iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
    iFlx3_r = fp_tex3D( tex_3, t_j+1, t_i, t_k);
    
    iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
    iFlx4_r = fp_tex3D( tex_4, t_j+1, t_i, t_k);
    
    iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
    iFlx5_r = fp_tex3D( tex_5, t_j+1, t_i, t_k);
  }
  else if ( coord == 2 ){
    delta = dt / dy;
    iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
    iFlx1_r = fp_tex3D( tex_1, t_j, t_i+1, t_k);
    
    iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
    iFlx2_r = fp_tex3D( tex_2, t_j, t_i+1, t_k);
    
    iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
    iFlx3_r = fp_tex3D( tex_3, t_j, t_i+1, t_k);
    
    iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
    iFlx4_r = fp_tex3D( tex_4, t_j, t_i+1, t_k);
    
    iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
    iFlx5_r = fp_tex3D( tex_5, t_j, t_i+1, t_k);
  }    
  else if ( coord == 3 ){
    delta = dt / dz;
    iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
    iFlx1_r = fp_tex3D( tex_1, t_j, t_i, t_k+1);
    
    iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
    iFlx2_r = fp_tex3D( tex_2, t_j, t_i, t_k+1);
    
    iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
    iFlx3_r = fp_tex3D( tex_3, t_j, t_i, t_k+1);
    
    iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
    iFlx4_r = fp_tex3D( tex_4, t_j, t_i, t_k+1);
    
    iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
    iFlx5_r = fp_tex3D( tex_5, t_j, t_i, t_k+1);
  } 
  //Advance the consv values 
  cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
  cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l );
  cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l );
  cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l );
  cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l );
}  
