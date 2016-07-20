#include <pyopencl-complex.h>

// 2d function
void kernel fill_patch2(read_only image2d_t src,
							 const int offset_x, const int offset_y,
							 __global cfloat_t *dest, const int offset_dest){

  const sampler_t sampler = CLK_ADDRESS_CLAMP|  CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float val = read_imagef(src,
  						   sampler,
						  (int2)(i+offset_x,j+offset_y)).x;
  
  dest[i+Nx*j+offset_dest] = cfloat_new(val,0.f);

 
}

// 3d function
void kernel fill_patch3(read_only image3d_t src,
							 const int offset_x, const int offset_y,
							 __global cfloat_t *dest, const int offset_dest){

  const sampler_t sampler = CLK_ADDRESS_CLAMP|  CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float val = read_imagef(src,
  						   sampler,
						  (int4)(i+offset_x,j+offset_y,k,0)).x;
  
  dest[i+Nx*j+Nx*Ny*k+offset_dest] = cfloat_new(val,0.f);

 
}



void kernel fill_patch2_buf( __global float * src,
						   const int src_Nx, const int src_Ny,
						   const int offset_x, const int offset_y,
						   __global cfloat_t *dest, const int offset_dest){

  const sampler_t sampler = CLK_ADDRESS_CLAMP|  CLK_FILTER_NEAREST;

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  int i2 = i+offset_x;
  int j2 = j+offset_y;
  

  //clamp to boundary
  float val = ((i2>=0)&&(i2<src_Nx)&&(j2>=0)&&(j2<src_Ny))?src[i2+src_Nx*j2]:0.f;
  
  dest[i+Nx*j+offset_dest] = cfloat_new(val,0.f);

}


//2d 
void kernel interpolate2( __global cfloat_t * src,
						 __global float * dest,
						 const int x0, const int y0,
						 const int Gx,const int Gy,
						 const int Npatch_x,const int Npatch_y){

  const sampler_t sampler = CLK_ADDRESS_CLAMP|  CLK_FILTER_NEAREST;

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  float _x = 1.f*i/(Nx-1.f);
  float _y = 1.f*j/(Ny-1.f);	  


  // the coordinates in the image
  int i_im = i+x0*Nx;
  int j_im = j+y0*Ny;
  int index_im = i_im+j_im*(Gx-1)*Nx;

  // the index in the patches

  
  int index11 = (i+Npatch_x/2)+Npatch_x*(j+Npatch_x/2)+Npatch_x*Npatch_y*x0+Npatch_x*Npatch_y*Gx*y0;
  
  int index12 = (i+Npatch_x/2-Nx)+Npatch_x*(j+Npatch_x/2)+Npatch_x*Npatch_y*(x0+1)+Npatch_x*Npatch_y*Gx*y0;

  int index21 = (i+Npatch_x/2)+Npatch_x*(j+Npatch_x/2-Ny)+Npatch_x*Npatch_y*x0+Npatch_x*Npatch_y*Gx*(y0+1);

  int index22 = (i+Npatch_x/2-Nx)+Npatch_x*(j+Npatch_x/2-Ny)+Npatch_x*Npatch_y*(x0+1)+Npatch_x*Npatch_y*Gx*(y0+1);
  
  

  float a11 = (1.f-_x)*(1.f-_y)*cfloat_real(src[index11]);
  float a12 = _x*(1.f-_y)*cfloat_real(src[index12]);
  float a21 = (1.f-_x)*_y*cfloat_real(src[index21]);
  float a22 = _x*_y*cfloat_real(src[index22]);
  
  dest[index_im] = a11+a12+a21+a22;
  // dest[index_im] = cfloat_real(src[index21]);

}


//3d 
void kernel interpolate3( __global cfloat_t * src,
						 __global float * dest,
						 const int x0, const int y0,
						 const int Gx,const int Gy,
						  const int Npatch_x,const int Npatch_y){

  const sampler_t sampler = CLK_ADDRESS_CLAMP|  CLK_FILTER_NEAREST;

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);
  
  float _x = 1.f*i/(Nx-1.f);
  float _y = 1.f*j/(Ny-1.f);	  


  // the coordinates in the image
  int i_im = i+x0*Nx;
  int j_im = j+y0*Ny;
  
  int index_im = i_im+j_im*(Gx-1)*Nx+k*(Gx-1)*Nx*(Gy-1)*Ny;

  int stride = Nz*Npatch_x*Npatch_y;
  
  // the index in the patches
  int index11 = (i+Npatch_x/2)+Npatch_x*(j+Npatch_x/2)+Npatch_x*Npatch_y*k
	+stride*x0+stride*Gx*y0;
  
  int index12 = (i+Npatch_x/2-Nx)+Npatch_x*(j+Npatch_x/2)+Npatch_x*Npatch_y*k
	+stride*(x0+1)+stride*Gx*y0;
  
  int index21 = (i+Npatch_x/2)+Npatch_x*(j+Npatch_x/2-Ny)+Npatch_x*Npatch_y*k
	+stride*x0+stride*Gx*(y0+1);

  int index22 = (i+Npatch_x/2-Nx)+Npatch_x*(j+Npatch_x/2-Ny)+Npatch_x*Npatch_y*k
	+stride*(x0+1)+stride*Gx*(y0+1);
  
  

  float a11 = (1.f-_x)*(1.f-_y)*cfloat_real(src[index11]);
  float a12 = _x*(1.f-_y)*cfloat_real(src[index12]);
  float a21 = (1.f-_x)*_y*cfloat_real(src[index21]);
  float a22 = _x*_y*cfloat_real(src[index22]);
  
  dest[index_im] = a11+a12+a21+a22;
  
}


