#include <pyopencl-complex.h>

// 2d function

#ifndef ADDRESS_MODE
#define ADDRESS_MODE CLK_ADDRESS_CLAMP
#endif

void kernel fill_patch2(read_only image2d_t src,
							 const int offset_x, const int offset_y,
							 __global cfloat_t *dest, const int offset_dest){


  const sampler_t sampler = ADDRESSMODE |  CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);


  float val = read_imagef(src,
  						   sampler,
						  (int2)(i+offset_x,j+offset_y)).x;
  
  dest[i+Nx*j+offset_dest] = cfloat_new(val,0.f);

 
}

// 3d function
void kernel fill_patch3(read_only image3d_t src,
							 const int offset_x,
							 const int offset_y,
							 const int offset_z,
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
						  (int4)(i+offset_x,j+offset_y,k+offset_z,0)).x;
  
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

  // src are the padded patches
  // dest is the actual img to fill (block by block)
  // the kernels runs over the blocksize
  // x0,y0 are the first dims of the patch buffer to interpolate x0 --> x0+1


  int i = get_global_id(0);
  int j = get_global_id(1);

  //the Nblock sizes
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  // relative coords within image block
  float _x = 1.f*i/(Nx-1.f);
  float _y = 1.f*j/(Ny-1.f);	  


  // the coordinates in the image
  int i_im = i+x0*Nx-Nx/2;
  int j_im = j+y0*Ny-Ny/2;
  int index_im = i_im+j_im*Gx*Nx;

  // the index in the patches

  int stride1 = Npatch_x*Npatch_y;
  int stride2 = Npatch_x*Npatch_y*Gx;


  int index11 = (i+Npatch_x/2)+Npatch_x*(j+Npatch_y/2)
                +stride1*(x0-1)+stride2*(y0-1);

  int index12 = (i+Npatch_x/2-Nx)+Npatch_x*(j+Npatch_y/2)
                +stride1*x0+stride2*(y0-1);
  int index21 = (i+Npatch_x/2)+Npatch_x*(j+Npatch_y/2-Ny)
                +stride1*(x0-1)+stride2*y0;
  int index22 = (i+Npatch_x/2-Nx)+Npatch_x*(j+Npatch_y/2-Ny)
                +stride1*x0+stride2*y0;

  //interpolation weights
  float a11 = ((x0>0)&&(y0>0))?(1.f-_x)*(1.f-_y):0;
  float a12 = ((x0<Gx)&&(y0>0))?_x*(1.f-_y):0;
  float a21 = ((x0>0)&&(y0<Gy))?(1.f-_x)*_y:0;
  float a22 = ((x0<Gy)&&(y0<Gy))?_x*_y:0;


  if ((i_im>=0)&&(i_im<Nx*Gx)&&(j_im>=0)&&(j_im<Ny*Gy)){

    float nsum = a11*cfloat_abs(src[index11])+
                 a12*cfloat_abs(src[index12])+
                 a21*cfloat_abs(src[index21])+
                 a22*cfloat_abs(src[index22]);

    float wsum = a11+a12+a21+a22;


    dest[index_im] = nsum/wsum;

    //dest[index_im] = wsum;



  }
  //dest[index_im] = a11+a12+a21+a22;

  //dest[index_im] = cfloat_abs(src[index22]);

  //dest[index_im] = (i+Npatch_x/2-3*Nx/2);

}


//3d 
void kernel interpolate3( __global cfloat_t * src,
						 __global float * dest,
						 const int x0, const int y0,const int z0,
						 const int Gx,const int Gy,const int Gz,
						  const int Npatch_x,
						  const int Npatch_y,
						  const int Npatch_z){

  const sampler_t sampler = CLK_ADDRESS_CLAMP|  CLK_FILTER_NEAREST;

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  //the Nblock sizes
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  // relative coords within image block
  float _x = 1.f*i/(Nx-1.f);
  float _y = 1.f*j/(Ny-1.f);	  
  float _z = 1.f*k/(Nz-1.f);


  // the coordinates in the image
  int i_im = i+x0*Nx-Nx/2;
  int j_im = j+y0*Ny-Ny/2;
  int k_im = k+z0*Nz-Nz/2;

  int index_im = i_im+j_im*Gx*Nx+k_im*Gx*Gy*Nx*Ny;

  // the index in the patches

  int stride1 = Npatch_x*Npatch_y*Npatch_z;
  int stride2 = Npatch_x*Npatch_y*Npatch_z*Gx;
  int stride3 = Npatch_x*Npatch_y*Npatch_z*Gx*Gy;


  int index111 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*(x0-1)+stride2*(y0-1)+stride3*(z0-1);

  int index112 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*x0+stride2*(y0-1)+stride3*(z0-1);

  int index121 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*(x0-1)+stride2*y0+stride3*(z0-1);

  int index211 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*(x0-1)+stride2*(y0-1)+stride3*z0;

  int index122 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*x0+stride2*y0+stride3*(z0-1);

  int index221 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*(x0-1)+stride2*y0+stride3*z0;

  int index212 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*x0+stride2*(y0-1)+stride3*z0;

  int index222 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*x0+stride2*y0+stride3*z0;




  //interpolation weights

  float a111 = ((x0>0)&&(y0>0)&&(z0>0))?(1.f-_x)*(1.f-_y)*(1.f-_z):0;
  float a112 = ((x0<Gx)&&(y0>0)&&(z0>0))?_x*(1.f-_y)*(1.f-_z):0;
  float a121 = ((x0>0)&&(y0<Gy)&&(z0>0))?(1.f-_x)*_y*(1.f-_z):0;
  float a211 = ((x0>0)&&(y0>0)&&(z0<Gz))?(1.f-_x)*(1.f-_y)*_z:0;
  float a122 = ((x0<Gx)&&(y0<Gy)&&(z0>0))?_x*_y*(1.f-_z):0;
  float a221 = ((x0>0)&&(y0<Gy)&&(z0<Gz))?(1.f-_x)*_y*_z:0;
  float a212 = ((x0<Gx)&&(y0>0)&&(z0<Gz))?_x*(1.f-_y)*_z:0;
  float a222 = ((x0<Gx)&&(y0<Gy)&&(z0<Gz))?_x*_y*_z:0;

  float w111 = ((x0>0)&&(y0>0)&&(z0>0))?(1.f-_x)*(1.f-_y)*(1.f-_z)*cfloat_abs(src[index111]):0;
  float w112 = ((x0<Gx)&&(y0>0)&&(z0>0))?_x*(1.f-_y)*(1.f-_z)*cfloat_abs(src[index112]):0;
  float w121 = ((x0>0)&&(y0<Gy)&&(z0>0))?(1.f-_x)*_y*(1.f-_z)*cfloat_abs(src[index121]):0;
  float w211 = ((x0>0)&&(y0>0)&&(z0<Gz))?(1.f-_x)*(1.f-_y)*_z*cfloat_abs(src[index211]):0;
  float w122 = ((x0<Gx)&&(y0<Gy)&&(z0>0))?_x*_y*(1.f-_z)*cfloat_abs(src[index122]):0;
  float w221 = ((x0>0)&&(y0<Gy)&&(z0<Gz))?(1.f-_x)*_y*_z*cfloat_abs(src[index221]):0;
  float w212 = ((x0<Gx)&&(y0>0)&&(z0<Gz))?_x*(1.f-_y)*_z*cfloat_abs(src[index212]):0;
  float w222 = ((x0<Gx)&&(y0<Gy)&&(z0<Gz))?_x*_y*_z*cfloat_abs(src[index222]):0;

  //float a111 = ((x0>0)&&(y0>0)&&(z0>0))?(1.f-_x)*(1.f-_y)*(1.f-_z):0;
  //float a112 = ((x0<Gx)&&(y0>0)&&(z0>0))?_x*(1.f-_y)*(1.f-_z):0;
  //float a121 = ((x0>0)&&(y0<Gy)&&(z0>0))?(1.f-_x)*_y*(1.f-_z):0;
  //float a211 = ((x0>0)&&(y0>0)&&(z0<Gz))?(1.f-_x)*(1.f-_y)*_z:0;

  //float a122 = ((x0<Gx)&&(y0<Gy)&&(z0>0))?_x*_y*(1.f-_z):0;
  //float a221 = ((x0>0)&&(y0<Gy)&&(z0<Gz))?(1.f-_x)*_y*_z:0;
  //float a212 = ((x0<Gx)&&(y0>0)&&(z0<Gz))?_x*(1.f-_y)*_z:0;

  //float a222 = ((x0<Gx)&&(y0<Gy)&&(z0<Gz))?_x*_y*_z:0;

  if ((i_im>=0)&&(i_im<Nx*Gx)&&(j_im>=0)&&(j_im<Ny*Gy)&&(k_im>=0)&&(k_im<Nz*Gz)){

    float nsum  = a111+a112+a121+a211+a122+a212+a221+a222;;
    float wsum = w111+w112+w121+w211+w122+w212+w221+w222;
    dest[index_im] = wsum/nsum;

    //dest[index_im] = index121;


    //dest[index_im] = wsum;


  }

}
