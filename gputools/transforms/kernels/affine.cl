
#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

#ifndef DTYPE
#define DTYPE float
#endif


__kernel void affine3(__read_only image3d_t input,
	      			 __global DTYPE* output,
				 __constant float * mat)
{

  const sampler_t sampler = SAMPLER_ADDRESS | SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float x = i;
  float y = j;
  float z = k;
  
  float x2 = (mat[8]*z+mat[9]*y+mat[10]*x+mat[11]);
  float y2 = (mat[4]*z+mat[5]*y+mat[6]*x+mat[7]);
  float z2 = (mat[0]*z+mat[1]*y+mat[2]*x+mat[3]);

  //ensure correct sampling of image, see opencl 1.2 specification pg. 329
  x2 += 0.5f;
  y2 += 0.5f;
  z2 += 0.5f;
  

  float4 coord_norm = (float4)(x2,y2,z2,0.f);

  float pix = read_imagef(input,sampler,coord_norm).x;

  output[i+Nx*j+Nx*Ny*k] = pix;


}
