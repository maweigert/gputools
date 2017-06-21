
#ifdef USENEAREST
#define SAMPLERFILTER CLK_FILTER_NEAREST
#else
#define SAMPLERFILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

__kernel void affine(__read_only image3d_t input,
	      			 __global float* output,
				 __constant float * mat)
{

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float x = (mat[0]*i+mat[1]*j+mat[2]*k+mat[3]);
  float y = (mat[4]*i+mat[5]*j+mat[6]*k+mat[7]);
  float z = (mat[8]*i+mat[9]*j+mat[10]*k+mat[11]);

  //ensure correct sampling, see opencl 1.2 specification pg. 329
  x += 0.5f;
  y += 0.5f;
  z += 0.5f;

  float4 coord_norm = (float4)(x/Nx,y/Ny,z/Nz,0.f);

  float pix = read_imagef(input,sampler,coord_norm).x;

  output[i+Nx*j+Nx*Ny*k] = pix;


}
