
#ifdef USENEAREST
#define SAMPLERFILTER CLK_FILTER_NEAREST
#else
#define SAMPLERFILTER CLK_FILTER_LINEAR
#endif





__kernel void scale(__read_only image3d_t input, __global TYPENAME* output)
{

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	SAMPLERFILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);


  float x = i + 0.5f;
  float y = j + 0.5f;
  float z = k + 0.5f;

    /*
  TYPENAME pix = READ_IMAGE(input,sampler,(float4)(1.f*x/(Nx-1.f),
						 1.f*y/(Ny-1.f),
						 1.f*z/(Nz-1.f),0)).x;
	*/

  TYPENAME pix = READ_IMAGE(input,sampler,(float4)(1.f*x/Nx,
						 1.f*y/Ny,
						 1.f*z/Nz,0)).x;

  
  output[i+Nx*j+Nx*Ny*k] = pix;
  

}

