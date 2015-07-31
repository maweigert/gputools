#ifdef SHORTTYPE
#define READ_IMAGE read_imageui
#define DTYPE short
#else
#define READ_IMAGE read_imagef
#define DTYPE float
#endif

__kernel void convolve1d(__read_only image2d_t input,__global float* h,__global float* output,const int Nx,const int Nhx){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);

  float res = 0.f;

  for (int i = 0; i < Nhx; ++i){
	float dx = -.5f*(Nhx-1)+i;
	res += h[i]*READ_IMAGE(input,sampler,(float2)(i0+dx,0)).x;
  }
  
  output[i0] = res;  

}
__kernel void convolve2d(__read_only image2d_t input,__global float* h,__global float* output,const int Nx,const int Ny,const int Nhx,const int Nhy){
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);


  float res = 0.f;

  for (int i = 0; i < Nhx; ++i){
	  for (int j = 0; j < Nhy; ++j){

		float dx = -.5f*(Nhx-1)+i;
		float dy = -.5f*(Nhy-1)+j;
		
		res += h[i+Nhx*j]*READ_IMAGE(input,sampler,(float2)(i0+dx,j0+dy)).x;

	  }
  }
  output[i0+j0*Nx] = res;
}

__kernel void convolve3d(__read_only image3d_t input,__global float* h,__global float* output,const int Nx,const int Ny,const int Nz,const int Nhx,const int Nhy,const int Nhz){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);


  float res = 0.f;

  for (int i = 0; i < Nhx; ++i){
  	  for (int j = 0; j < Nhy; ++j){
  		for (int k = 0; k < Nhz; ++k){

  		  float dx = -.5f*(Nhx-1)+i;
  		  float dy = -.5f*(Nhy-1)+j;
  		  float dz = -.5f*(Nhz-1)+k;
		
  		  res += h[i+Nhx*j+Nhx*Nhy*k]*READ_IMAGE(input,sampler,(float4)(i0+dx,j0+dy,k0+dz,0)).x;
  		}
  	  }
  }
  
  output[i0+j0*Nx+k0*Nx*Ny] = res;  

}

// separable versions

__kernel void convolve_sep2d_float(__read_only image2d_t input, __global float * h, const int N,__write_only image2d_t output, const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  
  const int dx = flag & 1;
  const int dy = (flag&2)/2;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
    res += h[i]*read_imagef(input,sampler,(float2)(i0+dx*j,j0+dy*j)).x;
  }

  write_imagef(output,(int2)(i0,j0),(float4)(res,0,0,0));
  
}


__kernel void convolve_sep2d_short(__read_only image2d_t input, __global float * h, const int N,__write_only image2d_t output, const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  
  const int dx = flag & 1;
  const int dy = (flag&2)/2;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
    res += h[i]*read_imageui(input,sampler,(float2)(i0+dx*j,j0+dy*j)).x;
  }

  write_imageui(output,(int2)(i0,j0),(uint4)(res,0,0,0));
  
}



__kernel void convolve_sep3d_float(__read_only image3d_t input, __global float * h, const int N,__write_only image3d_t output,const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  // flag = 4 -> in z axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);
  uint k0 = get_global_id(2);

  const int dx = flag & 1;
  const int dy = (flag&2)/2;
  const int dz = (flag&4)/4;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
	res += h[i]*read_imagef(input,sampler,(float4)(i0+dx*j,j0+dy*j,k0+dz*j,0)).x;
  }

  write_imagef(output,(int4)(i0,j0,k0,0),(float4)(res,0,0,0));
  
}


__kernel void convolve_sep3d_short(__read_only image3d_t input, __global float * h, const int N,__write_only image3d_t output,const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  // flag = 4 -> in z axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);
  uint k0 = get_global_id(2);

  const int dx = flag & 1;
  const int dy = (flag&2)/2;
  const int dz = (flag&4)/4;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
	res += h[i]*read_imageui(input,sampler,(float4)(i0+dx*j,j0+dy*j,k0+dz*j,0)).x;
  }

  // if ((i0==100)&&(j0==100)&&(k0==40))
  // 	  printf("kernel %.3f   %.3f\n",read_imageui(input,sampler,(float4)(i0,j0,k0,0)).x,res);

  write_imageui(output,(int4)(i0,j0,k0,0),(uint4)(res,0,0,0));
  
}








__kernel void foo(__read_only image3d_t input,__global float* h,const int Nx,const int Ny,const int Nz,__write_only image3d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);


  float res = 0.f;
  write_imagef(output,(int4)(i0,j0,k0,0),(float4)(i0,0,0,0));
  

}
