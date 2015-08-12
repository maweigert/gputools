__kernel void convolve1d_buf(__global float * input,
	 		     __constant float * h,
			     __global float * output,
			     const int Nhx){

  int i = get_global_id(0);
  
  int Nx = get_global_size(0);
  
  float res = 0.f;

  int startx = i-Nhx/2;
  
  const int hx_start = ((i-Nhx/2)<0)?Nhx/2-i:0;
  const int hx_end = ((i+Nhx/2)>=Nx)?Nhx-(i+Nhx/2-Nx+1):Nhx;

  
  for (int htx = hx_start; htx< hx_end; ++htx)
      res += h[htx]*input[startx+htx];

  output[i] = res;
}


__kernel void convolve2d_buf(__global float * input,
	 		     __constant float * h,
			     __global float * output,
			     const int Nhy, const int Nhx){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = 0.f;

  int startx = i-Nhx/2;
  int starty = j-Nhy/2;

  const int hx_start = ((i-Nhx/2)<0)?Nhx/2-i:0;
  const int hx_end = ((i+Nhx/2)>=Nx)?Nhx-(i+Nhx/2-Nx+1):Nhx;

  const int hy_start = ((j-Nhy/2)<0)?Nhy/2-j:0;
  const int hy_end = ((j+Nhy/2)>=Ny)?Nhy-(j+Nhy/2-Ny+1):Nhy;

  for (int htx = hx_start; htx< hx_end; ++htx)
    for (int hty = hy_start; hty< hy_end; ++hty)
      res += h[htx+Nhx*hty]*input[startx+htx+(starty+hty)*Nx];

	
  output[i+j*Nx] = res;
}



__kernel void convolve3d_buf(__global float * input,
	 		     __constant float * h,
			     __global float * output,
			     const int Nhz,const int Nhy, const int Nhx){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float res = 0.f;

  int startx = i-Nhx/2;
  int starty = j-Nhy/2;
  int startz = k-Nhz/2;	
  
  const int hx_start = ((i-Nhx/2)<0)?Nhx/2-i:0;
  const int hx_end = ((i+Nhx/2)>=Nx)?Nhx-(i+Nhx/2-Nx+1):Nhx;

  const int hy_start = ((j-Nhy/2)<0)?Nhy/2-j:0;
  const int hy_end = ((j+Nhy/2)>=Ny)?Nhy-(j+Nhy/2-Ny+1):Nhy;

  const int hz_start = ((k-Nhz/2)<0)?Nhz/2-k:0;
  const int hz_end = ((k+Nhz/2)>=Nz)?Nhz-(k+Nhz/2-Nz+1):Nhz;

  for (int htx = hx_start; htx< hx_end; ++htx)
    for (int hty = hy_start; hty< hy_end; ++hty)
    	for (int htz = hz_start; htz< hz_end; ++htz)	    
  	res += h[htx+Nhx*hty+Nhx*Nhy*htz]*
	       input[startx+htx+(starty+hty)*Nx+(startz+htz)*Ny*Nx];

	
  output[i+j*Nx+k*Nx*Ny] = res;
}
