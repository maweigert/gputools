//2D

__kernel void conv_sep2_x(__global DTYPE * input,
						__constant float * h,
						__global DTYPE * output,
						  const int Nh, const int Nx0, const int stride){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = 0.f;

  const int h_start = ((i*stride+Nh/2)>=Nx0)?i*stride+Nh/2+1-Nx0:0;
  const int h_end = ((i*stride-Nh/2)<0)?i*stride+Nh/2+1:Nh;
  const int start = i*stride+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*(float)input[start-ht+j*Nx0];

  /* scipy casts at every seperable conv step */
  output[i+j*Nx] = (DTYPE)res;
}

__kernel void conv_sep2_y(__global DTYPE * input,
						__constant float * h,
                          __global DTYPE * output,
						  const int Nh, const int Ny0, const int stride){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = 0.f;


  const int h_start = ((j*stride+Nh/2)>=Ny0)?j*stride+Nh/2+1-Ny0:0;
  const int h_end = ((j*stride-Nh/2)<0)?j*stride+Nh/2+1:Nh;
  const int start = j*stride+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[(start-ht)*Nx+i];

  output[i+j*Nx] = (DTYPE)res;
  
}



//3d

__kernel void conv_sep3_x(__global DTYPE * input,
						__constant float * h,
						__global DTYPE * output,
						  const int Nh, const int Nx0, const int stride){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = 0.f;

  const int h_start = ((i*stride+Nh/2)>=Nx0)?i*stride+Nh/2+1-Nx0:0;
  const int h_end = ((i*stride-Nh/2)<0)?i*stride+Nh/2+1:Nh;
  const int start = i*stride+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[start-ht+j*Nx0+k*Nx0*Ny];

	
  output[i+j*Nx+k*Nx*Ny] = (DTYPE)res;


}

__kernel void conv_sep3_y(__global DTYPE * input,
						__constant float * h,
						__global DTYPE * output,
						  const int Nh, const int Ny0, const int stride){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = 0.f;

  const int h_start = ((j*stride+Nh/2)>=Ny0)?j*stride+Nh/2+1-Ny0:0;
  const int h_end = ((j*stride-Nh/2)<0)?j*stride+Nh/2+1:Nh;
  const int start = j*stride+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[i+(start-ht)*Nx+k*Nx*Ny0];

  /* if ((i==11) && (j==14) && (k==12)) */
  /*   printf("YYY  %.8f   \n", res); */
	
  output[i+j*Nx+k*Nx*Ny] = (DTYPE)res;
}

__kernel void conv_sep3_z(__global DTYPE * input,
						__constant float * h,
						__global DTYPE * output,
						  const int Nh, const int Nz0, const int stride){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);


  float res = 0.f;

  const int h_start = ((k*stride+Nh/2)>=Nz0)?k*stride+Nh/2+1-Nz0:0;
  const int h_end = ((k*stride-Nh/2)<0)?k*stride+Nh/2+1:Nh;
  const int start = k*stride+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[i+j*Nx+(start-ht)*Nx*Ny];

  output[i+j*Nx+k*Nx*Ny] = (DTYPE)res;

}
