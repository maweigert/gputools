//2D



__kernel void filter_2_x(__global float * x,__global float * y,
						__global float * ux,__global float * uy,
						__global float * uxx,__global float * uxy,__global float * uyy)
					 {

  int i = get_global_id(0);
  int j = get_global_id(1);

  int Nx = get_global_size(0);

  float r_ux = 0.f;
  float r_uy = 0.f;
  float r_uxx = 0.f;
  float r_uxy = 0.f;
  float r_uyy = 0.f;

  int start = i-WIN_SIZE/2;

  const int h_start = max(0,WIN_SIZE/2-i);
  const int h_end = min(WIN_SIZE,Nx-i+WIN_SIZE/2);

  for (int ht = h_start; ht< h_end; ++ht){
       float x_val = x[start+ht+j*Nx];
       float y_val = y[start+ht+j*Nx];

	   r_ux += x_val;
	   r_uy += y_val;
       r_uxx += x_val*x_val;
	   r_uxy += y_val*y_val;
       r_uyy += x_val*y_val;
	  }

  ux[i+j*Nx] = r_ux;
  uy[i+j*Nx] = r_uy;
  uxx[i+j*Nx] = r_uxx;
  uxy[i+j*Nx] = r_uxy;
  uyy[i+j*Nx] = r_uyy;


}

__kernel void filter_2_y(__global float * input,
						__global float * output,
						const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = -INFINITY;

  int start = j-Nh/2;

  const int h_start = max(0,Nh/2-j);
  const int h_end = min(Nh,Ny-j+Nh/2);



  for (int ht = h_start; ht< h_end; ++ht)
	res = fmax(res,input[i+(start+ht)*Nx]);

  output[i+j*Nx] = res;
}


__kernel void min_2_x(__global float * input,
						__global float * output,
						const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);


  float res = INFINITY;

  int start = i-Nh/2;

  const int h_start = max(0,Nh/2-i);
  const int h_end = min(Nh,Nx-i+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	  res = fmin(res,input[start+ht+j*Nx]);

  output[i+j*Nx] = res;
}

__kernel void min_2_y(__global float * input,
						__global float * output,
						const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = INFINITY;

  int start = j-Nh/2;

  const int h_start = max(0,Nh/2-j);
  const int h_end = min(Nh,Ny-j+Nh/2);



  for (int ht = h_start; ht< h_end; ++ht)
	res = fmin(res,input[i+(start+ht)*Nx]);

  output[i+j*Nx] = res;
}


//3D

__kernel void max_3_x(__global float * input,
				    __global float * output,
					   const int Nh
					 ){

					 int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);




  float res = -INFINITY;

  int start = i-Nh/2;

  const int h_start = max(0,Nh/2-i);
  const int h_end = min(Nh,Nx-i+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	  res = fmax(res,input[start+ht+j*Nx+k*Nx*Ny]);

  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void max_3_y(__global float * input,
				    __global float * output,
					   const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);



  float res = -INFINITY;

  int start = j-Nh/2;

  const int h_start = max(0,Nh/2-j);
  const int h_end = min(Nh,Ny-j+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	res = fmax(res,input[i+(start+ht)*Nx+k*Nx*Ny]);


  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void max_3_z(__global float * input,
				    __global float * output,
					   const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = -INFINITY;

  int start = k-Nh/2;

  const int h_start = max(0,Nh/2-k);
  const int h_end = min(Nh,Nz-k+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	res = fmax(res,input[i+j*Nx+(start+ht)*Nx*Ny]);


  output[i+j*Nx+k*Nx*Ny] = res;
}


__kernel void min_3_x(__global float * input,
				    __global float * output,
					   const int Nh
					 ){

					 int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);




  float res = INFINITY;

  int start = i-Nh/2;

  const int h_start = max(0,Nh/2-i);
  const int h_end = min(Nh,Nx-i+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	  res = fmin(res,input[start+ht+j*Nx+k*Nx*Ny]);

  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void min_3_y(__global float * input,
				    __global float * output,
					   const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);




  float res = INFINITY;

  int start = j-Nh/2;

  const int h_start = max(0,Nh/2-j);
  const int h_end = min(Nh,Ny-j+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	res = fmin(res,input[i+(start+ht)*Nx+k*Nx*Ny]);


  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void min_3_z(__global float * input,
				    __global float * output,
					   const int Nh
					 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = INFINITY;

  int start = k-Nh/2;

  const int h_start = max(0,Nh/2-k);
  const int h_end = min(Nh,Nz-k+Nh/2);

  for (int ht = h_start; ht< h_end; ++ht)
	res = fmin(res,input[i+j*Nx+(start+ht)*Nx*Ny]);


  output[i+j*Nx+k*Nx*Ny] = res;
}
