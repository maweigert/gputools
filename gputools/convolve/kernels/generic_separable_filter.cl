

//2D

__kernel void filter_2_x(__global ${DTYPE} * input,
                         __global ${DTYPE} * output, const int Nx0){

  const int i = get_global_id(0);
  const int j = get_global_id(1);

  const int Nx = get_global_size(0);


  ${DTYPE} res = ${DEFAULT};
  int start = i*${STRIDE_X}-${FSIZE_X}/2;


  const int h_start = max(0,${FSIZE_X}/2-i*${STRIDE_X});
  const int h_end = min(${FSIZE_X},Nx0-i*${STRIDE_X}+${FSIZE_X}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[start+ht+j*Nx0];
	  res = ${FUNC};
	  }

  output[i+j*Nx] = res;
}

__kernel void filter_2_y(__global ${DTYPE} * input,
						__global ${DTYPE} * output, const int Ny0){

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  
  const int Nx = get_global_size(0);
  const int Ny = get_global_size(1);

  ${DTYPE} res = ${DEFAULT};

  int start = j*${STRIDE_Y}-${FSIZE_Y}/2;

  const int h_start = max(0,${FSIZE_Y}/2-j*${STRIDE_Y});
  const int h_end = min(${FSIZE_Y},Ny0-j*${STRIDE_Y}+${FSIZE_Y}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+(start+ht)*Nx];
	res = ${FUNC};
	}

  output[i+j*Nx] = res;
}

//3D

__kernel void filter_3_x(__global ${DTYPE} * input,
				    __global ${DTYPE} * output, const int Nx0, const int Ny0){

					 int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  ${DTYPE} res = ${DEFAULT};

  int start = i-${FSIZE_X}/2;

  const int h_start = max(0,${FSIZE_X}/2-i*${STRIDE_X});
  const int h_end = min(${FSIZE_X},Nx0-i*${STRIDE_X}+${FSIZE_X}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[start+ht+j*Nx0+k*Nx0*Ny0];
    res = ${FUNC};
	  }

  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void filter_3_y(__global ${DTYPE} * input,
                         __global ${DTYPE} * output, const int Nx0, const int Ny0){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);



  ${DTYPE} res = ${DEFAULT};

  int start = j-${FSIZE_Y}/2;

  const int h_start = max(0,${FSIZE_Y}/2-j*${STRIDE_Y});
  const int h_end = min(${FSIZE_Y},Ny0-j*${STRIDE_Y}+${FSIZE_Y}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+(start+ht)*Nx0*+k*Nx0*Ny0];
	res = ${FUNC};
	}


  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void filter_3_z(__global ${DTYPE} * input,
				    __global ${DTYPE} * output, const int Nz0){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  ${DTYPE} res = ${DEFAULT};

  int start = k-${FSIZE_Z}/2;

  const int h_start = max(0,${FSIZE_Z}/2-k);
  const int h_end = min(${FSIZE_Z},Nz-k+${FSIZE_Z}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+j*Nx+(start+ht)*Nx*Ny];
	res = ${FUNC};
	}


  output[i+j*Nx+k*Nx*Ny] = res;
}
