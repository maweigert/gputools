

//2D

__kernel void filter_2_x(__global ${DTYPE} * input,
                         __global ${DTYPE} * output, const int Nx0, const int stride){

  const int i = get_global_id(0);
  const int j = get_global_id(1);

  const int Nx = get_global_size(0);


  float res = ${DEFAULT};
  int start = i*stride-${FSIZE_X}/2;


  const int h_start = max(0,${FSIZE_X}/2-i*stride);
  const int h_end = min(${FSIZE_X},Nx0-i*stride+${FSIZE_X}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[start+ht+j*Nx0];
	  res = ${FUNC};
	  }

  output[i+j*Nx] = (${DTYPE})(res);
}

__kernel void filter_2_y(__global ${DTYPE} * input,
						__global ${DTYPE} * output, const int Ny0, const int stride){

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  
  const int Nx = get_global_size(0);

  float res = ${DEFAULT};

  int start = j*stride-${FSIZE_Y}/2;

  const int h_start = max(0,${FSIZE_Y}/2-j*stride);
  const int h_end = min(${FSIZE_Y},Ny0-j*stride+${FSIZE_Y}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+(start+ht)*Nx];
	res = ${FUNC};
	}

  output[i+j*Nx] = (${DTYPE})(res);
}

//3D

__kernel void filter_3_x(__global ${DTYPE} * input,
				    __global ${DTYPE} * output, const int Nx0, const int stride){

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const int Nx = get_global_size(0);
  const int Ny = get_global_size(1);

  float res = ${DEFAULT};

  int start = i*stride-${FSIZE_X}/2;

  const int h_start = max(0,${FSIZE_X}/2-i*stride);
  const int h_end = min(${FSIZE_X},Nx0-i*stride+${FSIZE_X}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[start+ht+j*Nx0+k*Nx0*Ny];
    res = ${FUNC};
	  }

  output[i+j*Nx+k*Nx*Ny] = (${DTYPE})(res);
}

__kernel void filter_3_y(__global ${DTYPE} * input,
                         __global ${DTYPE} * output, const int Ny0, const int stride){

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const int Nx = get_global_size(0);
  const int Ny = get_global_size(1);


  float res = ${DEFAULT};

  int start = j*stride-${FSIZE_Y}/2;

  const int h_start = max(0,${FSIZE_Y}/2-j*stride);
  const int h_end = min(${FSIZE_Y},Ny0-j*stride+${FSIZE_Y}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+(start+ht)*Nx+k*Nx*Ny0];
	res = ${FUNC};
	}


  output[i+j*Nx+k*Nx*Ny] = (${DTYPE})(res);
}

__kernel void filter_3_z(__global ${DTYPE} * input,
                         __global ${DTYPE} * output, const int Nz0, const int stride){

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const int Nx = get_global_size(0);
  const int Ny = get_global_size(1);

  float res = ${DEFAULT};

  int start = k*stride-${FSIZE_Z}/2;

  const int h_start = max(0,${FSIZE_Z}/2-k*stride);
  const int h_end = min(${FSIZE_Z},Nz0-k*stride+${FSIZE_Z}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+j*Nx+(start+ht)*Nx*Ny];
	res = ${FUNC};
	}


  output[i+j*Nx+k*Nx*Ny] = (${DTYPE})(res);
}
