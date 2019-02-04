
__kernel void histogram_partial(__global DTYPE*  src, __global uint *part_hist, const float x0, const float x1){

  uint N = get_global_size(0);
  uint Nloc = get_local_size(0);  

  uint i = get_global_id(0);
  uint igroup = get_group_id(0);
  uint iloc = get_local_id(0);

  __local uint  tmp_histogram[N_BINS];

  // clear the local buffer that will generate the partial
  // histogram

  uint j = iloc;
  while (j<N_BINS){
	tmp_histogram[j] = 0;
	j+= Nloc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  float val = (float)(src[i]);

  uint idx = (uint)(clamp(floor((val-x0)/(x1-x0)*N_BINS),0.f,1.f*N_BINS-1.f));

  atomic_inc(&tmp_histogram[idx]);

  barrier(CLK_LOCAL_MEM_FENCE);

  
  j = iloc;
  while (j<N_BINS){
	part_hist[igroup+j*RED_SIZE] = tmp_histogram[j];
	j+= Nloc;
  }

}


__kernel void histogram_sum( __global uint *part_hist, __global uint *hist){

  uint N = get_global_size(0);
  uint i = get_global_id(0);

  uint res = 0;
  
  for(uint j =0;j<RED_SIZE;j++){
	res += part_hist[i*RED_SIZE+j];
  };
  
  hist[i] = res;
  
}

