// https://github.com/rudrasohan/algorithms/blob/master/quickselect.c

inline void swap(${DTYPE}* a, ${DTYPE}* b)
{
    ${DTYPE} t = *a;
    *a = *b;
    *b = t;
}

inline int partition(${DTYPE}* arr, const int l, const int r) 
{ 
    ${DTYPE} x = arr[r];
    int i = l; 
    for (int j = l; j <= r - 1; j++) { 
        if (arr[j] <= x) { 
            swap(&arr[i], &arr[j]); 
            i++; 
        } 
    } 
    swap(&arr[i], &arr[r]); 
    return i; 
} 

inline ${DTYPE} kthSmallest(${DTYPE} arr[], int l, int r, int k) 
{ 
    // If k is smaller than number of  
    // elements in array 
    if (k > 0 && k <= r - l + 1) { 
  
        // Partition the array around last  
        // element and get position of pivot  
        // element in sorted array 
        int index = partition(arr, l, r); 
  
        // If position is same as k 
        if (index - l == k - 1) 
            return arr[index]; 
  
        // If position is more, recur  
        // for left subarray 
        if (index - l > k - 1)  
            return kthSmallest(arr, l, index - 1, k); 
  
        // Else recur for right subarray 
        return kthSmallest(arr, index + 1, r,  
                            k - index + l - 1); 
    } 
    else
        return (${DTYPE})0;
  
} 
  


__kernel void rank_2(__global ${DTYPE} * input,
                    __global ${DTYPE} * output,
                    const int Nx0, const int Ny0, 
                    const int rank){

  int x = get_global_id(0);
  int y = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  


  ${DTYPE} a[${FSIZE_Y}*${FSIZE_X}];

  for (int m = 0; m < ${FSIZE_Y}; ++m) {
    for (int n = 0; n < ${FSIZE_X}; ++n) {
		
	  int x2 = x*${FSIZE_X}+n;
	  int y2 = y*${FSIZE_Y}+m;
		
	  bool inside = ((x2>=0)&&(x2<Nx0)&&(y2>=0)&&(y2<Ny0));

	  a[n+${FSIZE_X}*m] = inside?input[x2+y2*Nx0]:(${DTYPE})(${CVAL});
	  }
	}


  output[x+y*Nx] = kthSmallest(a, 0, ${FSIZE_X}*${FSIZE_Y}-1, rank+1);

}



__kernel void rank_3(__global ${DTYPE} * input,
						__global ${DTYPE} * output,
                        const int Nx0, const int Ny0, const int Nz0,
                        const int rank){

  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  ${DTYPE} a[${FSIZE_Z}*${FSIZE_Y}*${FSIZE_X}];

  for (int p = 0; p < ${FSIZE_Z}; ++p) {
	for (int m = 0; m < ${FSIZE_Y}; ++m) {
	  for (int n = 0; n < ${FSIZE_X}; ++n) {
		
	  int x2 = x*${FSIZE_X}+n;
	  int y2 = y*${FSIZE_Y}+m;
	  int z2 = z*${FSIZE_Z}+p;
		
	  bool inside = ((x2>=0)&&(x2<Nx0)&&(y2>=0)&&(y2<Ny0)&&(z2>=0)&&(z2<Nz0));

	  a[n+${FSIZE_X}*m+${FSIZE_X}*${FSIZE_Y}*p] = inside?input[x2+y2*Nx0+z2*Nx0*Ny0]:(${DTYPE})(${CVAL});
	  }
	}
  }


  output[x+y*Nx+z*Nx*Ny] = kthSmallest(a, 0, ${FSIZE_X}*${FSIZE_Y}*${FSIZE_Z}-1, rank+1);

}
