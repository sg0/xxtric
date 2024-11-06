template<const int blocksize>
__global__
  void local_count_gpu
(
 GraphElem start,
 GraphElem end,
 GraphElem jstart,
 GraphElem jend,
 GraphElem* __restrict__ edge_indices,
 GraphElem* __restrict__ edge_list,
 GraphElem* local_count 
 )
{
  volatile __shared__ short data[blocksize];
  __shared__ GraphElem count = 0;
  GraphElem step = gridDim.x;

  for(GraphElem i = start+blockIdx.x; i < end; i += step)
  {
    for(GraphElem k = jstart + threadIdx.x; k < jend; k+=blocksize)
    {
      if (edge_list[k] == edge_list[i])
        count++;
    }
  }

  data[threadIdx.x] = count;
  __syncthreads();

  for (unsigned int s=blocksize/2; s>=32; s>>=1)
  {
    if (threadIdx.x < s)
      data[threadIdx.x] = data[threadIdx.x+s];
    __syncthreads();
  }
  count = data[threadIdx.x%32];
  for (int offset = 16; offset > 0; offset /= 2)
    count += __shfl_down_sync(0xffffffff, count, offset);

  if(threadIdx.x == 0)
    *local_count = count; 
  __syncthreads();
}


