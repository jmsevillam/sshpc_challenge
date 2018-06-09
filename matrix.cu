#include <stdio.h>
#define N 4
#define n 3


__global__ void matrixMul( float * a, float * b, float * c )
{
  float val = 0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
if (row < n && col < n){
   for ( int k = 0; k < N; ++k )
    val += a[row * N + k] * b[ k*N  + col];
}    
c[row * N + col] = val;
}


__global__ void matrixTrans(float * M,float * MT)
{
int val=0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

      MT[row + col*N] = 0;
if (row < N && col < N)
  {
  val = M[col + row*N];
  MT[row + col*N] = val;
      
}
}

void print(float * M,int cols,int rows){

  for( int row = 0; row < rows; ++row ){
    for( int col = 0; col < cols; ++col )
    {
     printf("%f,",M[row*N + col]);
    }
     printf("\n");
   }
}

int main()
{
  float *a, *b, *c_gpu;
  int size = N * n * sizeof (float); // Number of bytes of an N x N matrix
  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_gpu, n*n* sizeof(float));
  // Initialize memory

  for( int row = 0; row < N; ++row )
   {
    for( int col = 0; col < n; ++col )
    {
      a[row*N + col] = row+2.*col;
      b[row*N + col] = 1.;
      c_gpu[row*N + col] = 0.;
    }
   }
	print(a,n,N);  

  dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
  dim3 number_of_blocks ((N / threads_per_block.x) + 1, (n / threads_per_block.y) + 1, 1);

   matrixTrans <<< number_of_blocks, threads_per_block >>> ( a,b );
   cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

   dim3 threads_per_block2 (16, 16, 1); // A 16 x 16 block threads
   dim3 number_of_blocks2 ((n / threads_per_block.x) + 1, (n / threads_per_block.y) + 1, 1);

   matrixMul <<< number_of_blocks2, threads_per_block2 >>> ( b, a, c_gpu );
   cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

print(b,N,n);  
print(a,n,N);  


print(c_gpu,n,n);  
  
  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_gpu );
}
