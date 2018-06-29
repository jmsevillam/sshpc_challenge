#include<stdio.h>
#define blocksize 1

__global__ void nodiag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}
	
}

__global__ void diag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(float *A, float *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

__global__ void set_zero(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

int main()
{
	const int n = 3;

	float *d_A, *dI;

	int size = n*n*sizeof(float);

	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);

cudaMallocManaged(&d_A,size);
cudaMallocManaged(&dI,size);

	for (int i = 0; i<n; i++){
		for (int j = 0; j<n; j++){
			d_A[i*n+j]=5*i/n-j*2;
			if (i == j) dI[i*n + i] = 1.0;
			else dI[i*n + j] = 0.0;
		}
	}
d_A[0]=16;
d_A[1]=4;
d_A[2]=1;
d_A[3]=12;
d_A[4]=0;
d_A[5]=11;
d_A[6]=11;
d_A[7]=0;
d_A[8]=0;

for (int i = 0; i<n; i++){
	for (int j = 0; j<n; j++){
		printf("%f ",d_A[i*n+j]);
	}
	printf("\n");
}


	for (int i = 0; i<n; i++){
		nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		diag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		gaussjordan << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		set_zero << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
}

cudaDeviceSynchronize();
for (int i2 = 0; i2<n; i2++){
	for (int j = 0; j<n; j++){
		printf("%f",dI[i2*n+j]);
	}
	printf("\n");


	
}

	cudaFree(d_A);
	cudaFree(dI);


	return 0;
}

