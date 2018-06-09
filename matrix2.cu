#include <stdio.h>
#define N 1024
#define n 256
#define n2 256.0
#define blocksize 16
#define PI 3.141592654
#define seed 7

//Definition of Functions


void print(double * M,int cols,int rows);
__global__ void matrix_Vec_Mul( double * A, double * b, double * c );
double dot( double * a, double * b);
__global__ void matrixSum( double * a, double * b,double const_a,double const_b, double * c );
__global__ void matrixMul( double * a, double * b, double * c );
__global__ void matrixMul2( double * a, double * b, double * c );
__global__ void matrixTrans(double * M,double * MT);
__global__ void matrixTrans2(double * M,double * MT);
__global__ void nodiag_normalize(double *A, double *I, int nn, int i);
__global__ void diag_normalize(double *A, double *I, int nn, int i);
__global__ void gaussjordan(double *A, double *I, int nn, int i);
__global__ void set_zero(double *A, double *I, int nn, int i);
void Inverse(double * A, double * I,int nn);
double normal_rand(void);

int main()
{
  srand(seed);// Seed initilized
  
  //Definition of Variables
  
  double *X, *Xt, *XXt;  // Matrix, Transpose and product
  double *y, *H, *beta, *beta0, *I; //Data y, HAT matrix, Optimal coefficents, auxilliar and inverse of XXt
  double *ones;   // Matrix filled of ones
  double *ssr_matrix, *ssr_matrixx, *ssr_matrixt, *Id;
//  double ssr=0,sst=0;
  int size = N * n * sizeof (double); // Number of bytes of an N x n matrix
  int size2 =n * n * sizeof (double); // Number of bytes of an n x n matrix
  
  // Allocate memory
  
  cudaMallocManaged (&X, size);
  cudaMallocManaged (&Xt, size);
  cudaMallocManaged (&beta, n*sizeof(double));
  cudaMallocManaged (&ssr_matrixx, n*sizeof(double));
  cudaMallocManaged (&y,n*sizeof(double));
  cudaMallocManaged (&H, size2);
  cudaMallocManaged (&beta0, size2);
  cudaMallocManaged (&I, size2);
  cudaMallocManaged (&Id, size2);
  cudaMallocManaged (&XXt, size2);
  cudaMallocManaged (&ones, size2);
  cudaMallocManaged (&ssr_matrix, size2);
  cudaMallocManaged (&ssr_matrixt, size2);

  //Initialization of the arrays
  for(int i=0;i<n;i++){
	y[i]=normal_rand();
  }

  for( int row = 0; row < N; ++row )
   {
    for( int col = 0; col < n; ++col )
    {
      X[row*n + col] =normal_rand();   //Gaussiand distributed (Function taken from the sshpc github)
      Xt[row*n + col] = 1.;
	}
  }
  //Arrays filled of zeros
  
cudaMemset(XXt,0,n*n*sizeof(double)); 
cudaMemset(I,0,n*n*sizeof(double));
cudaMemset(Id,0,n*n*sizeof(double));
cudaMemset(H,0,n*n*sizeof(double));
cudaMemset(ssr_matrix,0,n*n*sizeof(double));
cudaMemset(ssr_matrixt,0,n*n*sizeof(double));
cudaMemset(ssr_matrixx,0,n*sizeof(double));
 
 for (int i = 0; i<n; i++){
   I[i*n + i] = 1.0;          //Identity matrix
	Id[i*n+i]=1.0;
   for (int j = 0; j<n; j++){
     ones[i*n+j]=1.0;         //Ones
   }
 }
 
 dim3 threads_per_block (16, 16, 1); 
 dim3 number_of_blocks2 ((n / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);
 // --------------------------------------------------------------------
 matrixTrans <<< number_of_blocks2, threads_per_block >>> (X,Xt);
 cudaDeviceSynchronize();
 
 matrixMul <<< number_of_blocks2, threads_per_block >>> ( X, Xt, XXt );
 cudaDeviceSynchronize(); 


 Inverse(XXt,I,n);
 
 dim3 number_of_blocks ((n / threads_per_block.x) + 1, (n / threads_per_block.y) + 1, 1);

 matrixMul2<<<number_of_blocks,threads_per_block>>>(I,Xt,beta0);
 cudaDeviceSynchronize(); 

 matrixMul2<<<number_of_blocks,threads_per_block>>>(X,beta0,H);
 cudaDeviceSynchronize(); 

 matrix_Vec_Mul<<<1,n>>>(beta0,y,beta);
 cudaDeviceSynchronize(); 

 matrixSum<<<number_of_blocks,threads_per_block>>>(H,ones,1,-1.0/n2,ssr_matrix);
 cudaDeviceSynchronize(); 

 matrixTrans2 <<< number_of_blocks, threads_per_block >>> (ssr_matrix,ssr_matrixt);
 cudaDeviceSynchronize();

 matrix_Vec_Mul<<<1,n>>>(ssr_matrixt,y,ssr_matrixx); 
cudaDeviceSynchronize(); 

//ssr=dot(ssr_matrixx,y);

//printf("%f\n",ssr);

matrixSum<<<number_of_blocks,threads_per_block>>>(Id,ones,1,-1.0/n2,ssr_matrix);
 cudaDeviceSynchronize(); 

 matrixTrans2 <<< number_of_blocks, threads_per_block >>> (ssr_matrix,ssr_matrixt);
 cudaDeviceSynchronize();

 matrix_Vec_Mul<<<1, n>>>(ssr_matrixt,y,ssr_matrixx);
 cudaDeviceSynchronize(); 

//sst=dot(ssr_matrixx,y);
//double RR;
//printf("%f\n",sst);

// RR=ssr/sst;
//printf("R*R=%f\n",RR);
/**/
 //Release memory
 cudaFree(I);
 cudaFree(Id);
 cudaFree(X);
 cudaFree(Xt);
 cudaFree(beta0);
 cudaFree(XXt); 
 cudaFree(beta);
 cudaFree(y);
 cudaFree(H);
 cudaFree(ones); 
 cudaFree(ssr_matrix); 
 cudaFree(ssr_matrixx); 
 return 0;
}


void print(double * M,int cols,int rows){
  
  for( int row = 0; row < rows; ++row ){
    for( int col = 0; col < cols; ++col )
      {
	printf("%f,",M[col + row*cols]);
      }
    printf("\n");
  }
}
__global__ void matrixMul( double * a, double * b, double * c )
{
  double val = 0.;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < n && col < n){
    for (int k = 0; k < N; k++){
      val += a[col * N + k] * b[ k*n  + row];
    }
  }    
  c[row*n + col] = val;
}
__global__ void matrixSum( double * a, double * b,double const_a,double const_b, double * c )
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < n && col < n){
      c[col*n+row]= const_a*a[col * n + row] +const_b* b[ col*n  + row];
    }
}
__global__ void matrix_Vec_Mul( double * A, double * b, double * c )
{
  double val = 0.;
  int row = threadIdx.x; 
  if (row  < n ){
    for (int k = 0; k < n; k++){
      val += A[row*n + k] * b[k];
//	printf("%f %f\n",A[row*n+k],b[k]);
}
 }    
  c[row] = val;
}


double dot( double * a, double * b)
{double c0=0;
    for (int k = 0; k < n; k++){
      c0 += a[k] * b[k];
}
return c0;
}


__global__ void matrixMul2( double * a, double * b, double * c )
{
  double val = 0.;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < n && col < n){
    for (int k = 0; k < n; k++){
      val += b[col * n + k] * a[ k*n  + row];
    }
  }
  c[row*n + col] = val;
}

__global__ void matrixTrans(double * M,double * MT)
{
  double val=0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < n && col < N){   
    val = M[col + row*N];
    MT[row + n*col] = val;      
  } 
}
__global__ void matrixTrans2(double * M,double * MT)
{
  double val=0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < n && col < n){   
    val = M[col + row*n];
    MT[row + n*col] = val;      
  } 
}


__global__ void nodiag_normalize(double *A, double *I, int nn, int i){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x< nn && y < nn){
    if (x < nn && y < nn){
      if (x == i && x!=y){
	I[x*nn + y] /= A[i*nn + i];
	A[x*nn + y] /= A[i*nn + i];
      }
    }
  }	
}

__global__ void diag_normalize(double *A, double *I, int nn, int i){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  
  if (x < nn && y < nn){
    if (x == y && x == i){
      I[x*nn + y] /= A[i*nn + i];
      A[x*nn + y] /= A[i*nn + i];
    }
  }
}

__global__ void gaussjordan(double *A, double *I, int nn, int i)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x< nn && y < nn){
    
    if (x < nn && y < nn){
      if (x != i){
	I[x*nn + y] -= I[i*nn + y] * A[x*nn + i];
	if (y != i){
	  A[x*nn + y] -= A[i*nn + y] * A[x*nn + i];
	}	 
      }
    } 
  }
}

__global__ void set_zero(double *A, double *I, int nn, int i){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nn && y < nn){
    if (x != i){
      if (y == i){
	A[x*nn + y] = 0;
      }
    }
  }
}

void Inverse(double * A, double * I,int nn){
  dim3 threadsPerBlock(blocksize, blocksize);
  dim3 numBlocks((nn + blocksize - 1) / blocksize, (nn + blocksize - 1) / blocksize);
  for (int i = 0; i<nn; i++){
    nodiag_normalize << <numBlocks, threadsPerBlock >> >(A, I, nn, i);
    diag_normalize << <numBlocks, threadsPerBlock >> >(A, I, nn, i);
    gaussjordan << <numBlocks, threadsPerBlock >> >(A, I, nn, i);
    set_zero << <numBlocks, threadsPerBlock >> >(A, I, nn, i);
  }
  cudaDeviceSynchronize();
}

// Random number generator as per Abramowitz & Stegun
// Source taken from:
// http://c-faq.com/lib/gaussian.html

double normal_rand(void){
  static double U, V;
  static int phase = 0;
  double Z;
  if(phase == 0) {
    U = (rand() + 1.) / (RAND_MAX + 2.);
    V = rand() / (RAND_MAX + 1.);
    Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
  } else
    Z = sqrt(-2 * log(U)) * cos(2 * PI * V);
  phase = 1 - phase;  
  return Z;
}
