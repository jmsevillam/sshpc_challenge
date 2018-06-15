#include <iostream>
#include <cstdlib>
#include <vector>
#define N 4
#define n 6
#define blocksize 16
#define PI 3.141592654
#define seed 7

//Definition of Functions

void print(double * M,int cols,int rows);
__global__ void matrixMul(double * a,double * b, double * C, int rows,int cols, int cols2);
double matrixSum_comp(double * M,int rows,int cols);
double normal_rand(void);

int main(int argc, char *argv[])
{
  srand(atoi(argv[1]));// Seed initilized
  
  //Definition of Variables
  
  double *X,*Y,*Z;  // Matrix, Transpose and product

  int size = N * n * sizeof (double); // Number of bytes of an N x n matrix
  int size2 = N * 1 * sizeof (double); // Number of bytes of an N x n matrix
  int size3 = n * 1 * sizeof (double); // Number of bytes of an N x n matrix

  cudaMallocManaged (&X, size);
  cudaMallocManaged (&Y, size2);
  cudaMallocManaged (&Z, size3);

  //Initialization of the arrays
  for( int row = 0; row < N; ++row )
   {
    for( int col = 0; col < n; ++col )
    {
	      X[row*n + col] = normal_rand();   //Gaussiand distributed (Function taken from the sshpc github)
		Z[col]=0.;
	}
	Y[row]=normal_rand();
  }

  //Arrays filled of zeros 
 dim3 threads_per_block (1, 1, 1); 
 dim3 number_of_blocks ((n / threads_per_block.x) + 1, (n / threads_per_block.y) + 1, 1);

print(X,N,n);
print(Y,1,N);
print(Z,1,n);

matrixMul <<< number_of_blocks, threads_per_block >>> (X,Y,Z,1,n,N);
cudaDeviceSynchronize();
std::cout<<"------------"<<std::endl;
print(Z,1,n);

 cudaFree(X);
 cudaFree(Y);
 cudaFree(Z);
 return 0;
}

void print(double * M,int cols,int rows){
  for( int row = 0; row < rows; ++row ){
    for( int col = 0; col < cols; ++col )
      {
	std::cout<<M[col + row*cols]<<'\t';
      }
    std::cout<<"\n";
  }

}
double matrixSum_comp(double * M,int rows,int cols){
double sum=0;
for(int row=0;row<rows;row++){
for(int col=0;col<cols;col++){
sum+=M[row+col*rows];
}
}
return sum;
}

__global__ void matrixMul(double * a,double * b, double * C, int cols,int rows,int cols2)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;  
  
if (row < rows && col < cols){
    for (int k = 0; k < cols2; k++){
	C[row*cols+col]+=b[k*cols+col]*a[row*cols2+k];
 }
}

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
