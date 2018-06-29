//Library Definition
#include <iostream>  //cout
#include <fstream>   //Files
#include <cstdlib>   //atoi function
//Constant Definition
#define PI 3.141592654
#define blocksize 32

#define n 512
#define p 128

//Print matrix into standard output
void print(double * M,int cols,int rows);
void dot(double * a,double * b, double & c, int cols);
/*
DEVICE FUNCTIONS
*/	
//Matrix transposition (Rows and Cols of M)
__global__ void matrixTrans(double * M,double * MT, int rows, int cols);
//Matrix multiplication(Cols and Rows of the result)
__global__ void matrixMul(double * a,double * b, double * C, int cols,int rows,int cols2);
//INVERSION OF MATRICES ----GAUSS JORDAN METHOD --------
void Inverse(double * A, double * I,int nn);
__global__ void nodiag_normalize(double *A, double *I, int nn, int i);
__global__ void diag_normalize(double *A, double *I, int nn, int i);
__global__ void gaussjordan(double *A, double *I, int nn, int i);
__global__ void set_zero(double *A, double *I, int nn, int i);
//Sum of Matrices
__global__ void matrixSum(const double * M1,const double * M2,double * Msum,double alpha,double beta, int rows, int cols);


//Initialization of matrices, ones, zeros, identity
void set_ones(double * M, int l);
void set_zeros(double * M, int l);
void set_iden(double * M, int l);
//Print matrices into external files
void print_file(char const * NameArch, const double * M,int cols,int rows);
//Random numbers
double normal_rand(void);
/*
MAIN FUNCTION
*/
int main(int argc, char * argv[]){

srand(atoi(argv[1])); //Seed recieved from terminal

//int cols=p;
//int raws=n;
double *X, *Xt, *XXt, *Inv;
double *H0,*H, *J, *Suma;
double *Y,*Yt, *aux, *Id;

int size0 = n * sizeof(double);
int size2 = p * p * sizeof(double);
int size3 = n * n * sizeof(double);
int size4 = n * p * sizeof(double);

cudaMallocManaged(&X,size4);
cudaMallocManaged(&Xt,size4);
cudaMallocManaged(&H0,size4);
cudaMallocManaged(&H,size3);
cudaMallocManaged(&J,size3);
cudaMallocManaged(&Suma,size3);
cudaMallocManaged(&XXt,size2);
cudaMallocManaged(&Yt,size0);
cudaMallocManaged(&Inv,size2);
cudaMallocManaged(&Y,size0);
cudaMallocManaged(&aux,size0);
cudaMallocManaged(&Id,size3);

double ssr=0,sst=0;
double R2=0;
double F=0, Ftest=1.1962078803512777;
for(int row=0;row<n;row++){
        for(int col=0;col<p;col++){
                X[row*p+col]=0.1*normal_rand();//distribution(generator);
                Y[col]=0.1*normal_rand();
        }
}
print_file("x.dat",X,p,n);
print_file("y.dat",Y,1,n);
dim3 threadsPerBlock(blocksize, blocksize);
dim3 numBlocks((n + blocksize - 1) / blocksize, (p + blocksize - 1) / blocksize);
dim3 numBlocks1((p + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
dim3 numBlocks2((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
dim3 numBlocks3((1 + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);

matrixTrans<<<numBlocks,threadsPerBlock>>>(X,Xt,n,p);
cudaDeviceSynchronize();

matrixMul<<<numBlocks1,threadsPerBlock>>>(Xt,X,XXt,p,p,n);
cudaDeviceSynchronize();
set_iden(Inv,p);

Inverse(XXt,Inv,p);cudaDeviceSynchronize();
//std::cout<<"inv"<<std::endl;
print_file("Inv.dat",Inv,p,p);

//matrixMul<<<numBlocks,threadsPerBlock>>>(X,Xt,XXt,p,p,n);
//cudaDeviceSynchronize();

matrixMul<<<numBlocks1,threadsPerBlock>>>(Inv,Xt,H0,n,p,p);
cudaDeviceSynchronize();
print_file("H0.dat",H0,n,p);

matrixMul<<<numBlocks2,threadsPerBlock>>>(X,H0,H,n,n,p);
cudaDeviceSynchronize();
print_file("H.dat",H,n,n);
set_ones(J,n);
matrixSum<<<numBlocks2,threadsPerBlock>>>(H,J,Suma,1.,-1./n, n,n);
cudaDeviceSynchronize();
print_file("Suma.dat",Suma,n,n);

matrixMul<<<1,n>>>(Suma,Y,aux,1,n,p);
cudaDeviceSynchronize();
print_file("Aux.dat",aux,n,n);

//matrixMul<<<numBlocks,threadsPerBlock>>>(Y,aux,J,1,1,n);
//cudaDeviceSynchronize();
//ssr=J[0];
dot(Y,aux,ssr,n);
set_iden(Id,n);
set_zeros(Suma,n*n);
matrixSum<<<numBlocks,threadsPerBlock>>>(Id,J,Suma,1.,-1./n, n, n);
cudaDeviceSynchronize();
set_zeros(aux,n);

matrixMul<<<numBlocks,threadsPerBlock>>>(Suma,Y,aux,n,1,n);
cudaDeviceSynchronize();

//matrixMul<<<numBlocks,threadsPerBlock>>>(Y,aux,J,1,1,n);
//cudaDeviceSynchronize();
//sst=J[0];
dot(Y,aux,sst,n);

R2=ssr/sst;
F=(R2*(n-p-1.))/((1.-R2)*p);

std::cout<<R2<<' '<<ssr<<' '<<sst<<' '<<F<<std::endl;


cudaFree(X);
cudaFree(Xt);
cudaFree(XXt);
cudaFree(Inv);
cudaFree(H0);
cudaFree(H);
cudaFree(J);
cudaFree(Suma);
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
__global__ void matrixTrans(double * M,double * MT, int rows, int cols)
{
  double val=0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < rows && col < cols){   
    val = M[col + row*cols];
    MT[row + col*rows] = val;      
  } 
}
__global__ void matrixMul(double * a,double * b, double * C, int cols,int rows,int cols2)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;  
  if (row < rows && col < cols){
C[row*cols+col]  =0;
    for (int k = 0; k < cols2; k++){
	C[row*cols+col]+=b[k*cols+col]*a[row*cols2+k];
   }
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
  dim3 threadsPerBlock2(blocksize, blocksize);
  dim3 numBlocks2((nn + blocksize - 1) / blocksize, (nn + blocksize - 1) / blocksize);
  for (int i = 0; i<nn; i++){
    nodiag_normalize << <numBlocks2, threadsPerBlock2 >> >(A, I, nn, i);
    diag_normalize << <numBlocks2, threadsPerBlock2 >> >(A, I, nn, i);
    gaussjordan << <numBlocks2, threadsPerBlock2 >> >(A, I, nn, i);
    set_zero << <numBlocks2, threadsPerBlock2 >> >(A, I, nn, i);
  }
cudaDeviceSynchronize();
}

__global__ void matrixSum(const double * M1,const double * M2,double * Msum,double alpha,double beta, int rows, int cols)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;  
  if (row < rows && col < cols){   
    Msum[row + col*rows] = alpha*M1[row+col*rows]+beta*M2[row+col*rows];      
  } 
}

void print_file(char const * NameArch, const double * M,int cols,int rows){
  std::ofstream File(NameArch);
File.precision(16);
  for( int row = 0; row < rows; ++row ){
    for( int col = 0; col < cols; ++col )
      {
       	File<<M[col + row*cols]<<'\t';
      }
    File<<"\n";
  }
File.close();
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

void set_iden(double * M, int l){
for(int row=0;row<l;row++){
	for(int col=0;col<l;col++){
	M[row*l+col]=0;
	if (col==row){
		M[row*l+col]=1;
	}
	}
}
}

void set_ones(double * M, int l){
for(int row=0;row<l;row++){
	for(int col=0;col<l;col++){
	M[row*l+col]=1;
	}
}
}
void set_zeros(double * M, int l){
for(int row=0;row<l;row++){
	M[row]=0;
}
}


void dot(double * a,double * b, double & c, int cols){
c=0;
for(int i=0;i<cols;i++){
c+=a[i]*b[i];
}
}

