//Library Definition
#include<omp.h>
#include <iostream>  //cout
#include <fstream>   //Files
#include <cstdlib>   //atoi function
#include <cmath>

//Constant Definition
#define PI 3.141592654
#define blocksize 32
#define Repetitions 8192


//Print matrix into standard output
void print(double * M,int cols,int rows);
void dot(double * a,double * b, double & c, int cols);
void Create_New_Matrix(double * M,double * New,int * vec, int p0, int pp,int nn);

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
void restore_2zero(double * M,int size);

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
int main(int argc, char * argv[])
{   
  int n=512;
  int p=128;
  
  double ssr=0,sst=0, sse=0;
  double R2=0,sigma2=0,p0=0;
  double F=0, Ftest=1.1962078803512777;
  
  int size0 = n * sizeof(double);
  int size1 = p * sizeof(double);
  int size10 = p * sizeof(int);
  int size2 = p * p * sizeof(double);
  int size3 = n * n * sizeof(double);
  int size4 = n * p * sizeof(double);
  

  double *X, *Xt, *XXt, *Inv,*Xnew;
  double *H0,*H, *J, *Suma;
  double *Y, *aux, *Id, *beta;
  int *vec;
  srand(7); //Seed recieved from terminal
  
  for(int rep=0;rep<Repetitions;rep++)
    {
      // Start with the number of columns
      p=128;
      R2=0;
      F=0;
      
      // Define the size of the arrays
      size0 = n * sizeof(double);
      size1 = p * sizeof(double);
      size10 = p * sizeof(int);
      size2 = p * p * sizeof(double);
      size3 = n * n * sizeof(double);
      size4 = n * p * sizeof(double);
      
      // ask for a memory for each array
      cudaMallocManaged(&X,size4);
      cudaMallocManaged(&Xt,size4);
      cudaMallocManaged(&H0,size4);
      cudaMallocManaged(&H,size3);
      cudaMallocManaged(&J,size3);
      cudaMallocManaged(&Suma,size3);
      cudaMallocManaged(&XXt,size2);
      cudaMallocManaged(&Inv,size2);
      cudaMallocManaged(&Y,size0);
      cudaMallocManaged(&aux,size0);
      cudaMallocManaged(&Id,size3);
      cudaMallocManaged(&beta,size1);
      cudaMallocManaged(&vec,size10);

      // ------------ Start the X's and the Y's -------------------
      for(int row=0;row<n;row++)
	{
	  for(int col=0;col<p;col++)
	    {
	      X[row*p+col]=0.1*normal_rand();//distribution(generator);
	      Y[col]=0.1*normal_rand();
	    }
	}

      // --------------------- Start the Army to compute --------------
      dim3 threadsPerBlock(blocksize, blocksize);
      dim3 numBlocks((n + blocksize - 1) / blocksize, (p + blocksize - 1) / blocksize);
      dim3 numBlocks1((p + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
      dim3 numBlocks2((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
      dim3 numBlocks3((1 + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);

      // ------------------- we compute  the products to calculate H hat ----------------
      matrixTrans<<<numBlocks,threadsPerBlock>>>(X,Xt,n,p);
      cudaDeviceSynchronize();
      
      matrixMul<<<numBlocks1,threadsPerBlock>>>(Xt,X,XXt,p,p,n);
      cudaDeviceSynchronize();
      set_iden(Inv,p);
      
      Inverse(XXt,Inv,p);
      cudaDeviceSynchronize();
      
      matrixMul<<<numBlocks1,threadsPerBlock>>>(Inv,Xt,H0,n,p,p);
      cudaDeviceSynchronize();
      
      matrixMul<<<numBlocks2,threadsPerBlock>>>(X,H0,H,n,n,p);
      cudaDeviceSynchronize();
      
      // --------------------------- we compute beta -------------------------
      
      matrixMul<<<numBlocks2,threadsPerBlock>>>(H0,Y,beta,p,1,n);
      cudaDeviceSynchronize();
      
      set_ones(J,n);
      matrixSum<<<numBlocks2,threadsPerBlock>>>(H,J,Suma,1.,-1./n, n,n);
      cudaDeviceSynchronize();
      
      matrixMul<<<1,n>>>(Suma,Y,aux,1,n,n);
      cudaDeviceSynchronize();

      // ----------------------------- Computing SSR ------------
      
      dot(Y,aux,ssr,n);
      set_iden(Id,n);
      set_zeros(Suma,n*n);

      // -----------------------------  Computing SST ----------------
      matrixSum<<<numBlocks2,threadsPerBlock>>>(Id,J,Suma,1.,-1./n, n, n);
      cudaDeviceSynchronize();
      
      set_zeros(aux,n);
      matrixMul<<<1,n>>>(Suma,Y,aux,1,n,n);
      cudaDeviceSynchronize();
      
      dot(Y,aux,sst,n);
      set_zeros(aux,n);
      
      // --------- Finally we can compute R2 and F in terms of R2 -------------
      R2=ssr/sst;
      F=(R2*(n-p-1.))/((1.-R2)*p);
      sse=sst-ssr;
      
      // ----------- Computing the variance -----------------
      sigma2=sse/(n-1.);

      // -------------- F test -----------------
      if (F>Ftest)
	{
	  double t0=0,Pvalue=0;
	  
	  // ------------------ TEsting P-Value to know the important variables-----------
	  p0=0;
	  for(int ii=0;ii<p;ii++)
	    {  
	      t0=beta[ii]/std::sqrt(sigma2*Inv[ii*p+ii]);
	      Pvalue=2.*(1.-erf(t0));
	      if(Pvalue<0.25)
		{
		  vec[ii]=1;
		  p0+=1;
		}
	      else
		{
		  vec[ii]=0;
		}
	  }
	  
	  
	  
	  
	  if (p0==0)
	    {
	      
	      continue;
	    }
	  
	      
	  // ------------- if at least one variable pass the p-test we print the values of R2, F
	  std::cout<<'1'<<' '<<R2<<' '<<ssr<<' '<<sst<<' '<<F<<std::endl;
	  
	  // --------------- Declare the new matrix Xnew for the new variables ------------
	  
	  size4 = n*p0*sizeof(double);
	  
	  cudaMallocManaged(&Xnew,size4);
	  
	  Create_New_Matrix(X,Xnew,vec,p0,p,n);
	  
	  // -------- Start over for the new set of variables Xnew-----------
	  p=p0;
	  ssr=0;sst=0; sse=0;
	  R2=0,sigma2=0,F=0;
	  
	  // ----------------- release the old memory to ask for other one------- 
	  cudaFree(Xt);
	  cudaFree(H0);
	  cudaFree(H);
	  cudaFree(J);
	  cudaFree(Suma);
	  cudaFree(XXt);
	  cudaFree(Inv);
	  cudaFree(aux); 
	  cudaFree(Id);
	  cudaFree(beta);
	  
	  // ------------------ we redefine de sizes for the new arrays -------------
	  
	  size0 = n * sizeof(double);
	  size1 = p * sizeof(double);
	  size10 = p * sizeof(int);
	  size2 = p * p * sizeof(double);
	  size3 = n * n * sizeof(double);
	  size4 = n * p * sizeof(double);
	  
	  // ------------------ Ask for memory for the new arrays----------
	  
	  cudaMallocManaged(&Xt,size4);
	  cudaMallocManaged(&H0,size4);
	  cudaMallocManaged(&H,size3);
	  cudaMallocManaged(&J,size3);
	  cudaMallocManaged(&Suma,size3);
	  cudaMallocManaged(&XXt,size2);
	  cudaMallocManaged(&Inv,size2);
	  cudaMallocManaged(&aux,size0);
	  cudaMallocManaged(&Id,size3);
	  cudaMallocManaged(&beta,size1);
	  
	  // -------------------- Ask for a new army of threads to compute ----------------
	  
	  dim3 threadsPerBlock(blocksize, blocksize);
	  dim3 numBlocks((n + blocksize - 1) / blocksize, (p + blocksize - 1) / blocksize);
	  dim3 numBlocks1((p + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	  dim3 numBlocks2((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	  dim3 numBlocks3((1 + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	  
	  // --------------------- Compute de new H hat matrix ---------------
	  matrixTrans<<<numBlocks,threadsPerBlock>>>(Xnew,Xt,n,p);
	  cudaDeviceSynchronize();
	  
	  matrixMul<<<numBlocks1,threadsPerBlock>>>(Xt,Xnew,XXt,p,p,n);
	  cudaDeviceSynchronize();
	  
	  set_iden(Inv,p);	
	  Inverse(XXt,Inv,p);
	  cudaDeviceSynchronize();
	  
	  set_zeros(H0,n*p);	
	  
	  matrixMul<<<numBlocks1,threadsPerBlock>>>(Inv,Xt,H0,n,p,p);
	  cudaDeviceSynchronize();
	  
	  matrixMul<<<numBlocks2,threadsPerBlock>>>(Xnew,H0,H,n,n,p);
	  cudaDeviceSynchronize();
	  
	  set_ones(J,n);
	  matrixSum<<<numBlocks2,threadsPerBlock>>>(H,J,Suma,1.,-1./n, n,n);
	  cudaDeviceSynchronize();
	  
	  matrixMul<<<1,n>>>(Suma,Y,aux,1,n,n);
	  cudaDeviceSynchronize();
	  
	  dot(Y,aux,ssr,n);
	  set_iden(Id,n);
	  set_zeros(Suma,n*n);
	  
	  matrixSum<<<numBlocks2,threadsPerBlock>>>(Id,J,Suma,1.,-1./n, n, n);
	  cudaDeviceSynchronize();
	  
	  
	  set_zeros(aux,n);
	  matrixMul<<<1,n>>>(Suma,Y,aux,1,n,n);
	  cudaDeviceSynchronize();
	  
	  dot(Y,aux,sst,n);
	  set_zeros(aux,n);
	  
	  R2=ssr/sst;
	  F=(R2*(n-p-1.))/((1.-R2)*p);
	  sse=sst-ssr;
	  sigma2=sse/(n-1.);
	  
	  std::cout<<'2'<<' '<<R2<<' '<<ssr<<' '<<sst<<' '<<F<<std::endl;
	  
	  cudaFree(Xnew);
	  
	}
      
      cudaFree(Y);
      cudaFree(aux); 
      cudaFree(Id);
      cudaFree(beta);
      cudaFree(vec);
      cudaFree(X);
      cudaFree(Xt);
      cudaFree(XXt);
      cudaFree(Inv);
      cudaFree(H0);
      cudaFree(H);
      cudaFree(J);
      cudaFree(Suma);
    }
  return 0;
}

void Create_New_Matrix(double * M,double * Nnew,int * vec0, int p0, int pp,int nn)
{
  int col=0;
  for(int i =0; i<pp;i++)
    {
      if(vec0[i]==1){
	for(int row=0;row<nn;row++)
	  {
	    Nnew[row*p0+col]=M[row*pp+i];
	  }
	col+=1;
      }
      else
	{
	  continue;
	}
      
    }  
}


void print(double * M,int cols,int rows)
{
  for( int row = 0; row < rows; ++row )
    {
      for( int col = 0; col < cols; ++col )
	{
	  std::cout<<M[col + row*cols]<<'\t';
	}
      std::cout<<"\n";
    }
}
__global__ void NOVA(double * Beta,double * Inverse,int * Vec, int p0,double Sigma2)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  double t0,Pvalue;
  t0=Beta[x]/sqrt(Sigma2*Inverse[x*p0+x]);
  Pvalue=2.*(1.-erf(t0));
  if(Pvalue<0.25)
    {
      Vec[x]=1;
    }
  else
    {
      Vec[x]=0;
    }
}
__global__ void matrixTrans(double * M,double * MT, int rows, int cols)
{
  double val=0;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < rows && col < cols)
    {   
      val = M[col + row*cols];
      MT[row + col*rows] = val;      
    } 
}
__global__ void matrixMul(double * a,double * b, double * C, int cols,int rows,int cols2)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;  
  if (row < rows && col < cols)
    {
      C[row*cols+col]  =0;
      for (int k = 0; k < cols2; k++)
	{
	  C[row*cols+col]+=b[k*cols+col]*a[row*cols2+k];
	}
    }
}

__global__ void nodiag_normalize(double *A, double *I, int nn, int i)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x< nn && y < nn)
    {
      if (x < nn && y < nn)
	{
	  if (x == i && x!=y)
	    {
	      I[x*nn + y] /= A[i*nn + i];
	      A[x*nn + y] /= A[i*nn + i];
	    }
	}
    }	
}

__global__ void diag_normalize(double *A, double *I, int nn, int i)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;  
  if (x < nn && y < nn)
    {
      if (x == y && x == i)
	{
	  I[x*nn + y] /= A[i*nn + i];
	  A[x*nn + y] /= A[i*nn + i];
	}
    }
}

__global__ void gaussjordan(double *A, double *I, int nn, int i)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x< nn && y < nn)
    {
      if (x < nn && y < nn)
	{
	  if (x != i)
	    {
	      I[x*nn + y] -= I[i*nn + y] * A[x*nn + i];
	      if (y != i)
		{
		  A[x*nn + y] -= A[i*nn + y] * A[x*nn + i];
		}	 
	    }
	} 
    }
}

__global__ void set_zero(double *A, double *I, int nn, int i)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nn && y < nn)
    {
      if (x != i)
	{
	  if (y == i)
	    {
	      A[x*nn + y] = 0;
	    }
	}
    }
}

void Inverse(double * A, double * I,int nn)
{
  dim3 threadsPerBlock2(blocksize, blocksize);
  dim3 numBlocks2((nn + blocksize - 1) / blocksize, (nn + blocksize - 1) / blocksize);
  for (int i = 0; i<nn; i++)
    {
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
  if (row < rows && col < cols)
    {   
      Msum[row + col*rows] = alpha*M1[row+col*rows]+beta*M2[row+col*rows];      
    } 
}

void print_file(char const * NameArch, const double * M,int cols,int rows)
{
  std::ofstream File(NameArch);
  File.precision(16);
  for( int row = 0; row < rows; ++row )
    {
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

double normal_rand(void)
{
  static double U, V;
  static int phase = 0;
  double Z;
  if(phase == 0)
    {
      U = (rand() + 1.) / (RAND_MAX + 2.);
      V = rand() / (RAND_MAX + 1.);
      Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
    }
  else
    {
      Z = sqrt(-2 * log(U)) * cos(2 * PI * V);
    }
  phase = 1 - phase;  
  return Z;
}

void set_iden(double * M, int l)
{
  for(int row=0;row<l;row++)
    {
      for(int col=0;col<l;col++)
	{
	  M[row*l+col]=0;
	  if (col==row)
	    {
	      M[row*l+col]=1;
	    }
	}
    }
}

void set_ones(double * M, int l)
{
  for(int row=0;row<l;row++)
    {
    for(int col=0;col<l;col++)
      {
	M[row*l+col]=1;
      }
    }
}
void set_zeros(double * M, int l)
{
  for(int row=0;row<l;row++)
    {
      M[row]=0;
    }
}


void dot(double * a,double * b, double & c, int cols)
{
  c=0;
  for(int i=0;i<cols;i++)
    {
      c+=a[i]*b[i];
    }
}

void restore_2zero(double * M,int size)
{
  for(int i=0;i<size;i++)
    {
      M[i]=0.; 
    }  
}
