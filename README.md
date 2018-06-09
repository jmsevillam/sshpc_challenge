# sshpc_challenge
Solution challenge sshpc 2018

The solution is proposed by using CUDA based software.

The program is partially working, the following functions are implemented using CUDA:

    - Matrices Multiplication 
    - Matrices Sum
    - Matrices Transposition
    - Matrices Inversion
To run the program, no specially libreries are needed.
```bash
cuda matrix2.cu -o matrix.out
./matrix.out
```

It computes a randomly distributed matrix of size 1024 x 256 transpose, then the product, inversion and some multiplications. That task is done on average on 0.5 secs on a Nvidia Tesla M60. 

Inline-style: 

![alt text](https://github.com/jmsevillam/sshpc_challenge/1.png "Example")
