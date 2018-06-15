# sshpc_challenge
Solution challenge sshpc 2018

The solution is proposed by using CUDA based software.

The code Freedman.cu, perform 8192 times the following procedure:

    - Create a matrix $X$ of 128x512 filled of gaussian random numbers.
    - Create a vector $Y$ of 128
    - Find the $\beta$ that minimizes the problem.
    - Compute the variances, SSR, SST and SSE.
    - The NULL hypothesys from F0 is tested using the Fischer distribution.
    - The p-value is calculated
    - After completing the ANOVA test, the values of $r^2$ and $F$ are printed.
    - The relevant subset is chosen and saved in another matrix.
    - The ANOVA test is done again.

To run the program, no specially libreries are needed.
```bash
cuda Freedman.cu -o Freedman.out
./Freedman.out > data.dat
python plots.py
```
The computation is done for the 8192 repetitions on average on 6 minutes on a Nvidia Tesla M60. 


Besides that,´Intento.py´ is a python code that also solves the problem.
