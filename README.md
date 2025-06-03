## QR-factorization
Any (full-rank real) matrix $`A`$ can be written as $`A = QR`$ with $`R`$ upper-triangular and $`Q`$ orthogonal (both real). 
This form  is called QR-factorization of $`A`$. The main use of QR-factorization is in modern methods of numerically solving eigenvalue problems.

For any numerical approximations $`\hat{Q} \approx Q`$, $`\hat{R} \approx R`$ we can measure:
- the **orthogonality error** $`\left\| \hat{Q}^T \hat{Q} - \mathbb{1} \right\|`$
- the **reconstruction error** $`\left\| \hat{Q} \hat{R} - A \right\|`$

(we take $`\left\| \cdot \right\|`$ to be $`L^{\infty}`$-norm).

In this project, we compare different QR-factorization algorithms using those two types of errors (one could also measure how much $`R`$ differs
from being upper-triangular, but we don't consider it here).

## Classical Gram-Schmidt algorithm
Suppose $`A`$ is square matrix, and let its column vectors be $`\left\{ a_i \right\}_{i=1}^{n}`$; similarly, let column vectors of $`Q`$ be 
$`\left\{ q_i \right\}_{i=1}^{n}`$, and let the entries of $`R`$ be $`r_{ij}`$. Then QR-factorization of $`A`$ is equivalent to
```math
a_1 = r_{11}q_1, \quad \dots \quad, a_n = \sum_{i=1}^n r_{i n} q_i
```
which can be rewritten as 
```math
q_1 = a_1/r_{11}, \quad \dots \quad, q_n = (a_n - \sum_{i=1}^{n-1} r_{in}q_i)/r_{nn}
```
This is exactly Gram-Schmidt orthogonalization process applied to $`\left\{ q_i \right\}_{i=1}^{n}`$, if we take 
```math
r_{ij} = q_i \cdot a_j, \quad i \neq j
```
and 
```math
r_{ii} = \left\| a_i - \sum_{k=1}^{i-1} r_{ki} q_k \right\| _2
```
where $`\left\| \cdot \right\| _2`$ is $`L^2`$-norm

We denote this method as `GS1`

## Modified Gram-Schmidt algorithm
As we'll see in the moment, the above algorithm is numerically unstable. The remedy is simple though, and the stable version differs only 
by a single symbol in our code! 

Namely, instead of subtracting the projections of $`(k+1)`$th vector onto the first $`k`$ vectors at each step, we subtract the projection
onto the first vector from all other vectors as soon as its calculated, then subtract projection on the second vector from all remaining ones, etc.
In other words, instead of subtracting projections *after* accumulating all of them, we update the remaining vectors *immediately* after each projection.

We denote this method as `GS2`.
## `GS1` and `GS2` compared
We've generated 10 random matrices for each of the dimensions 2, 4, 8, 16, 32, 64 and plotted mean orthogonality and reconstruction errors for 
`GS1` and `GS2`. The results are given below. Errors are given in machine precision `eps` units.
![](GS1%20vs%20GS2%20Orth.png) ![](GS1%20vs%20GS2%20Reconst.png)

We see that `GS1` indeed gives unstable orthogonality error, although reconstruction error is the same as for `GS2`.

## `GS2` with reorthogonalization
To reduce orthogonality error, we've further modified `GS2` to `GS2_reorth` algorithm, which calculates ratio $`\left\| a_i \right\|_2 / \left\| q_i \right\|_2`$
for each $`i`$, and if this ratio is greater than 10, it orthogonalizes $`q_i`$ again, without changing $`R`$. The idea is that, if the length of $`a_i`$ is 
reduced significantly, the corresponding $`q_i`$ is calculated with bad precision. 

## `GS2` and `GS2_reorth` compared
![](GS2%20vs%20GS2_reorth%20Orth.png) ![](GS2%20vs%20GS2_reorth%20Reconst.png)

as we see, orthogonality error is indeed significantly reduced; however, reconstruction error now blows up to infinity. We're not sure why it happens: `GS2_reorth`
works fine for small matrices, so it's not just some blatant code error.

## Householder algorithm
Householder algorithm takes different approach. It constructs orthogonal matrices $`\left\{ H_i \right\}_{i=1}^n`$ such that $`H_1`$ creates zeros under diagonal
in the first column of $`A`$, $`H_2`$ creates zeros under diagonal in the second column of $`H_1 A`$ (while leaving zeros in the first column), etc. 
Then
```math
H_n \dots H_1 A = R
```
so that, by orthogonality
```math
A = H_1^T \dots H_n^T R
```
and we can take $`Q = \left( H_n \dots H_1 \right)^T`$.

We've included in repository PDF with pages from Golub's *Matrix Computations* where he explains stable version of Householder algorithm which we've used here.

## `GS2` and `Householder` compared
(We've added matrices of dimension 128.)
![](GS2%20vs%20Householder%20Orth.png) ![](GS2%20vs%20Householder%20Reconst.png)

`Householder` gives significantly smaller orthogonality error, although its reconstruction error grows faster than that of `GS2`.

## Comparison with `numpy.linalg.qr`

We conclude by comparing the algorithms with the one implemented in NumPy library. For orthogonality error, we compare with `GS2`, and for reconstruction error
we compare with `Householder`. We add matrices of dimension 256.

![](Householder%20vs%20np.linalg.qr%20Orth.png) ![](GS2%20vs%20np.linalg.qr%20Reconst.png) 

`GS2` has reconstruction error growing slower than that of `numpy.linalg.qr`, and orthogonality error of `Householder` grows faster than that of `numpy.linalg.qr`.

From our understanding, the implementation of `Householder` given in this repo is quite similar to the one used in `numpy.linalg.qr`. And, in general, Householder
method is preffered in practice.
