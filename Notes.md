# Practical Linear Algebra for Data Science
## Mike X Cohen


### Vector Dot Product 
* Element wise multiplication and sum

```
v = np.array([1,2,3,4])
w = np.array([5,6,7,8])
np.dot(v,w)
```
Something to remember: Orthogonal vectors, i.e. vectors that meet at a 90 degree angle, have a dot product of zero.  

### Hadamard Multiplication
* element wise multiplication
```
v * w
```

### Outer Product

```
v = np.array([1,2,3])
w = np.array([5,6])
np.dot(v,w.T)  # yeilds a 3 x 2 matrix, a rank 1 matrix

# could also use np.outer() and avoid having to transpose
```

### Orthogonal Vector Decomposition
* break a vector into two vectors that meet constraints.  Could be for data compression
* two vectors, a and b, looking to find scalar B such that Ba is is a project of b onto a.  Aka, this is also the minimum distance between the two.  
* vector subtraction defines the line between b & Ba, i.e. b - Ba
* b - Ba will be orthogonal to Ba, so the dot product is 0
```
a.T(b - Ba) = 0
a.Tb -Baa.T = 0
a.Tb = Baa.T
B = a.Tb / a.Ta
```
* to do vector decomposition, use function above to find scalar of reference vector that is the orthogonal to the line between the target and the reference vector.  Called the parallel component
* this uses a slightly different formula, which finds a vector and not just the scaler
```
parallent component = r (t.Tr) / r.Tr
``` 
* then find the perpendicular component using vector substraction
```
t = perpedicular vector + reference vector
perpendciular vector = t - reference vector
```

### Vector Set
* collection of vectors

### Linear Weighted Combination
* mix info from multiple vectors with an option to apply different weights.  aka linear mixture, weighted combination
```
w = At + Bu + Cv
```

### Linear Independence
* linear dependent if one vector int he set can be expressed as a linear weight combo of other vectors in the set
* linear independent if no vector can be expressed as a linear weighted combo
* independence is a property of sets, not of vectors
* trival solution of all 0 scalars are ignored
```
# Equation for linear depedence
0 = At + Bu + Cv
# AKA this, where one vector is expressed as combo of the other two
At = Bu + Cv
```

### Subspace and Span
* For a finite set of vectors, there is an infinite set of weights to combine them creating the vector subspace
* mechanism for combining is called the span, a vector span as the different scalars are applied
* dimensionality of the subspace spanned by a set of vectors is the smallest number of vectors that forms a linearly independent set

### Basis
* set of rulers used to describe info int he matrix
* most common is a cartesian axis, the familiar XY plane 
```
# Cartesian axis
S = { [1,0], [0,1]}
```
* basis is an independent set of vectors that spans the subspace
* once it satisfies this defintion, you can use any that helps the problem.  Might be { [3,1], [-3,1]} instead

### Types of Matrices
* Random ```np.random.randn(Mrows, Ncols)```, Square, Diagonal, Triangular, Identity ```np.eye()```, Zeros ```np.zeros()```
* Symetric - rows and columns are equal, aka elements on opposite sides of the diagonal are the same

### Hadamard Multiplication
```
A * B
# OR 
np.multiply(A,B)
```

### Matrix Multiplication
* [[sum or elementwise multplication for row 1 and col 1], [... for row 1 and col 2], [row 2 and col 1], [row 2 and col 2]]
* M x N multiplied by N x K results in M x K.  Same N is required.  
* matrix that stores all the pairwise linear relationships between rows of the left matrix and columsn of the right matrix
* linear weight combos can use matrix multiplication, here 4 and 3 are the scalars for two columns: 
```
[3,1,
 0,2,
  6,5] @ [4,3]
```
* dot product is matrix multiplication on 1 x M and M x 1 matrices, resulting in 1 x 1 matrix


### Transpose
* swap rows and columns
```
A.T or np.transpose(A)
```
* multiply any matrix by its transpose to get a symetric matrix

### Matrix Norm
* vector norm is euclidean geometric length, square root of the sum of squared vector elements
* matrix norm more complicated;  There are multiple, each providing one number to characterize matrix.  || A || is norm for matrix A.
* two families:  element-wise or entry-wise, reflecting the magnitudes of the elements in the matrix.  induced norms, measuring how much a matrix transforms a vector
* element wise euclidean norm, or Frobenius:  square root of sum of all matrix elms
* often serve as cost function in minization algos to prevent large params or sparse solutions (ridge and lasso regressions)
* distance between two matrices, is frobenius matrix distance between C, which is A - B. 

### Matrix Trace
* sum of its diagonal elements, indicated as tr(A). 
* Exists only for square matrices
* trace of A.T@A = Fr. Norm of A, because A.T@A is the dot product of each row with itself

### Matrix Space




## Applications

### Pearson Correlation Coefficient
* uses dot product: 
```
  # compute cosine similarity
  num = np.dot(x,y) # numerator
  den = np.linalg.norm(x) * np.linalg.norm(y) # denominator
  cos = num / den

  # compute correlation (similar to above but mean-centered!)
  xm  = x-np.mean(x)
  ym  = y-np.mean(y)
  num = np.dot(xm,ym) # numerator
  den = np.linalg.norm(xm) * np.linalg.norm(ym) # denominator
  cor = num / den
```

### Time Series Filtering and Feature Detection
* a template, called a kernal is matched against portions of the a time series signal
```
# for loop: 
np.dot(kernel,signal[t-1:t+1])
```

