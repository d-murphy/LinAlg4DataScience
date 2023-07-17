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
* Column Space - conceptualize matrix as set of column vectors, vector has a number of dimensions.  Each vector scaled by all possible real number scalers.  The dimensions of the column space is determined by how many columns are in the linearly independent set
* indicated by C(A)
* Ax = b, i.e is there a set of coefficients in x such that weight combo produces vector b.  if so, b E C(A) with E is a member of
```
[[1,3].T, [1,4].T] # this spans all of R^2
[[1,3].T, [2,6].T] # this only spans parallel to [1.3].T since not linearly ind.
[[1,5,1].T, [0,2,2].T] # column space is 2D embedde in R^3
```
* Row space is all possible weighted combos of the rows instead of the cols. Indicated as R(A). 
* Null space asks Ay = 0, what vector y satisfies this equation (excludes the 0 vector, a trivial solution).  Null space is "empty set"  if none.
  * null space is empty when columns of matrix form linear indepent set
  * ```scipy.linalg.null_space(A)```  returns a unit vector if it exists, .7071
  * null space is orthogonal to row space.  dot product of each row and null space is 0.  
* Right Null Space exists but not covered here.  

### Rank
* feature of a matrix expressed as nonnegative integer.  Indicated by r(A), described as "rank-r matrix".  Max is min of M or N, the row or col counts.  
* "full-rank" vs "reduced-rank" or "rank-deficient" or "singular"
* interpretations:  largest number of cols (or rows) forming a linearly independent set.  dimensionality of col space.  number of dimensions with info on matrix.  number of nonzero singular values of a matrix
* Col space and row space are different but dimensionality of matrix spaces, and so the rank, are the same
* Ranks on special matrices:  Vectors - 1, Zero - 0, Identity - N, Diagonal - Number of nonzero elms
* Matrix addition - resulting rank is <= rank(A) + rank(B); Matrix multiplication - resulting rank is min of rank(A) and rank(B)
* shifted matrices have full rank, even as correlation of elements is very high
* *Vector in Column Space* - Augment matrix with vector youre checking.  Calculate the rank of new and original.  If rank of new is higher than old, vector v is not in the column space, i.e. it adds new information to the matrix and the vector is not dependent on the others.  
* if rank = max possible rank, i.e. min{M, N}, then matrix is linearly ind.

### Determinant
* defined only for square matrices; zero for singular / reduced-rank matrices
* geometric interpretation: how much a matrix stretches vectors during matrix-vector multiplication.  A negative mean one axis is rotated.
* expressed as det(A)
* det([[a,b], [c,d]]) = ad - bc
* in 3x3, there are 6 products, 4x4 24
* additional geometric insight - det of 0 means at least one of axes is flattened during transformation

### Characteristic Polynomial
* combine matrix shifting with determinant.  
* read more some time

### Inverse
* Another matrix such that Inverse times A = I.  Allows matrix algebra such as solving for x in Ax = b, x = A.inv b
* Full for square, full-rank matrices; one-sided for rectangular; pseudo for all shapes and ranks
* For [[a,b], [c,d]] inverse is (1/ad-bc) * [[d,-b], [-c,a]]
* geometric explanation: inverse restores tranformation done by matrix.   for no inverse in singular matrices:  the dependent column flattens and once flattened can't be restored

### Orthogonal Matrix
* all columns are pair-wise orthogonal.  The norm of each column is 1. 
* Transpose of the matrix * the matrix is the identity matrix
* computer from a non-orthogonal via QR decomposition, aka a sophisticated version of Gram-Schmidt
* QR gives the original matrix A.  Transpose of Q * A = R 
* More numerically stable way to compute the matrix inverse, i.e. A Inv = R Inv * Q.T.  Still requires inverse of R, but triangular matrices stable through back substitution

### Gauss Jordan Elimination






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

