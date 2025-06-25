import numpy as np

x = np.arange(5) # 1D array (not list) with 5 elements from 0 to 4 =>array([0,1,2,3,4])
print(x)
# arange is analogous to range() in python. we can upto 3 parameters as per our requirement
'''Np arrays are faster than python lists except for append operation.
indexing an array: by a tuple of integers, booleans, by another array or by integers
rank of array: no of dimensions
shape of array : size of array along each dimension => returns a tuple of integers '''

# python list to np.array
b = np.array([1.,2.,3.])
print(b)

''' No space is reserved at the end of the array to facilitate (appends in O(1) as in lists)
So 2 ways for dynamic memory:
1. grow a python list by append and then convert to array
2. np.zeros()'''

c = np.zeros(3,int) # => [0 0 0], dtype = np.int32

# creating an empty array that mathces the existing one's shape and datatype
print(c)
d = np.zeros_like(c)
print(d)
print(np.shape(x)) # => (5,)

print(np.zeros(3), np.ones(3), np.full(3,7),np.ones_like(c)) # zeros like, ones like, empty like , full like
# don't use np.empty() or empty_like bcz it is giving garbage initialization
# np.zeros(3), ones(3) => without providing datatype, default is float

# print(list(range(0,0.5,6))) # error range always handles integer elements
print(list(range(0,7,6))) # only gives [0,6] because in 0 to 7 we can jump 6 elements only 1 time
e = np.linspace(0,0.5,6) # for creating monotonically increasing sequence of floating elements
print(e) # [0.  0.1 0.2 0.3 0.4 0.5]

print(np.arange(0.4,0.8,0.2)) # [0.4 0.6]
print(np.arange(0.5, 1, 0.25)) # [0.5  0.75]
print(np.linspace(0.5,0.7,3)) # [0.5 0.6 0.7]

# random numbers

print(np.random.randint(0,10,3)) # generate any 3 random numbers in any order as array from 0 to 10
''' this implies that prob of a number generated from 0 to 10 = 1/10
uniform x E [0,10)'''

print(np.random.rand(3)) # generate any 3 random float numbers of any decimal places (ex: [0.1 0.01 0.001]) from 0 to 1 in any order by default
''' uniform x E [0,1)'''
print(np.random.uniform(1,10,3)) # generate any 3 random float numbers of any decimal places (ex: [1 1.01 1.001]) in any order from 1 to 9
''' uniform x E [1,10)'''

''' np.random.randint(0,10) is [0,10) but random.randint(0,10) is [0,10]'''

print(np.random.randn(3)) # 3 random numbers belonging to std normal distribution
''' std normal , mean = 0, std = 1'''

print(np.random.normal(5,2,3)) # 3 random numbers belonging to  normal distribution
''' normal , mean = 5,std = 2 '''

# indexing an array

a = np.array([1,2,3,4,5,7,6]) # [1 2 3 4 5 7 6]

print(a[1], a[2:4], a[-2:], a[::2]) # just like slicing of lists 
#     2     [3,4]   [7 6]   [1 3 5 6]
# one new feature is fancy indexing
print(a[[1,3,4]]) # [2 4 5]
a[2:4] = 0 # [1 2 0 0 5]
print(a)
# boolean indexing
print(a > 5) # [False False False False False  True  True]
bl = np.any(a>5)
print(bl) # true
print(a[a>5]) # [7 6]
print(np.all(a>5)) # false

a[a<5]  = 0
a[a>=5] = 1
''' np.where(a>5) returns all the indices i where a[i] > 5.
combinedly we can write the above 2 stmts as np.where(a>=5, 1,0)'''

a[a<2] = 2
a[a>5] = 5
''' combinedly, np.clip(a,2,5)'''

print(a,x) # [2 2 2 2 2 2 2] [0 1 2 3 4]
# array operations
print(a.shape)
print(x.shape)
print(a[:5]+x)
print(a[:5]-x)
print(a[:5]*x) # [0 2 4 6 8]

x[x<1] = 1
print(a[:5]/x)
print(a[:5]//x)
print(a**2)
print(np.sqrt(a))
print(np.exp(a))

print(np.log(a))
print(np.dot(a[:5],x))
print(np.cross(a[:3],x[:3])) # for cross product, dimensions of each array must be 2 or 3
print(np.max(a))
print(np.argmax(a))
print(np.mean(a),np.sum(a),np.var(a),np.std(a))

# 2D arrays

a2 = np.array([[1,2,3],[4,5,6]]) # shape = (2,3)
z2 = np.zeros((3,2))
f2 = np.full((3,2),7)
o2 = np.ones((3,2))
# new one
e2 = np.eye(3,3) # 3x3 identity matrix of floating 1s and zeros

print(len(a2)) # len(a2) = a2.shape[0] # no of rows

# for random numbers, just replace the shape with shape of the matrix (row, col)
print(np.random.randint(0,10,(3,2)))

# indexing 2 D
print(a2)
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a[1,2]) # element at (1,2) index
print(a[1,:],a[1]) # 1th row of a = a[1]
print(a[:,2]) # 2th column of a
print(a[:,1:3]) # gives the 1th and 2th column combined leaving 3
print(a[-2:,-2:]) # start from last 2nd row to last row, start from last 2nd col to last col
print(a[::2,1::2]) # jump 2 rows (curr row to curr+2 th row), start from 1th column and jump  2 columns and take untill last column

# axis sigma aij => i=> axis = 0, j => axis =1
print(a.sum())
print(a.sum(axis=0)) # column wise sum (sigma i aij)
print(a.sum(axis = 1)) # row wise sum (sigma j aij)

# matrix operations

x = np.random.randint(1,5,(2,2))
y = np.random.randint(1,5,(2,2))

print(x+y)
print(x-y)
print(x * y) # element wise multiplication ( ith row ith col of x with ith row ith col of y)
print(x @ y) # actual multiplication of matrices
print(x/y) # element wise division => used for normalization
print(x/9)
print(x * [1 -1])

print(x.min())
print(x.min(axis=0))
print(x.max(axis = 1))

# nD array
an = np.arange(1,9).reshape(2,(2,2)) # 1 matrix containign 2 matrices with each (2,2) shape => partition matrix 
                            # number of mat, dimension of each one



