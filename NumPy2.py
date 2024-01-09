#!/usr/bin/env python
# coding: utf-8

# # Index

# In[ ]:


* 1: NumPy is Numerical Python
* 2: Module NumPy
* 3: One Dimensional arraay
* 4: To check a Dimension
* 5: Two Dimensional Array
* 6: shape
* 7: 3D Dimensional Array
* 8: Create 5 Dimensional Array by useing "ndmin"
* 9: Size
* 10: Data Types
* 11: astype( )
* 12: Indexing
*    12.1: Indexing in 2 Dimensional
*    12.2: Indexing in 3 Dimensional
* 13: Slicing
*    13.1: Slicing In 2D Array
* 14: Array Manipulations
*    14.1: Transpose( ) 
*    14.2: identity Matrix
*    14.3: Matrix Multiplication
* 15: Array Broadcasting
* 16: Reshape( )
* 17: Flattening Array
* 18: Resize
* 19: Stack( )
*    19.1: vstack( )
*    19.2: hstack( )
*    19:3: dstack
* 20: Unknown Quantity
* 21: Split( )
*    21.1: Spliting a Perticular Portion of Array
* 22: Universal functions and operations
*    22.1: Vectorization 
*    22.2: Trignometric Functions
*    22.3: Exponential
*    22.4: Log Function
*    22.5: NumPy Mean and Median Function
*    22.6: Standard deviation and variant
* 23: Sorting
*    23.1: Reverse Sorting
* 24: Searching
* 25: Fancy Index
* 26: Boolean Index
* 27: Iterating Over Arrays
*    27.1: Array Values Modification
* 28: Vectorized Operation
* 29: Time Complexity
* 30: Conditional Exception
* 31: Linear Algebra With NumPy
*    31.1: Scalar Multiply
*    31.2: Division in Matrix
* 32: Array Transpose ( )
* 33: Scalar Product or Dot Product
* 34: Cross Product
*    34.1: Cross Product in 2D Array
* 35: Matrix Determinant
* 36: Inverse
* 37: Eigen Value, Eigen Vectors
* 38: Trace
* 39: Rank of A Matrix
* 40: Solving Linear Equations


# # 24 Dec 2023

# # 0: Syllabus

# # 1: NumPy is Numerical Python [00:30:00]

# NumPy is sequence of elements, it is store in continous memory location and elements should be some data type.

# # 2: Module NumPy [00:47:00]

# In[1]:


import numpy as np


# In[2]:


print(np.__version__)


# In[3]:


print(dir(np))


# In[4]:


a = [1, 2, 3, 4]
print(type(a))


# In[5]:


'''In Output "numpy.ndarray" is a 
numpy.ndarray is package name
ndarray is a Nth dimensional array'''
arr = np.array(a)
print(arr)
print(type(arr))


# # 3: One Dimensional arraay [1:00:00]

# In[6]:


# no row no column it as have only elements
# [] To represents Dimension
arr_1D = np.array([1,2,3,4])
print(arr_1D)


# # 4: To check a Dimension[1:06:00]

# In[7]:


# To get Dimension of array we can use attribute "ndim"
print(arr_1D.ndim)


# # 5: Two Dimensional Array [1:09:00]

# In[8]:


arr_2D = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
# 3 Row 4 Column
print(arr_2D)
#To check Dimensional array
print(arr_2D.ndim)


# # 6: shape [1:14:00]

# shape is a attribute it indicate how row and column are present in matrix or array. 
# 
# use of shape attribute is a definding a structure of array

# In[9]:


print(arr_1D.shape)


# In[10]:


print(arr_2D.shape)
# 3rows and 4 column


# # 7: 3D Dimensional Array [1:16:00]

# In[11]:


arr_3D = np.array([
    [
        [1,2,3],[4,5,6],[7,8,9]
    ],
    [
        [10,11,12],[13,14,15],[16,17,18]
    ]
])
print(arr_3D)
print('___________________')
print("Dimensional: ",arr_3D.ndim)
print('___________________')
print("Shape :",arr_3D.shape)


# Above example 2 is a matrix, 3 is row and 3 is column

# # 8: Create 5 Dimensional Array by useing "ndmin" [1:24:00]

# In[12]:


arr_5D = np.array([1, 2, 3, 4, 5], ndmin= 5)
print(arr_5D)
print('___________________')
print("Dimensional :",arr_5D.ndim)
print('___________________')
print("Shape :",arr_5D.shape)


# Above example 5 is element of array or matrix, 1 are represent as element as single

# # 9: Size [1:27:00]

# Size is a attribute it indicate how much element are present in matrix

# In[13]:


print("Size :",arr_1D.size)
print('___________________')
print("Size :",arr_2D.size)
print('___________________')
print("Size :",arr_3D.size)
print('___________________')
print("Size :",arr_5D.size)


# # 10: Data Types [1:43:00]

# 1: i - integer
# 
# 2: b - boolean
# 
# 3: u - unsigned integer
# 
# 4: f - flot
# 
# 5: c - complex float
# 
# 6: m - timedelta
# 
# 7: M - datetime
# 
# 8: O - object
# 
# 9: S - string in binary Format
# 
# 10: U - Unicode string
# 
# 11: V - Fixed chunk of memory for other type(Void)

# Canverting One Data type to another Data type

# In[14]:


#example 1:
arr_1 = np.array([1, 2, 3, 4], dtype="i")
print(arr_1)


# In[15]:


#example 2:
arr_1 = np.array([1, 2, 3, 4], dtype="f")
print(arr_1)


# In[16]:


#example 3:
# String in binary format 
arr_1 = np.array([1, 2, 3, 4], dtype="S")
print(arr_1)


# In[17]:


#example 4:
#Unicode String
arr_1 = np.array([1, 2, 3, 4], dtype="U")
print(arr_1)


# # Error
# * A non integer string like 'a' can not be converted to integer

# In[115]:


b = np.array(['a','2','3'], dtype='i')
print(b)


# # 11: astype( )

# as type function is convert one data type to another

# In[18]:


arr = np.array([1, 2, 3, 4])
new_arr = arr.astype("f")
print(new_arr)


# In[19]:


#Data type of elements
print(new_arr.dtype)


# # 30 Dec 2023

# # 12: Indexing [00:19:00]

# In[20]:


arr = np.array([1, 2, 3, 4])
print(arr[2])


# # 12.1: Indexing in 2 Dimensional

# In[21]:


print(arr_2D)
print('___________________')
print(arr_2D[0][3])


# # 12.2: Indexing in 3 Dimensional

# In[22]:


print(arr_3D)
print('___________________')
print(arr_3D[1][2][2])


# In this code
# 
# print(arr_3D[1][2][2])
# 
# [1] is represent a matrix,
# [2] is represent a row,
# [2] is represent a column.

# # 13: Slicing [0:32:00]

# slicing means taking a element form part of the array.

# In[23]:


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr[3:8])
print('___________________')
print(arr[3:])
print('___________________')
print(arr[:5])
print('___________________')
print(arr[-5:-1])
print('___________________')
print(arr[0:8:3])


# # 13.1: Slicing In 2D Array

# In[24]:


print(arr_2D)
print('___________________')
print(arr_2D[1, 2])


# In[25]:


print(arr_2D[1:, 1:])


# Above Exampale The code is 
# 
# print(arr_2D[1:, 1:])
# 
# entire 2 Row and entire 2 Column

# # 14: Array Manipulations [0:44:00]

# In[26]:


#Find a even Number In matrix
print(arr_2D)
print('___________________')
print(arr_2D[arr_2D % 2 == 0])


# In[27]:


#Add a Two array
x = np.array([2, 3, 4])
y = np.array([4, 3, 2])
sum = np.add(x, y)
print(sum)


# In[28]:


# Finding power of array
a = np.array([1, 2, 3, 4])
print(a ** 2)


# # 14.1: Transpose( ) 

# Inter Changing Row and Column

# In[29]:


print(arr_2D)
print('___________________')
print(arr_2D.transpose())


# # 14.2: identity Matrix

# In[30]:


i = np.eye((4))
print(i)


# In[31]:


i = np.eye((4), dtype='i')
print(i)


# In[32]:


j = np.ones((3, 3))
print(j)


# In[33]:


j = np.ones((3, 3), dtype='U')
print(j)


# In[34]:


print(i)
print('___________________')
print(i.astype(bool))


# # 14.3: Matrix Multiplication

# In[35]:


a = np.array([
    [1, 2, 3],[4, 5, 6],[7, 8, 9]
])

b = np.array([
    [1, 2, 3],[4, 5, 6],[7, 8, 9]
])
print(a)
print('___________________')
print(b)
print('___________________')
print(np.matmul(a, b))


# In[ ]:


'''
[(1x1+2x4+3x7) (1x2+2x5+3x8) (1x3+2x6+3x9)]]

[(4x1+5x4+6x7) (4x2+5x5+6x8) (4x3+5x6+6x9)]

[(7x1+8x4+9x7) (7x2+8x5+9x8) (7x3+8x6+9x9)]


[(30)    (36)    (42)]

[(72)    (81)    (96)]

[(102)   (126)  (150)]
'''


# # 15: Array Broadcasting [1:07:00]

# Array Broadcasting means Manipulating diffent shape of Array.
# 
# In Broadcasting One should be matrix another shoyld be 1D Array.
# 
# In this situation only work.

# In[36]:


a = np.arange(5)
print(a)
print('___________________')
print(a+5)


# In[37]:


a = np.ones((3, 3), dtype='i')
print(a)
print('___________________')

b = np.arange(3)
print(b)
print('___________________')
print(a+b)


# In[38]:


a = np.arange(3, 6)
print(a)
print('___________________')

b = np.arange(3).reshape(3, 1)
print(b)
print('___________________')

print(a+b)


# In[39]:


a = arr_3D
b = np.arange(3)
print(a)
print('___________________')
print(b)
print('___________________')
print(a + b)


# In[40]:


a = np.array([1, 2, 3])
print(a)
print('___________________')

b = np.array([
    [4], [5], [6]
])
print(b)
print('___________________')

c = a + b
print(c)


# # 16: Reshape( ) [1:37:00]

# Reshape gives new shape without changing data.
# 
# Reshape function are not effect Original array.

# In[41]:


arr = np.arange(12)
print(arr.shape)


# In[42]:


arr.reshape(4, 3)


# In[43]:


#arr converting to 3D
print(arr.reshape(2, 3, 2))


# # 17: Flattening Array [1:47:00]

# Converting to Higher Dimensional to One Dimensional.

# In[44]:


print(arr_2D)
print('___________________')
print(arr_2D.reshape(-1))


# # 18: Resize [1:51:00]

# resize it's effect both Original and New arrays.

# In[45]:


#Creating Array
data = np.array([10,20,30,40,50,60,70,80])
print(data)
print('___________________')

#Reshape
print(data.reshape(2,4))
print('___________________')

#after reshape print data
print(data)
#reshape not effect orginal array
print('___________________')

#resize( ) change orginal array 
#line 17 resize done
print(data.resize(4, 2))
print('___________________')
#now we can print resied array
print(data)


# # 31 Dec 2023

# # 19: Stack( ) [00:19:00]

# * Combining two or more array is called stack.
# 
# * In stack process, if I want to stack, both are one dimensional array. The result is get two dimensional.
# 
# * Axis 0 is 1st Dimensional or row
# 
# * Axis -1 is Last Dimensional
# 
# * Axis 1 is Column

# In[46]:


x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
print(np.stack((x, y)))


# In[47]:


# axis 0 is repersent row
print(np.stack((x, y), axis=0))
print('___________________')

# axis 1 is represent column
print(np.stack((x, y), axis=1))


# # 19.1: vstack( )[00:29:00]

# * vstack means stacking a vertical order along the row vise.

# In[48]:


print(np.vstack((x, y)))


# # 19.2: hstack( )[00:30:00]

# * hstack means stacking a horizondal order along the column vise.

# In[49]:


print(np.hstack((x, y)))


# In[50]:


a = np.array([
    [1, 2, 3, 4], [5, 6, 7, 8]
])
b = np.array([
    [9, 10, 11, 12], [13, 14, 16, 18]
])
print(a)
print('___________________')

print(b)
print('___________________')
print(np.hstack((a, b)))
print('___________________')
print(np.vstack((a, b)))


# # 19:3: dstack [0:36:20]

# * dstack means stacking a death stack it's also called hight stack.
# 
# * depth represent in array 3D.
# 
# * dstack any D array convert to 3D.

# In[51]:


print(a)
print('___________________')
print(b)
print('___________________')
print(np.dstack((a, b)))


# In[52]:


print(x)
print('___________________')
print(y)
print('___________________')
print(np.dstack((x, y)))


# In[53]:


a = np.array([[1,2],[3,4]])
print(a)
print('___________________')
b = np.array([[1], [2]])
print(b)
print('___________________')
c = b.reshape((1, 2))
print(c)
print('___________________')
print(np.vstack((a, c)))


# # 20: Unknown Quantity[0:42:00]

# In[54]:


a = np.array([
    [1,2],[3,4]
])
print(a)
print('___________________')
b = np.array([
    [5], [6]
]).reshape(1, 2)
print(b)
print('___________________')
print(np.vstack((a, b)))


# In[55]:


b = np.array([
    [5], [6]
]).reshape(1, -1)
print(np.vstack((a, b)))


# * In this above example "reshape(1, -1)" represent 1 row -1 is represent take a column you need to join.

# In[56]:


b = np.array([
    [5], [6]
]).reshape(-1, 2)
print(np.vstack((a, b)))


# * In this above example "reshape(-1, 2)" represent -1 row is represent take a row you need to join and 2 column.
# 
# * If "reshape(-1, -1)" can't given like this it will be get Error. Can't give a both row and column are unknow(-1).
# 
# * Only One unknow can given

# # 21: Split( ) [0:46:00]

# In[57]:


print(arr_2D)
print('___________________')
print(np.split(arr_2D, 3, axis=0))
print('___________________')
print(np.split(arr_2D, 4, axis=1))
print('___________________')
b = np.array_split(arr_2D, 6, axis=0)
print(b)
print('___________________')
for i in b:
    print(i)


# # 21.1: Spliting a Perticular Portion of Array [0:51:40]

# In[58]:


s = np.split(arr_2D, indices_or_sections=
             [1],axis=1)

print(s)
print('___________________')
for i in s:
    print(i)


# In[59]:


s = np.split(arr_2D, indices_or_sections=
             [2],axis=1)

print(s)
print('___________________')
for i in s:
    print(i)


# # 22: Universal functions and operations[1:00:00]

# # 22.1: Vectorization 

# In[60]:


sal = np.array([20000, 30000, 40000])
print(sal + 5000)


# # 22.2: Trignometric Functions[1:03:00]

# In[61]:


print(np.sin(90))
#This output is radian


# In[62]:


print(np.cos(0))


# In[63]:


print(np.tan(0))


# In[64]:


# This output is redian
arr = np.array([0, 45, 60])
print(np.tan(arr))


# In[65]:


#This output in degree
print(np.tan(arr * (np.pi / 180)))


# # 22.3: Exponential [1:09:00]

# In[66]:


#e ** 3
np.exp(3)


# # 22.4: Log Function

# In[67]:


np.log(1)


# In[68]:


log_num = np.array([1, 2, 3, 4])
print(np.log(log_num))


# In[69]:


print(np.log10(log_num))


# # 22.5: NumPy Mean and Median Function[1:13:00]

# In[70]:


import statistics as s


# In[71]:


# Avg 
s.mean([10, 20, 30])


# In[72]:


#After a Sorting get Medial Value
print(s.median([10, 20, 30]))
print('___________________')
print(s.median([2, 1, 5, 3, 4]))


# In[73]:


np.mean([10, 20, 30])


# In[74]:


np.median([10, 20, 30])


# # 22.6: Standard deviation and variant[1:21:00]

# In[75]:


# Standard deviation 
a = np.array([
    [1, 2], [3, 4]
])
print(np.std(a))


# In[76]:


# Variant
print(np.var(a))


# # 23: Sorting [1:26:00]

# In[77]:


a = np.array([22, 33, 4, 11, 66]) 
print(np.sort(a))


# In[78]:


#Sorting in 2D 
b = np.array([
    [5, 3, 2],[20, 1, 16], [1, 7, 3]
])
print(np.sort(b))


# # 23.1: Reverse Sorting [1:28:30]

# In[79]:


print(b)
print('___________________')
k = np.sort(b)
print(k[::-1])


# In[80]:


a = np.array([1,0,4,6,8,2,9])
rs = np.sort(a)
print(rs[::-1])


# # 24: Searching [1:32:00]

# Searchsorted function work after sorted and then search

# In[81]:


arr = np.array([12,93,24,52])
op = np.searchsorted(arr, 93)
print(op)


# In[82]:


arr = np.array([5, 12, 56, 7,2,18,1,100])
print(arr)
print('___________________')
print(np.sort(arr))
print('___________________')
print(np.searchsorted(arr, 1))


# # 25: Fancy Index[1:48:51]

# manipulating a multiple array at one time is Called Fancy indexing

# In[83]:


arr = np.array([23,17,55,76,12,33,2,99])
print(arr)
print('___________________')
ind = [1, 3, 6]
# insert New value at this index position
newvalue = [10, 20, 30]
arr[ind] = newvalue
print(arr)


# In[84]:


#fancy index to select specific rows
arr = np.array([
    [1, 2, 3],[15, 17, 19],[40, 20, 50]
])
print(arr)
print('___________________')
#Selecting Specific row index 0 and 2
rows = arr[[0,2],:]
print(rows)
print('___________________')
columns = arr[:,[0, 2]]
print(columns)


# # 1 Jan 2024

# # 26: Boolean Index [00:12:19]

# * If want a output in True or False in this case we can use boolean indexing.
# 
# * We can also filter the elements by useing a boolean indexing.

# In[85]:


data = np.arange(16)
print(data)
print('___________________')

# Condition to filter the data
con = (data % 3 == 0)
print(con)
print('___________________')

# Create boolean mask
mask = np.array(con, dtype=bool)
print(mask)
print('___________________')


filter_data = data[mask]
print(filter_data)


# # 27: Iterating Over Arrays [00:24:00]

# * Iterating object is **nditer** it is a multi dimensional iterating attribute.
# * We can also use **for** or **while** loops

# In[86]:


b = np.array([
    [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
])
print(b)
print('___________________')

# Iterating a arraay
for i in np.nditer(b):
    print(i, end=' ')


# #  27.1: Array Values Modification [00:29:54]

# * When Iterating array We can modifi the array by useing **op_flag** like a readonly, write only and readwrite

# In[87]:


#writeonly
print(b)
print('___________________')

# Calling a iterating
for i in np.nditer(b, op_flags=['writeonly']):
    # '[...]' 3 dot means calling a all element
    i[...] = i * 5
    print(i, end=' ')
print('\n___________________')
print(b)


# In[88]:


#readwrite
print(b)
print('___________________')

# Calling a iterating
for i in np.nditer(b, op_flags=['readwrite']):
    # '[...]' 3 dot means calling a all element
    i[...] = i * 5
    print(i, end=' ')
print('\n___________________')
print(b)


# # Error

# In[116]:


#readonly
print(b)
print('___________________')

# Calling a iterating
for i in np.nditer(b, op_flags=['readonly']):
    # '[...]' 3 dot means calling a all element
    i[...] = i * 5
    print(i, end=' ')
print('\n___________________')
print(b)


# # 28: Vectorized Operation[00:37:00]

# * Vectorized operation means element wise operation on Array.
# * Vectorized operation useing for eliminate the need for loop concpet.
# * Vector is a Single dimensional elements.

# In[89]:


a = np.arange(20)
print(a)
print('___________________')

result = a + 2000
print(result)


# In[90]:


# Sum of 10 like a
# 1+2+3+4+5+6+7+8+9 = 45
print(np.sum(np.arange(10)))
print('___________________')
print(np.sum(np.arange(100)))
print('___________________')
print(np.sum(np.arange(10000)))


# # 29: Time Complexity [00:45:39]

# * Time complexity is calculating the time how much time taken give a output.
# * Output is a in microsecond.
# * For checking a time complexity we can use **time** module.

# In[91]:


import time


# In[92]:


#Cheking a time  executive the  programme
start = time.time()

#Start time module
print(start)
print('___________________')

#write a programme to check time
print(np.sum(np.arange(100000000)))
print('___________________')

#Ending the time
end = time.time()
print(end)
print('___________________')

#time taken in microsecond
print(end-start)


# # 30: Conditional Exception [00:50:00]

# * Conditional exception is giveing a condition to get result, in this we can use a **where(condition)** function.

# In[93]:


a = np.arange(20)
print(a)
print('___________________')

#Giveing a condition
b = np.where(a > 3, a * 2, a)
print(b)


# In the above example code **b = np.where(a > 3, a * 2, a)** is
# * **a>3** is Condition 
# * If Condition is True then multiply by two **a * 2** 
# * If Condition is False then print as it is **a**

# In[94]:


#Finding a index of 4
a = np.array([1, 2, 3, 4, 5, 6, 4, 3, 2, 4, 6, 7, 4])
print(a)
print('___________________')

b = np.where(a == 4)
print(b)


# In[95]:


#Finding a multiples of three and multipy by two
# multiples of three is: 0,3,6,9,12,15,18
a = np.arange(20)
print(a)
print('___________________')

b = np.where(a % 3 == 0, a * 2, a)
print(b)


# # 31: Linear Algebra With NumPy [00:59:00]

# In[96]:


# Addition of Matrix
arr1 = np.arange(9)
a = arr1.reshape(3, 3)
print(a)
print('___________________')

arr2 = np.arange(10, 19)
b = arr2.reshape(3, 3)
print(b)
print('___________________')

print(a + b)


# In[97]:


#Subtraction of matrix
print(b - a)


# # 31.1: Scalar Multiply [1:05:00]

# * Multiply by one number into all array elements

# In[98]:


#Multiply by 5 into Array a
print(a)
print('___________________')
print(a * 5)


# # 31.2: Division in Matrix [1:06:00]

# In[99]:


print(b)
print('___________________')
print(b/2)


# # 32: Array Transpose ( ) [1:07:00]

# * **Transpose()** function is a converting matrix to row to columns and columns to row.
# * In this function not reflect a original Array.

# In[100]:


print(a)
print('___________________')
print(np.transpose(a))
print('___________________')
print(a)


# # 33: Scalar Product or Dot Product [1:08:30]

# * It is also called inner product
# * We can use a **np.inner( )** or **np.dot( )** function

# In[101]:


a = np.arange(3)
b = np.arange(4, 7)
print(a)
print('___________________')
print(b)
print('___________________')
print(np.dot(a, b))
print('___________________')
print(np.inner(a, b))


# In the above example
# * array **a = [a11  a12  a13]**
# * array **b = [b11  b12  b13]**
# * In dot product = **[(a11xb11)+(a12xb12)+(a13xb13)]**
# * **[(0x4)+(1x5)+(2x6)] = (0+5+12) = 17**

# # 34: Cross Product [1:12:42]

# * Cross product also called vector products.
# * We are useing **np.cross( )** function.
# * In Cross Product Array Size should be some like a 3x3, 4x4 etc.

# In[102]:


a = [4, 3]
b = [5, 2]
result = np.cross(a, b)
print(result)


# In the above example
# * array a = [a11, a12]
# * array b = [b11, b12]
# * result = [(a11xb12)-(b11xa12)]
# * [(4x2)-(5x3)] = [8-15] = -7

# # 34.1: Cross Product in 2D Array

# In[103]:


a = [[1, 2], [3, 4]]

b = [[5, 6], [7, 8]]

result = np.cross(a, b)
print(result)


# # 35: Matrix Determinant [1:20:00]

# * In NumPy we don't have Determinant function so we calling a Linear Algebra Sub module.
# * **np.linalg.det( )** function for Determinant.

# In[104]:


a = np.arange(1,10).reshape(3, 3)
print(a)
print('___________________')
d = np.linalg.det(a)
print(int(d))


# In[ ]:


'''
In the above example
* [a11, a12, a13],
* [a21, a22, a23],
* [a31, a32, a33]

* d = a11[(a22xa33)-(a32xa23)]-a12[(a21xa33)-(a31xa23)]+a13[(a21xa32)-(a31xa22)]
   
    = 1[(5x9)-(8x6)]-2[(4x9)-(7x6)]+3[(4x8)-(7x5)]
    
    = 1[45 - 48] -2[36 - 42] +3[32 - 35]
    
    = 1(-3) -2(-6) +3(-3)
    
    = -3 +12 -9
    
    = 0
'''


# # 36: Inverse [1:25:00]

# A-inv = 1 / |A| AT

# In[105]:


a = np.array([
    [2, 0, -1],[5, 1, 0],[0, 1, 3]
])
print(a)
print('___________________')
a_inv = np.linalg.inv(a)
print(a_inv)


# In[ ]:


''' 
In the above example

    A                  I
* [2, 0, -1]         [1, 0, 0]          
* [5, 1, 0]          [0, 1, 0]
* [0, 1, 3]          [0, 0, 1]

R1x5
R2x2

* [10,   0,  -5]   [5,  0, 0]
* [-10, -2,   0]   [0, -2, 0]
* [0,    1,   3]   [0,  0, 1]

R2+R1 = R2

* [10, 0, -5]  [5,  0, 0]
* [0, -2, -5]  [5, -2, 0]
* [0,  1,  3]  [0,  0, 1]

R2 + 2R3 = R2

* [10, 0, -5]       [5,  0, 0]
* [0,  0,  1]       [5, -2, 2]
* [0,  1,  3]       [0,  0, 1]

R1+5R2 = R1
R3-3R2 = R3

* [10, 0, 0]         [30,  -10, 10]
* [0,  0, 1]         [5,   -2,   2]
* [0,  1, 0]         [-15,  6,  -5]

1/10 R1 = R1

* [1, 0, 0]      [3,  -1,  1]
* [0, 0, 1]      [5,  -2,  2]
* [0, 1, 0]      [-15, 6, -5]

INTER CHANGE A ROW R2 to R3

* [1, 0, 0]    [3,  -1,   1]
* [0, 1, 0]    [-15, 6,  -5]
* [0, 0, 1]    [5,  -2,   2]
'''


# In[106]:


a = np.random.randint(1, 10,[3, 3])
print(a)


# In[107]:


inv = np.linalg.inv(a)
print(inv)


# # 37: Eigen Value, Eigen Vectors [1:30:00]

# * In **np.linalg.eigvals()** That is function only get Eigen Value

# In[108]:


a = np.arange(9).reshape(3, 3)
print(a)
print('___________________')
ev = np.linalg.eigvals(a)
print(ev)


# * If we want both Eigen Value and Eigen Vectors we can use **np.linalg.eig()** function

# In[109]:


a = np.arange(9).reshape(3, 3)
evl, evv = np.linalg.eig(a)
print("Eigen Value", evl)
print('___________________')
print("Eigen Vector\n", evv)


# # 38: Trace [1:40:00]

# * We want to find a sum of diagonal element in matrix we can use a **np.trace()** function.

# In[110]:


a = np.arange(1, 10).reshape(3, 3)
print(a)
print('___________________')
print(np.trace(a))


# * Output is 1+5+9 = 15

# # 39: Rank of A Matrix [1:42:00]

# In[111]:


print(a)
print('___________________')
print(np.linalg.matrix_rank(a))


# In[112]:


c = np.eye((8),dtype='i')
print(c)
print('___________________')
print(np.linalg.matrix_rank(c))


# # 40: Solving Linear Equations[1:44:00]

# In[113]:


# In here Define the Coefficients
# ax+b = 0
a = np.array([[3]])

#Constants
b= np.array([6])

#Solve
x = np.linalg.solve(a, -b)
print(x)


# In[ ]:


'''
In the above example
ax + b = 0
3x + 6 = 0
3x = -6
x = -6/3 = -2 
'''


# In[114]:


#Coefficients of x and y
a = np.array([
    [3, 2], [1, -1]
])

#Define Vector of constants
b = np.array([7, -1])

# Solve
x = np.linalg.solve(a, b)
print(x)


# In[ ]:


'''
3x + 2y = 7
x - y = -1

Multiply by 2 for b array

3x + 2y = 7
2x - 2y = -2
____________
5x + 0y = 5

5x = 5
 x = 5 / 5
 x = 1
 __________________
 
 x - y = -1
 1 - y = -1
   - y = -1 - 1
   - y = -2
     y =  2
     
  (x, y) = (1, 2)
  '''

