#%%
print("Hello world")

# %%
import numpy as np

ar = np.array([0,1,2,3,4,5,6,7,8,9])
ar
type(ar)

# %%
# 벡터연산이 가능하다.
data = [0,1,2,3,4,5,6,7,8,9]
x = np.array(data) * 2
x

# %%
a = np.array(range(0,5))
b = np.array(range(5,10))

2 * a + b

a == 2

b > 7

(a == 2) & (b > 10)

# %%
c = np.array([[0,1,2],[3,4,5]])
print(c)
len(c[0])

# %%
arr = np.array([range(10,41,10),range(50,81,10)])
arr2 = np.array(range(10,81,10)).reshape(2,4)
arr2 == arr
# arr.shape
# %%
#3차원 배열 만들기
import random
arr3 = np.array(random.sample(range(0,100),18)).reshape(2,3,3)
print(arr3)
print(arr3[0][1][0])
print(arr3[1][1][0])

arr3.shape
arr3.ndim
# %%
#배열 인덱싱
a = np.array([[0, 1, 2], [3, 4, 5]])

a[:2,:2]
# %%
#연습문제
m = np.array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]])
m

print(m[1,2])
print(m[-1,-1])
print(m[1,1:3])
print(m[1:3,2])
print(m[:2,3:])
# %%
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
a[a%2==1]
# %%
a = np.array(range(1,13)).reshape(3,4)
a[:,[True,False,True,False]]
# %%
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

#3의 배수
print(x[x%3==0])
print(x[x%4==1])
print(x[(x%3==0) & (x%4==1)])
x.dtype
# %%
np.array([0, 1, -1, 0]) / np.array([1, 0, 0, 0])
np.log(0)
np.exp(-np.inf)
# %%
a = np.zeros(5, dtype='i')
a

b = np.zeros((2,3),dtype='f')
b
# %%
g = np.empty((4, 3))
g


# %%
a = np.arange(10).reshape(2,5)
b = np.arange(3,21,2).reshape(3,3)
print(b)
# %%
np.linspace(0, 100, 10)  # 시작, 끝(포함), 갯수
b.T
# %%
a = np.arange(12)
a.reshape(2,-1,2)
# %%

a = np.zeros((3,3))
b = np.ones((3,2))
c = np.hstack([a,b])
d = np.arange(10,151,10).reshape(3,5)
e = np.vstack([c,d])
f = np.tile(e,(2,1))


# print(a)

# print(b)
# print(c)
# print(e)
print(f)



# %%
#meshgrid

x = np.arange(3)
y = np.arange(5)

X,Y = np.meshgrid(x,y)

print(X)
print(Y)

[list(zip(x,y)) for x,y in zip(X,Y)]
# %%

x = np.arange(1, 10001)
y = np.arange(10001, 20001)

x + y

# %%

aa = np.arange(10)

np.exp(aa)

# %%

arr = np.arange(30).reshape(5,6)

print(arr.max())
print(arr.sum(axis=1))
print(arr.max(axis=1))
print(arr.mean(axis=0))
print(arr.min(axis=0))

# %%
a = np.array([42, 38, 12, 25])
j = np.argsort(a)
j
# %%
arr = np.array([[  1,    2,    3,    4],
       [ 46,   99,  100,   71],
       [ 81,   59,   90,  100]])

arr.sort(axis=0)
arr