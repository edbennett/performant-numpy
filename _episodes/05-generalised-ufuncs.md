---
title: "Generalised ufuncs"
teaching: 30
exercises: 20
questions:
- "What if I want performant functions operating on Numpy arrays that aren't
element-wise?"
objectives:
- "Be able to use Numba to write custom generalised universal functions"
keypoints:
- "Use the `@guvectorize` decorator to turn elemental functions into ufuncs"
---

(Adapted from the [Scipy 2017 Numba tutorial](https://github.com/gforsyth/numba_tutorial_scipy2017/blob/master/notebooks/08.Make.generalized.ufuncs.ipynb).)

In the previous episode we looked at writing our own ufuncs using
Numba. These act on all elements of an array in the same way,
broadcasting like a scalar. Sometimes, however, we would like a
function that operates on more than one array element at once. Is
there a way we can do this, but still keep the ability to broadcast to
larger array shapes?

There is; such a function is called a *generalised ufunc*, and Numba
lets us define these using the `guvectorize` decorator.

With great power, however, comes great responsibility. The increased
flexibility of generalised ufuncs means that we need to give Numba
more information to allow it to work. For example,

~~~
import numpy as np
from numba import guvectorize

@guvectorize('int64[:], int64, int64[:]', '(n),()->(n)')
def g(x, y, result):
    for i in range(x.shape[0]):
        result[i] = x[i] + y
~~~
{: .language-python}

Now, this could be done with a simple broadcast, but it demonstrates a
point. We have had to make two declarations within the call to
`guvectorize`: the first resembles what we saw for parallel `ufuncs`;
we need to tell Numba what data types (and dimensionalities) we expect
to receive and output. The second declaration tells Numba what
dimensions are shared between inputs and outputs. In this case, we
expect the array (or array slice) `x` to be the same size as the
output array `result`, while `y` is a scalar number. Similarly to
`einsum`, more letters indicate more array dimensions.

The function body has another change from the scalar `ufunc` case,
too. We no longer have a `return` statement at the end; instead, we
have an explicitly-passed `result` variable that we edit. This is
necessary, as otherwise we would have to construct a data
structure to hold the output, since the whole point of generalised
ufuncs was to avoid being restricted to scalar outputs.

We can test this:

~~~
x = np.arange(10)
result = np.zeros_like(x)
result = g(x, 5, result)
print(result)
~~~
{: .language-python}

~~~
[ 5  6  7  8  9 10 11 12 13 14]
~~~
{: .output}

What happens if we try and use this like a more typical Numpy
function, which uses a return value rather than taking the output as
a parameter?

~~~
res = g(x, 5)
print(res)
~~~
{: .language-python}

~~~
[ 5  6  7  8  9 10 11 12 13 14]
~~~
{: .output}

This works, and is useful in cases where you are replacing an existing
function and want to maintain the existing interface (rather than
having to modify every calling point). However, it can be dangerous:
this effectively uses `np.empty` to declare the output array, so
unless you define all elements of the output array within the
function, then the behaviour is unpredictable (the same as using an
uninitialised variable in C-like languages).

Let's look at another example, a generalised ufunc for matrix
multiplication.

~~~
@guvectorize('float64[:,:], float64[:,:], float64[:,:]', 
            '(m,n),(n,p)->(m,p)')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
~~~
{: .language-python}

This now takes two rectangular arrays `A` and `B`, and returns a
rectangular array `C`, all of type `float64`. As you might expect, the
shapes are such that the first dimension of `A` and second dimension
of `B` form the dimensions of `C`, while the second dimension of `A`
must equal the first dimension of `B`.

As a sanity check, let's confirm that the identity matrix works as we expect

~~~
matmul(np.identity(5), np.arange(25).reshape((5, 5)), np.zeros((5, 5)))
~~~
{: .language-python}

~~~
array([[ 0.,  1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.,  9.],
       [10., 11., 12., 13., 14.],
       [15., 16., 17., 18., 19.],
       [20., 21., 22., 23., 24.]])
~~~
{: .output}

How does this perform on a slightly more substantial problem, and how
does it compare to Numpy's built-in matrix multiplication?

~~~
a = np.random.random((500, 500))
%timeit matmul(a, a, np.zeros_like(a))
%timeit a @ a
~~~
{: .language-python}

~~~
259 ms ± 4.04 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
24.7 ms ± 683 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
~~~
{: .output}

The performance stil leaves a little to be desired&mdash;ten times
slower than Numpy's built-in matrix multiplier.

> ## Writing signatures
>
> The function `ftcs` below uses second-order finite differences to
> solve a heat transfer problem. Use `guvectorize` to turn `ftcs` into
> a generalised ufunc, so that `run_ftcs` works properly. `ftcs` has
> already been written from the point of view of acting as a gufunc,
> so only the signature needs to be provided.
>
> ~~~
> import numpy
> from numba import guvectorize
>
> def ftcs(T, alpha, dt, dx, Tn):
>     I, J = T.shape
>     for i in range(1, I - 1):
>         for j in range(1, J - 1):
>             Tn[i, j] = (T[i, j] + 
>                       alpha * 
>                       (dt / dx**2 * (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) + 
>                        dt / dx**2 * (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1])))
>
>     for i in range(I):
>         Tn[i, 0] = T[i, 0]
>         Tn[i, J - 1] = Tn[i, J - 2]
>
>     for j in range(J):
>         Tn[0, j] = T[0, j]
>         Tn[I - 1, j] = Tn[I - 2, j]
>
> def run_ftcs():
>     L = 1.0e-2
>     nx = 101
>     nt = 1000
>     dx = L / (nx - 1)
>     x = numpy.linspace(0, L, nx)
>     alpha = .0001
>     sigma = 0.25
>     dt = sigma * dx**2 / alpha
>
>     Ti = numpy.ones((nx, nx), dtype=numpy.float64)
>     Ti[0,:]= 100
>     Ti[:,0] = 100
>
>     for t in range(nt):
>         Tn = ftcs(Ti, alpha, dt, dx)
>         Ti = Tn.copy()
>
>     return Tn, x
>
> Tn, x = run_ftcs()
> ~~~
> {: .language-python}
>
> The following will plot the solution as a check that the function is
> working correctly:
> ~~~
> from matplotlib import pyplot, cm
> %matplotlib inline
>
> pyplot.figure(figsize=(8, 8))
> mx, my = numpy.meshgrid(x, x, indexing='ij')
> pyplot.contourf(mx, my, Tn, 20, cmap=cm.viridis)
> pyplot.axis('equal');
> ~~~
> {: .language-python}
>
>> ## Solution
>>
>> ~~~
>> @guvectorize(['float64[:,:], float64, float64, float64, float64[:,:]'], 
>>             '(m,m),(),(),()->(m,m)', nopython=True)
>> ~~~
>> {: .language-python}
> {: .solution}
{: .challenge}

> ## Relocating loops
>
> The above `ftcs` example performs an inner loop, but the outer loop
> in time `t` is still done outside of our generalised ufunc, using a
> plain Python loop. Plain Python loops are frequently the enemy of
> performance, so try removing this loop and instead implementing it
> within `ftcs`.
>
> You will need to:
>
> * Modify the parameter list to accept a number of timesteps `nt`
> * Adjust the decorator's signature to match the new parameter
> * Add the extra loop over time
> * Incorporate the `Tn.copy()` operation from the old outer loop
> * Adjust `run_ftcs` to use a single call to the new function rather
>   than a looped call as at present
>
> How does this implementation compare performance-wise to the
> previous version using a pure Python outer loop?
>
>> ## Solution
>>
>> ~~~
>> @guvectorize(['float64[:,:], float64, float64, float64, int64, float64[:,:]'], 
>>             '(m,m),(),(),(),()->(m,m)', nopython=True)
>> def ftcs_loop(T, alpha, dt, dx, nt, Tn):
>>     I, J = T.shape
>>     for n in range(nt):
>>         for i in range(1, I - 1):
>>             for j in range(1, J - 1):
>>                 Tn[i,j] = (T[i, j] + 
>>                           alpha * 
>>                           (dt/dx**2 * (T[i + 1, j] - 2*T[i, j] + T[i - 1, j]) + 
>>                            dt/dx**2 * (T[i, j + 1] - 2*T[i, j] + T[i, j - 1])))
>> 
>>         for i in range(I):
>>             Tn[i, 0] = T[i, 0]
>>             Tn[i, J - 1] = Tn[i, J - 2]
>> 
>>         for j in range(J):
>>             Tn[0, j] = T[0, j]
>>             Tn[I - 1, j] = Tn[I - 2, j]
>> 
>>         T = Tn.copy()
>> ~~~
>> {: .language-python}
> {: .solution}
{: .challenge}
