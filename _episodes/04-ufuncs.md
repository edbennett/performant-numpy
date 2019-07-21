---
title: "Custom ufuncs"
teaching: 20
exercises: 20
questions:
- "What can I do if Numpy's built-in ufuncs don't do what I need them to?"
objectives:
- "Be able to use Numba to write custom universal functions"
keypoints:
- "Use the `@vectorize` decorator to turn elemental functions into ufuncs"
---

We know that due to various design decisions in Python, programs written
using pure Python operations are slow compared to equivalent code written
in a compiled language. We have seen that Numpy provides a lot of
operations written in compiled languages that we can use to escape from
the performance overhead of pure Python. However, sometimes we do still
need to write our own routines from scratch. This is where Numba comes in.
Numba provides a *just-in-time compiler*. If you have used languages like
Java, you may be familiar with this. While Python can't easily be compiled
in the way languages like C and Fortran are, due to its flexible type
system, what we can do is compile a function for a given data type once
we know what type it can be given. Subsequent calls to the same function
with the same type make use of the already-compiled machine code that was
generated the first time. This adds a significant overhead to the first
run of a function, since the compilation takes longer than the less
optimised compilation that Python does when it runs a function; however,
subsequent calls to that function are generally significantly faster.
If another type is supplied later, then it can be compiled a second time.

Numba makes extensive use of a piece of Python syntax known as
"decorators". Decorators are labels or tags placed before function
definitions and prefixed with `@`; they modify function definitions,
giving them some extra behaviour or properties.


## Universal functions in Numba

(Adapted from the
[Scipy 2017 Numba tutorial](https://github.com/gforsyth/numba_tutorial_scipy2017/blob/master/notebooks/07.Make.your.own.ufuncs.ipynb))

Recall how Numpy gives us many operations that operate on whole arrays,
element-wise. These are known as "universal functions", or "ufuncs"
for short. Numpy has the facility for you to define your own ufuncs,
but it is quite difficult to use. Numba makes this much easier with
the `@vectorize` decorator. With this, you are able to write a
function that takes individual elements, and have it extend to operate
element-wise across entire arrays.

For example, consider the (relatively arbitrary) trigonometric
function:

~~~
import math

def trig(a, b):
    return math.sin(a ** 2) * math.exp(b)
~~~
{: .language-python}

If we try calling this function on a Numpy array, we correctly get an
error, since the `math` library doesn't know about Numpy arrays, only
single numbers.

~~~
%env OMP_NUM_THREADS=1
import numpy as np

a = np.ones((5, 5))
b = np.ones((5, 5))

trig(a, b)
~~~
{: .language-python}

~~~
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-1-0d551152e5fe> in <module>
      9 b = np.ones((5, 5))
     10 
---> 11 trig(a, b)

<ipython-input-1-0d551152e5fe> in trig(a, b)
      2 
      3 def trig(a, b):
----> 4     return math.sin(a ** 2) * math.exp(b)
      5 
      6 import numpy as np

TypeError: only size-1 arrays can be converted to Python scalars
~~~
{: .output}

However, if we use Numba to "vectorize" this function, then it becomes
a ufunc, and will work on Numpy arrays!

~~~
from numba import vectorize

@vectorize
def trig(a, b):
    return math.sin(a ** 2) * math.exp(b)

a = np.ones((5, 5))
b = np.ones((5, 5))

trig(a, b)
~~~
{: .language-python}

How does the performance compare with using the equivalent Numpy
whole-array operation?

~~~
def numpy_trig(a, b):
    return np.sin(a ** 2) * np.exp(b)
	

a = np.random.random((1000, 1000))
b = np.random.random((1000, 1000))

%timeit numpy_trig(a, b)
%timeit trig(a, b)
~~~
{: .language-python}

~~~
Numpy: 19 ms ± 168 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
Numba: 25.4 ms ± 1.06 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
~~~
{: .output}

So Numba isn't quite competitive with Numpy in this case. But Numba
still has more to give here: notice that we've forced Numpy to only
use a single core. What happens if we use four cores with Numpy?
We'll need to restart the kernel again to get Numpy to pick up the
changed value of `OMP_NUM_THREADS`.

~~~
%env OMP_NUM_THREADS=4
import numpy as np
import math
from numba import vectorize

@vectorize()
def trig(a, b):
    return math.sin(a ** 2) * math.exp(b)

def numpy_trig(a, b):
    return np.sin(a ** 2) * np.exp(b)

a = np.random.random((1000, 1000))
b = np.random.random((1000, 1000))

%timeit numpy_trig(a, b)
%timeit trig(a, b)
~~~
{: .language-bash}

~~~
Numpy: 7.84 ms ± 54.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
Numba: 24.9 ms ± 134 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

Numpy has parallelised this, but isn't icnredibly efficient - it's used 
$$7.84 \times 4 = 31.4$$ core-milliseconds rather than 19. But
Numba can also parallelise. If we alter our call to `vectorize`, we
can pass the keyword argument `target='parallel'`. However, when we do
this, we also need to tell Numba in advance what kind of variables it
will work on&mdash;it can't work this out and also be able to
parallelise. So our `vectorize` decorator becomes:

~~~
@vectorize('float64(float64, float64)', target='parallel')
~~~
{: .language-python}

This tells Numba that the function accepts two variables of type
`float64` (8-byte floats, also known as "double precision"), and
returns a single `float64`. We also need to tell Numba to use as many
threads as we did Numpy; we control this via the `NUMBA_NUM_THREADS`
variable. Restarting the kernel and re-running the timing gives:

~~~
%env NUMBA_NUM_THREADS=4

import numpy as np
import math
from numba import vectorize

@vectorize('float64(float64, float64)', target='parallel')
def trig(a, b):
    return math.sin(a ** 2) * math.exp(b)

a = np.random.random((1000, 1000))
b = np.random.random((1000, 1000))

%timeit trig(a, b)
~~~
{: .language-bash}

~~~
12.3 ms ± 162 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
~~~
{: .output}

In this case this is even less efficient than Numpy's. However, comparing
this against the parallel version running on a single thread tells a different
story. Retrying the above with `NUMBA_NUM_THREADS=1` gives

~~~
47.8 ms ± 962 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
~~~
{: .output}

So in fact, the parallelisation is almost perfectly efficient, just the
parallel implementation is slower than the serial one. If we had more
processor cores available, then using this parallel implementation would
make more sense than Numpy. (If you are running your code on a High-
Performance Computing (HPC) system then this is important!)


> ## Another ufunc
>
> Try creating a ufunc to calculate the discriminant of a quadratic
> equation, $\Delta = b^2 - 4ac$. (For now, make it a serial
> function.)
>
> Compare the timings with using Numpy whole-array operations in
> serial. Do you see the results you might expect?
>
> > ## Solution
> >
> > ~~~
> > @vectorize
> > def discriminant(a, b, c):
> >     return b**2 - 4 * a * c
> > ~~~
> > {: .language-python}
> >
> > Timing this gives me 3.73 microseconds, whereas the `b ** 2 - 4 *
> > a * c` Numpy expression takes 13.4 microseconds&mdash;almost four
> > times as long. This is because each of the Numpy arithmetic
> > operations needs to create a temporary array to hold the results,
> > whereas the Numba ufunc can create a single final array, and
> > use smaller intermediary values.
> {: .solution}
{: .challenge}

> ## Mandelbrot ufunc
>
> Look back at the Mandelbrot example we rewrote using Numpy
> whole-array operations. Try rewriting the `mandelbrot_numpy`
> function to be a ufunc using Numba. How does the performance compare
> to the pure Numpy version?
>
>> ## Solution
>>
>> ~~~
>> import numpy as np
>> from numba import vectorize
>>
>> @vectorize
>> def mandelbrot_ufunc(c, maxiter):
>>     z = c
>>     for n in range(maxiter):
>>         if abs(z) > 2:
>>             return n
>>         z = z*z + c
>>     return 0
>>
>>
>> def mandelbrot_set_ufunc(xmin, xmax, ymin, ymax, width, height, maxiter):
>>     real_range = np.linspace(xmin, xmax, width)
>>     imaginary_range = np.linspace(ymin, ymax, height)
>>     return mandelbrot_ufunc(real_range + 1j * imaginary_range[:,
>>     np.newaxis], maxiter)
>> ~~~
>> {: .language-python}
> {: .solution}
{: .challenge}

> ## Multiple datatypes
>
> Vectorization isn't limited to a single datatype&mdash;you don't have
> to define separate functions for each different data type you want
> to act on. `@vectorize` will accept a list of different signatures.
> The only caveat is that Numba will check them in order, and use the first
> one that matches, so if one data type is a superset of another, it should
> be listed second, otherwise the subset type version will never be used.
> (The effect of this would be, for example, always promoting single-precision
> numbers to double-precision before operating on them, which at best would
> halve the speed.)
{: .callout}
