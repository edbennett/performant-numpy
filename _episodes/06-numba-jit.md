---
title: "Compiling regular functions with Numba"
teaching: 10
exercises: 10
questions:
- "What if I want to speed up non-broadcastable functions?"
objectives:
- "Be able to use Numba's `jit` decorator to just-in-time compile functions"
keypoints:
- "Use the `@jit` decorator to just-in-time compile plain Python functions
(operating on Numpy arrays or otherwise)"
- "Use the `nopython=True` argument to `@jit` to raise an error if
something can't be compiled, so you know to fix it to get maximum speed"
---

While this is starting to diverge from "performant Numpy" towards
"performant Python" in general, it's useful to see how Numba can speed
up things that don't work element-wise at all.

## First example of Numba

(Adapted from the
[5-minute introduction to Numba](https://numba.pydata.org/numba-doc/latest/user/5minguide.html).)

~~~
from numba import jit
import numpy as np

@jit(nopython=True)
def a_plus_tr_tanh_a(a):
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace
~~~
{: .language-python}

Some things to note about this function:

* The decorator `@jit(nopython=True)` tells Numba to compile this code
  in "no Python" mode (i.e. if it can't work out how to compile this
  function entirely to machine code, it should give an error rather than
  partially using Python)
* The function accepts a Numpy array; Numba performs better with Numpy
  arrays than with e.g. Pandas dataframes or objects from  other libraries.
* The array is operated on with Numpy functions (`np.tanh`) and broadcast
  operations (`+`), rather than arbitrary library functions that Numba
  doesn't know about.
* The function contains a plain Python loop; Numba knows how to turn
  this into an efficient compiled loop.

To time this, it's important to run the function once during the
setup step, so that it gets compiled before we start trying to time
its run time.

~~~
a = np.arange(10000).reshape((100, 100))
a_plus_tr_tanh_a(a)
%timeit a_plus_tr_tanh_a(a)
~~~
{: .language-python}

~~~
12.1 µs ± 26.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
~~~
{: .output}

How does this compare with the naive version? Commenting out the
`@jit` decorator in `first_numba.py` an re-running the same timing
command:

~~~
443 µs ± 2.11 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
~~~
{: .output}

This is a 36x speedup&mdash;not too shabby!

It might be possible to rearrange this function so that it uses
pure Numpy operations throughout rather than a regular Python loop,
but in many cases it either isn't possible or significantly reduces
the readability of the code. In these cases, Numba can provide an
alternative route to a comparable level of performance, with a
lot less work, and more readable code at the end of it.

## Getting parallel

The `@jit` decorator accepts a relatively wide range of parameters.
One is `parallel`, which tells Numba to try and optimise the function
to run with multiple threads. Like previously, we need to control this
threads count at run-time using the `NUMBA_NUM_THREADS`
environment variable.

Restarting the kernel and setting this variable:

~~~
%env NUMBA_NUM_THREADS=4
from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def a_plus_tr_tanh_a(a):
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


a = np.arange(10000).reshape((100, 100))
a_plus_tr_tanh_a(a)
%timeit a_plus_tr_tanh_a(a)
~~~
{: .language-python}

~~~
20.9 µs ± 242 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
~~~
{: .output}

Parallelism has successfully multiplied our run time by 4. This is
because when running in parallel, Numba (in fact, the OpenMP runtime)
needs to spin up a team of threads to run the code, and then keep
them synchronised. This takes a finite amount of time, so on very
small functions like the one we've run here it takes longer than the
time saved by running in parallel.

> ## Larger problem size
>
> Retry the example above with a matrix size of $$1000 \times 1000$
> instead of $100 \times 100$, and see how the parallel and serial
> performance compare.
{: .challenge}


## Programming GPUs

We don't have time to look at this in detail, but an example of how
GPUs can be programmed with Numba:

~~~
from numba import vectorize

@vectorize(['int64(int64, int64)'], target='cuda')
def add_ufunc(x, y):
    return x + y
~~~
{: .language-python}

More information on programming your GPU with Numba can be found at
[this tutorial](https://github.com/ContinuumIO/gtc2018-numba) given at
the GPU Technology Conference 2018.
