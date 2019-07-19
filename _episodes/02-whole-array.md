---
title: "Whole-array operations"
teaching: 30
exercises: 20
questions:
- "How can I avoid looping when performing operations across arrays in Numpy?"
- "What about when there are conditionals?"
- "What about when there are complex array multiplications?"
objectives:
- "Understand the arithmetic operations available across whole arrays
in Numpy"
- "Be able to use masks and `einsum` to broaden the range of
whole-array operations available"
keypoints:
- "Numpy will broadcast operations to all elements, and has many
functions to implement commonly-performed tasks across arrays"
- "Conditionals can be recast as array masks, arrays of true/false
values"
- "Use `numpy.einsum` for more complex array multiplications and
reductions, but only where the performance boost is worth the reduced
readability"
---

## One-dimensional arrays

Let's start off with an example with one-dimensional arrays,
calculating the Euclidean
distance between two vectors. We'd like to convert the following
function to use Numpy in the most performant way possible:

~~~
def naive_dist(p, q):
    square_distance = 0
    for p_i, q_i in zip(p, q):
        square_distance += (p_i - q_i) ** 2
    return square_distance ** 0.5
~~~
{: .language-python}

Let's test this to get a baseline performance measure:

~~~
p = [i for i in range(1000)]
q = [i + 2 for i in range(1000)]

%timeit naive_dist(p, q)
~~~
{: .language-python}

On my machine, this gives:

~~~
485 µs ± 365 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
~~~
{: .output}

Now, we know that Numpy can do the subtraction and square operations
element-wise all at once, rather than requiring an explicit loop. Let's
try that:

~~~
import numpy as np

def simple_numpy_dist(p, q):
    return (np.sum((p - q) ** 2)) ** 0.5
~~~
{: .language-python}

To test this, we'll need to use a Numpy array rather than the lists we
made for the previous test, since Python can't subtract lists.

~~~
p = np.arange(1000)
q = np.arange(1000) + 2

%timeit simple_numpy_dist(p, q)
~~~
{: .language-python}

~~~
26.7 µs ± 275 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
~~~
{: .output}

We're already 18 times better! (This will vary from machine-to-machine;
the wider your CPU's vector units, the better this will do.) But can we
do better?

In general, it's best to use the most specific operation that's available.
This is because the more specialised the function, the more of it is
implemented in compiled C, and the less time is spent in the Python
glue logic.
Here we are calculating the difference between the two vectors, and then
finding the length of the result. Finding the length of a vector is
something you'd imagine would be provided by any good numerical library,
and Numpy is no exception.

~~~
def numpy_norm_dist(p, q):
    return np.linalg.norm(p - q)

%timeit numpy_norm_dist(p, q)
~~~
{: .language-python}

~~~
18.4 µs ± 594 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
~~~
{: .language-python}

Using the specialised function here has given a 31% improvement over
the more generic operations.


## Higher dimensionalities

For one-dimensional arrays, translating from naive to whole-array
operations is normally quite direct. But when it comes to
multi-dimensional arrays, some additional work may be needed to get
everything into the right shape.

Let's extend the previous example to work on multiple vectors at
once. We would like to calculate the Euclidean distances between $M$
pairs of vectors, each of length $N$. In plain Python we could take
this as a list of lists, and re-use the previous function for each
vector in turn.

~~~
def naive_dists(ps, qs):
    return [naive_dist(p, q) for p, q in zip(ps, qs)]
~~~
{: .language-python}

We'll need to generate some multi-dimensional test data here:

~~~
ps = [[i + 1000 * j for i in range(1000)] for j in range(1000)]
qs = [[i + 1000 * j + 2 for i in range(1000)] for j in range(1000)]

%timeit naive_dists(ps, qs)
~~~
{: .language-python}

~~~
497 ms ± 5.14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

Half a second per iteration is still not that long in the grand scheme
of things, but if this were in an inner loop of a larger program, it
could slow things down quite quickly.

Moving this to Numpy, we can subtract as we did previously, but for
the summation, we need to be able to tell Numpy to leave us
with a one-dimensional array of distances, rather than a single
number. To do this, we pass the `axis` keyword argument, which tells
Numpy which axis to sum over. In this case, axis 0 controls which
vector we are selecting, and axis 1 controls which element of the
vector. Thus here we only want to sum over axis 1, leaving axis 0
still representing the vector of sums.

~~~
def simple_numpy_dists(ps, qs):
    return np.sum((ps - qs) ** 2, axis=1) ** 0.5
~~~
{: .language-python}

`axis` is a keyword argument that is quite important when working
with higher-dimensional arrays, so it's worth taking some time to
make sure that the values make sense.

To test the performance of this, we need to generate the same sample
data we previously used. To do this, we'll use Numpy's `reshape`
function, to turn a long one-dimensional array into a multidimensional
one. This is something you will see used very frequently for quickly
creating sample data. It splits the array into equal-length slices
and stacks them, working left-to-right, top-to-bottom; for the general
higher-dimensional case, the indices are filled left-to-right.

![An illustration showing a long rainbow-coloured array being divided into four blocks, with the four blocks stacked top-to-bottom.](../fig/rainbow-reshape.svg)

~~~
ps = np.arange(1000000).reshape((1000, 1000))
qs = np.arange(1000000).reshape((1000, 1000)) + 2

%timeit simple_numpy_dists(ps, qs)
~~~
{: .language-python}

~~~
6.7 ms ± 384 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
~~~
{: .output}

Once again we've achieved almost a 100x speedup by switching over to
Numpy. What about the `norm` function we tried previously? This also
supports the `axis` keyword argument.

~~~
def numpy_norm_dists(ps, qs):
    return np.linalg.norm(ps - qs, axis=1)

%timeit numpy_norm_dists(ps, qs)
~~~
{: .language-python}

Timing this gives:

~~~
10.3 ms ± 318 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
~~~
{: .output}

Unlike the 1-dimensional case, using the dedicated `norm` function
provided by Numpy is slower here than the explicit computation! As you
can see, it's always important to test your improvements on data that
resemble those that you will be performing your computation on, as the
performance characteristics will change from case to case.

Numpy does have one more trick up its sleeve, however...

### Einsum

The problem with these more complex cases is that we're having to use
optional arguments to `np.sum` and `np.linalg.norm`. By necessity,
Numpy can't optimise all cases of these general functions as well as
it would be able to optimise the specific case that we are interested
in.

Numpy does give us a way to express an exact reduction that we would
like to perform, and can execute it in a highly optimised way. The
function that does this is `np.einsum`, which is short for "Einstein
summation convention". If you're not familiar with this, it is a
notation used in physics for abbreviating common expressions that have
many summations in them.

`np.einsum` requires as arguments a string specifying the operations
to be carried out, and then the array(s) that the operations will be
carried out on. The string is formatted in a specific way: indices to
look up (starting from `i`, `j`, `k`, ...) are given for each array,
separated by commas, followed by `->` (representing an arrow), and
then the indices for the resulting array are given. Any indices not on
the right-hand side are summed over. So the operation:

~~~
C = np.einsum('ijk,jkl->ijl', A, B)
~~~
{: .language-python}

is equivalent to the matrix (or tensor) operation:

$$C_{ijl} = \sum_{k} A_{ijk} B_{jkl}$$

For example, `'ii->i'` gives a one-dimensional array (or a vector)
containing the diagonal elements of a two-dimensional array (or
matrix). This is because the $i$th element of this vector contains the
element $(i, i)$ of the matrix.

In this case, we want to calculate the array's product with itself,
summing along axis 1. This can be done via `'ij,ij->i'`, although the
array will need to be given twice or this to work. Because of this,
we'll need to put the difference array into a variable before passing
it into `np.einsum`. Putting this together:

~~~
def einsum_dists(ps, qs):
    difference = ps - qs
    return np.einsum('ij,ij->i', difference, difference) ** 0.5
~~~
{: .language-python}

Timing this:

~~~
1000 loops, best of 3: 1.7 msec per loop
~~~
{: .output}

Wow! At the expense of some readability, we've gained another factor
of 2 in performance. Since the resulting line of code is significantly
harder to read, and any errors in the `np.einsum` syntax are likely to
be impenetrable, it is important to leave a comment explaining exactly
what you are trying to do with this syntax when you use it. That way,
if someone (yourself included) comes back to it in six months, they
will have a better idea of what it does, and can double-check that it
does indeed do this. And if there is a bug, you can check whether the
bug is in the design (was the line trying to do the wrong thing) or in
the implementation (was it trying to do the right thing, but the
`np.einsum` syntax for that thing was implemented wrong).

`np.einsum` is best used sparingly, at the most performance-critical
parts of your program. Used well, though, it can be a game-changer,
since it can express many operations that would be difficult to
express any other way without using explicit `for` loops.


## Avoiding conditionals

Sometimes we would like to diverge between two code paths based on
some condition. Typically we would do this with an `if` block. However,
to do this on an element-by-element basis requires introducing a loop.
Since Python loops are the enemy of performance when working with Numpy,
we would like to find an alternative to loops + `if` blocks that still
let us distinguish between different cases.

What we can do in these cases is to use masks. Masks take various forms,
but in general they are arrays of boolean (true/false) values that can
be used to identify which data to consider and which to discard from
a calculation.

For example, consider the peaked data generated by:

~~~
import matplotlib.pyplot as plt

x_range = np.arange(-30, 30.1, 0.1)
y_range = np.arange(-30, 30.1, 0.1)
x_values, y_values = np.meshgrid(x_range, y_range, sparse=False)
peaked_function = (np.sin(x_values**2 + y_values**2) /
                   (x_values**2 + y_values**2) ** 0.25)
plt.imshow(peaked_function)
plt.show()
~~~
{: .language-python}

If we are only interested in the highest points, we can isolate these
using the `>` sign as usual. This gives us back an array of boolean
values that is the same shape as the input array:

~~~
peaked_function > 0.8
~~~
{: .language-python}

~~~
array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])
~~~
{: .output}

Plotting this, we can see that the highest values are a ring around
the centre of the distribution:

~~~
plt.imshow(peaked_function > 0.8)
plt.show()
~~~
{: .language-python}

This array is an example of a mask. We can use this as its own array;
for example, if we want to know how many points of the plot are larger
than 0.8, then we can use `np.sum(peaked_function > 0.8)`.

Numpy also gives three ways of using this
to perform calculations with the original array.

### Multiplication

By multiplying a mask array with the input array, we suppress the unwanted
elements of the original array, without changing its shape. For example,
removing only the negative elements of the original array:

~~~
mask = peaked_function >= 0
plt.imshow(mask * peaked_function)
plt.show()
~~~
{: .language-python}


### Fancy Indexing

Numpy also lets you use a mask array as an index. This will pick out only
the array elements selected by the mask, discarding the unwanted ones.
This gives a one-dimensional array rather than preserving the shape.

For example:

~~~
mask = peaked_function > 0.9
print('Values of peaked_function greater than 0.9:')
print(peaked_function[mask])
~~~
{: .language-python}


### Masked arrays

Numpy also allows arrays with a built-in mask to be constructed, so that
all function calls on these arrays ignore unwanted values (while still
leaving the ignored data intact).

~~~
mask = peaked_function > 0.9
masked_peaks = np.ma.masked_array(peaked_function, mask=~mask)
np.mean(masked_peaks)
~~~
{: .language-python}

~~~
0.903742378484697
~~~
{: .output}

Note the `~mask`! This is because the `mask` keyword expects the
opposite convention to the one we have been using: `True` values are
dropped, and `False` values are included. `~` is the logical NOT
operator.

> ## Performance comparison
>
> The mean that we calculated using a masked arrays could equally
> well have been calcualted using multiplication or fancy indexing.
> Implement these two methods, and compare their performance.
>
>> ## Solution
>>
>> With fancy indexing:
>>
>> ~~~
>> %%timeit
>> mask = peaked_function > 0.9
>> peaked_function[mask].mean()
>> ~~~
>> {: .language-python}
>>
>> ~~~
>> 343 µs ± 8.97 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
>> ~~~
>> {: .output}
>>
>> With multiplication:
>>
>> ~~~
>> %%timeit
>> masked_peaks = np.ma.masked_array(peaked_function, mask=~mask)
>> np.mean(masked_peaks)
>> ~~~
>> {: .language-python}
>>
>> ~~~
>> 2.59 ms ± 22.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
>> ~~~
>> {: .output}
>>
>> With masks:
>>
>> ~~~
>> %%timeit
>> mask = peaked_function > 0.9
>> np.sum(mask * peaked_function) / np.sum(mask)
>> ~~~
>> {: .language-python}
>>
>> ~~~
>> 1.71 ms ± 7.04 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
>> ~~~
>> {: .output}
> {: .solution}
{: .challenge}

> ## Calculating Pi
>
> A common example problem in numerical computing is the Monte Carlo
> computation of $\pi$. The way this works is as follows.
>
> Consider the
> unit circle, centered on the origin. Now, imagine firing bullets at
> random locations in the square from the origin to the point (1,
> 1). The area of that square that overlaps with the circle is
> $\frac{\pi}{4}$, while the area of the square is 1. So the
> proportion of bullets that will land in the circle is
> $\frac{\pi}{4}$.
>
> So, by generating a large number of random points in the unit
> square, and counting the number that lie within the unit circle, we
> can find an estimate for $\frac{\pi}{4}$, and by extension, $\pi$.
>
> ![A diagram illustrating the description above.](../fig/pi_dartboard.svg)
>
> A plain Python function that would achieve this might look as
> follows:
>
> ~~~
> from random import random
>
> def naive_pi(number_of_samples):
>     within_circle_count = 0
>
>     for _ in range(number_of_samples):
>         x = random()
>         y = random()
>
>         if x ** 2 + y ** 2 < 1:
>             within_circle_count += 1
>
>     return within_circle_count / number_of_samples * 4
> ~~~
> {: .language-python}
>
> Time this for a million iterations. Check that it gives the correct
> result for $\pi$.
>
> Try and convert this to Numpy. Think about how you can utilise
> whole-array operations. Time your result and see how it compares to
> the plain Python case. How much faster can you get?
>
> You'll want to use `np.random.random` for your random numbers. Take
> a look at the documentation for that function to see what arguments
> it takes that will be helpful for this problem.
>
>> ## Solution
>>
>> Two possible solutions:
>>
>> ~~~
>> import numpy as np
>> 
>> def numpy_pi_1(number_of_samples):
>>     # Generate all of the random numbers at once to avoid loops
>>     samples = np.random.random(size=(number_of_samples, 2))
>>
>>     # Use the same np.einsum trick that we used in the previous example
>>     # Since we are comparing with 1, we don't need the square root
>>     squared_distances = np.einsum('ij,ij->i', samples, samples)
>> 
>>     # Identify all instances of a distance below 1
>>     # "Sum" the true elements to count them
>>     within_circle_count = np.sum(squared_distances < 1)
>> 
>>     return within_circle_count / number_of_samples * 4
>> 
>> 
>> def numpy_pi_2(number_of_samples):
>>     within_circle_count = 0
>> 
>>     xs = np.random.random(size=number_of_samples)
>>     ys = np.random.random(size=number_of_samples)
>> 
>>     r_squareds = xs ** 2 + ys ** 2
>> 
>>     within_circle_count = np.sum(r_squareds < 1)
>> 
>>     return within_circle_count / number_of_samples * 4
>> ~~~
>> {: .language-python}
>>
>> While these are competitive with each other in performance, which
>> is the fastest depends on various factors. As always, test your
>> own specific workloads and hardware setup to see how solutions
>> perform on them.
> {: .solution}
{: .challenge}


> ## Multiple cores?
>
> Many functions in Numpy will try to take advantage of multi-core
> parallelism in your machine. While this is better than ignoring
> parallelism completely, it isn't perfect. In many cases there is no
> speedup from the parallel implementation; and many functions aren't
> parallelised at all.
>
> Something to beware is that if you try and take advantage of
> parallelism yourself via, for example, GNU Parallel, then Numpy
> might not be aware of this, and will try and use every CPU core you
> have available&mdash;for every copy of the program that is
> executing. While the parallelism may not have sped things up much,
> this will severely slow down all of your programs, since the
> processor is spending most of its time swapping between different
> threads rather than doing actual work.
>
> If you do want multi-core parallelism out of your Numpy code, it's
> better to do it more explicitly as we'll explore in a later section.
>
> To disable Numpy's parallelism, you need to set the environment
> variable `OMP_NUM_THREADS` to 1 before launching your program.
> Within a Jupyter notebook, this can be done by placing the following
> line magic at the top of your notebook:
>
> ~~~
> %env OMP_NUM_THREADS=1
> ~~~
> {: .language-python}
{: .callout}


