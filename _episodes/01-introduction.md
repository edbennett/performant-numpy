---
title: "Introduction"
teaching: 15
exercises: 10
questions:
- "How does Numpy give faster performance than Python?"
- "What routes are there to get better performance out of Numpy?"
objectives:
- "Understand the difference between interpreted and compiled
languages"
- "Understand the available options in increasing Numpy performance"
keypoints:
- "Numpy comprises primarily compiled, explicitly-typed code, which
allows optimisations that can't be done for Python's interpreted,
implicitly typed code"
- "Avoid Python loops and using whole-array functions to get better
performance"
- "Using compiled code will frequently improve performance over
Numpy. This can be done in a compiled language, or by using Numba on
Python code."
---

## Characteristics of Python

Python is an incredibly popular, powerful, and easy-to-use programming
language. Unfortunately, its popularity and ease-of-use are in many
parts due to design trade-offs that it has made that reduce its
performance, or make it more difficult to make go faster.

### Compiled and interpreted languages

Programming languages are frequently divided into two categories:
compiled languages, and interpreted languages. Interpreted languages,
such as Python, supply a program called an "interpreter", which reads
input files line-by-line and translates them into instructions that
the computer knows how to execute. Once a line has been executed, the
interpreter then finds the next line, and translates that.

Compiled languages, on the other hand, supply a program called a
"compiler", which translates the entire program into machine code at
once, and then the program can be run directly with no further
intervention. This removes a significant amount of overhead from
running the program, since translating the program to machine code has
already been done. C, C++, and Fortran are examples of compiled
languages.


### Explicit and implicit typing

In some languages, like C, all variables must be told what data type
they hold. For example, some might hold integers or floating-point
numbers, and others may hold arrays of these. Similarly, functions
must declare what types they expect to be given, and what types they
will return; for example, the exponentiation function might take two
floating-point numbers and return floating-point number.

In Python, on the other hand, this is not necessary. Variables
automatically pick up their type when they are created, and functions
will operate on any type of data they are given (at least until they
encounter some behaviour that doesn't work, at which point they will
generate an error). (This is called "duck typing"&mdash;if a variable
looks like a duck and quacks like a duck, and those are the only
requirements that the function places on its arguments, then as far as
that function is concerned then the variable is a duck, even if on
closer inspection it turns out to be a cardboard cutout of a duck
with a duck impressionist hiding behind it!)

This has two consequences. Firstly, every value needs to carry around
with it information about what type it is. In an explicitly typed
language, this could be attached to the name, instead, significantly
reducing the volume of data stored in a particular variable. Secondly,
functions&mdash;even very small internal utility functions&mdash;need
to be able to operate on any kind of data thrown at them. For example,
consider the function:

~~~
def add_two_numbers(x, y):
    return x + y
~~~
{: .language-python}

If `x` and `y` are both integers, or both floating-point numbers,
then this function becomes a single processor instruction. However,
in Python, the function must first check the labels of `x` and `y`
to find out what types they are, then look to see whether it knows
how to add those two types, then unpack the numbers stored inside
and pass them to the CPU to add together. This overhead is easily
tens if not hundreds of times the time it would take to do the
single instruction in an explicitly typed, compiled language.

As such, one aim when trying to make Python go faster is to have as
little code as possbile executed by the standard Python interpreter,
and instead find other ways to execute things that will have less
overhead.


### The Global Interpreter Lock

A common way to gain speed in many languages is to use multiple "threads"
to allow a program to operate in parallel. However, this comes with some
dangers, since different threads can interfere with each others' operation.
In an interpreted language like Python, this is especially complicated,
since entire functions could change between when the program decides to run
it and when it is actually run. To avoid these kinds of risks, Python
(more specifically, the most common Python interpreter, CPython) uses
a "Global Interpreter Lock", to ensure that only one thread is allowed to
be actively running at a time. This means that in most computational
softwaer, there is no performance increase to be gained from using multiple
threads within Python. Instead, other methods need to be used in order
to run in parallel.


## Timing

Since we are focusing on performance today, we need to be able to
judge what is and isn't performant. While a full treatment of
profiling is out of the scope of this lesson, we can get a reasonable
picture of the performance of individual functions and code snippets
by using the `timeit` module.

In Jupyter, `timeit` is provided by line and cell magics. The line
magic:

~~~
%timeit result = some_function(argument1, argument2)
~~~
{: .language-python}

will report the time taken to perform the operation on the same line
as the `%timeit` magic. Meanwhile, the cell magic

~~~
%%timeit

intermediate_data = some_function(argument1, argument2)
final_result = some_other_function(intermediate_data, argument3)
~~~
{: .language-python}

will measure and report timing for the entire cell.

Since timings are rarely perfectly reproducible, `timeit` runs the
command multiple times, calculates an average timing per iteration,
and then repeats to get a best-of-*N* measurement. The longer the
function takes, the smaller the relative uncertainty is deemed to be,
and so the fewer repeats are performed. `timeit` will tell you how
many times it ran the function, in addition to the timings.

While today we'll work at the Jupyter notebook, you can also use
`timeit` at the command-line; for example,

~~~
$ python -m timeit --setup='import numpy; x = numpy.arange(1000)' 'x ** 2'
~~~
{: .language-bash}

Notice the `--setup` argument, since you don't usually want to time
how long it takes to import a library, only the operations that you're
going to be doing a lot.


## Characteristics of Numpy

While Numpy is a Python library, it is primarily written in C, which
is both explicitly typed and compiled. While it is written to be
somewhat less flexible than pure Python (for example, it can only act
on a limited range of data types), this has the effect of making it
significantly more efficient, since the compiler can pack arrays
efficiently and optimise operations across multiple array elements to
make use of processor features like vector instructions.


### Avoid Python loops

This performance does depend on you using Numpy in an efficient
way. Every call to a Numpy function starts life as a Python function
call, and so carries an overhead for that. There is then some overhead
of converting between Python and Numpy data types, and the time spent
executing the actual underlying instructions. If you operate across
Numpy arrays with a Python loop, you're likely to remove the option
for the operations to use vector instructions, and also introduce all
of these (large) overheads on what is a very small operation. In the
worst case, this will end up slower than using pure Python operations!

For example, after doing some setup:

~~~
import numpy as np

A = np.random.random(10_000)
B = np.random.random(10_000)
C = np.empty_like(A)
~~~
{: .language-python}

then the loop:

~~~
for i in range(10_000):
    C[i] = A[i] + B[i]
~~~
{: .language-python}

will be significantly slower than the whole-array operation:

~~~
C = A + B
~~~
{: .language-python}


> ## Test our assumptions
>
> While it is easy to make assertions about what we think should be
> faster or slower, we must always test these assumptions and use hard
> data to decide what to do.
>
> Use `timeit` to test the performance of the two code blocks
> above. How does their performance compare on your machine?
{: .challenge}

While this is a relatively trivial example, in the next sections we
will explore how to do some more advanced whole-array operations.


### Compile

When Numpy doesn't provide whole-array operations that do what we need
them to, then we need to think about writing our own. Since these need
to be compiled to be performant, you might think that we would need to
switch to a compiled language like C in order to write these. In fact,
the community has developed a method of compiling Python code directly
to machine code, via a library called Numba. In the later sections of
this lesson, we will look at how to use Numba to write universal
functions in Numpy.


{% include links.md %}

