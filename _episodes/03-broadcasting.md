---
title: "Broadcasting"
teaching: 20
exercises: 10
questions:
- "How can I do whole-array operations with arrays of different shape?"
objectives:
- "Understand the rules around broadcasting arrays of different sizes"
keypoints:
- "Numpy automatically expands smaller arrays to match the shape of larger
ones"
- "Axes are read right to left, and must be either the same size or size 1"
- "Where one array has more dimensions, the smaller array is interpreted
as having size 1 on the additional axes"
---


We know that when you add two arrays of the same size in Numpy that
they are added element-wise:

~~~
import numpy as np

A = np.arange(100).reshape((10, 10))
B = np.arange(10, 20, 0.1).reshape((10, 10))

A + B
~~~
{: .language-python}

~~~
array([[ 10. ,  11.1,  12.2,  13.3,  14.4,  15.5,  16.6,  17.7,  18.8, 19.9],
       [ 21. ,  22.1,  23.2,  24.3,  25.4,  26.5,  27.6,  28.7,  29.8, 30.9],
       [ 32. ,  33.1,  34.2,  35.3,  36.4,  37.5,  38.6,  39.7,  40.8, 41.9],
       [ 43. ,  44.1,  45.2,  46.3,  47.4,  48.5,  49.6,  50.7,  51.8, 52.9],
       [ 54. ,  55.1,  56.2,  57.3,  58.4,  59.5,  60.6,  61.7,  62.8, 63.9],
       [ 65. ,  66.1,  67.2,  68.3,  69.4,  70.5,  71.6,  72.7,  73.8, 74.9],
       [ 76. ,  77.1,  78.2,  79.3,  80.4,  81.5,  82.6,  83.7,  84.8, 85.9],
       [ 87. ,  88.1,  89.2,  90.3,  91.4,  92.5,  93.6,  94.7,  95.8, 96.9],
       [ 98. ,  99.1, 100.2, 101.3, 102.4, 103.5, 104.6, 105.7, 106.8, 107.9],
       [109. , 110.1, 111.2, 112.3, 113.4, 114.5, 115.6, 116.7, 117.8, 118.9]])
~~~
{: .output}


What happens when we want to operate using something of a different size?
We know that we can use a scalar number:

~~~
B + 100
~~~
{: .language-python}

~~~
array([[110. , 110.1, 110.2, 110.3, 110.4, 110.5, 110.6, 110.7, 110.8, 110.9],
       [111. , 111.1, 111.2, 111.3, 111.4, 111.5, 111.6, 111.7, 111.8, 111.9],
       [112. , 112.1, 112.2, 112.3, 112.4, 112.5, 112.6, 112.7, 112.8, 112.9],
       [113. , 113.1, 113.2, 113.3, 113.4, 113.5, 113.6, 113.7, 113.8, 113.9],
       [114. , 114.1, 114.2, 114.3, 114.4, 114.5, 114.6, 114.7, 114.8, 114.9],
       [115. , 115.1, 115.2, 115.3, 115.4, 115.5, 115.6, 115.7, 115.8, 115.9],
       [116. , 116.1, 116.2, 116.3, 116.4, 116.5, 116.6, 116.7, 116.8, 116.9],
       [117. , 117.1, 117.2, 117.3, 117.4, 117.5, 117.6, 117.7, 117.8, 117.9],
       [118. , 118.1, 118.2, 118.3, 118.4, 118.5, 118.6, 118.7, 118.8, 118.9],
       [119. , 119.1, 119.2, 119.3, 119.4, 119.5, 119.6, 119.7, 119.8, 119.9]])
~~~
{: .output}

In fact, this is a special case of a more general Numpy feature called
*broadcasting*. When you try to operate on two arrays of different shapes,
Numpy will try to duplicate the array along any missing dimensions in order
to make the shapes match. For example:

~~~
values = np.arange(16).reshape((4, 4))
column_weights = np.arange(0, 400, 100)

values * column_weights
~~~
{: .language-python}

~~~
array([[   0,  100,  400,  900],
       [   0,  500, 1200, 2100],
       [   0,  900, 2000, 3300],
       [   0, 1300, 2800, 4500]])
~~~
{: .output}

The `column_weights` array is expanded out to apply to every row. (In
reality, no copying of data happens; Numpy efficiently implements the
memory accesses so that it looks as if the copy occurred.)

What about if we wanted to weight the rows rather than the columns?

~~~
row_weights = np.arange(0, 40, 10).reshape((4, 1))
values * row_weights
~~~
{: .language-python}

~~~
array([[  0,   0,   0,   0],
       [ 40,  50,  60,  70],
       [160, 180, 200, 220],
       [360, 390, 420, 450]])
~~~
{: .output}

We can now start to see a pattern emerging. Looking at the shapes of these
arrays:

~~~
print("values:", values.shape)
print("column_weights:", column_weights.shape)
print("row_weights:", row_weights.shape)
~~~
{: .langauge-python}

~~~
values: (4, 4)
column_weights: (4,)
row_weights: (4, 1)
~~~
{: .output}

Recall that the right-most index of an `ndarray` is the most local, i.e.
for a 2D array it will be the index that scans across a row, while
the second index from the right moves down a column. (Alternatively,
the rightmost index controls which column is being considered, while the
next controls which row is being considered.)

So, schematically, the first multiplication has the form:

~~~
column_weights: [  0, 100, 200, 300]
                   *    *    *    *
arrayvalues:    [[ 0,   1,   2,   3],
                 [ 4,   5,   6,   7],
                 [ 8,   9,  10,  11],
                 [12,  13,  14,  15]]
~~~

while the second has the form:

~~~
row_weights             values
  [[  0 ]    *  [[ 0,   1,   2,   3],
   [ 10 ]    *   [ 4,   5,   6,   7],
   [ 20 ]    *   [ 8,   9,  10,  11],
   [ 30 ]]   *   [12,  13,  14,  15]]
~~~


To go into more detail: indices are matched up from right-to-left, and can
only match if they are equal to each other, or to 1. If the indices don't
match, then an error is raised. (Numpy *doesn't* try to match other indices!)
Any additional indices (if one array has a higher dimension than the other)
are taken to be 1.

In the first example above, the 4 of the rightmost (column) index of `values` 
matches the 4 of the only index of `column_weights`; this array is then
broadcast across to match every element of the row index of `values`. In
the second example, the 4 of the column index of `values` matches up with
the 1 of the second index of `row_weights`, and so this is broadcast to
match every column; the second 4 from the row index matches the 4 from the
row index of `row_weights`.

Looking now at a rectangular array:

~~~
rectangular_values = np.arange(6).reshape((3, 2))
two_vector = np.asarray([1, 10])
rectangular_values * two_vector
~~~
{: .language-python}

~~~
array([[ 0, 10],
       [ 2, 30],
       [ 4, 50]])
~~~
{: .output}

In this case, the rightmost index of `rectangular_values` matches
the size of the `two_vector` array, and so broadcasting is done over the
leftmost row index of `rectangular_values`. Conversely

~~~
three_vector = np.asarray([1, 10, 100])
rectangular_values * three_vector
~~~
{: .language-python}

raises an error:

~~~
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-14-4ce8501f6c65> in <module>
      1 three_vector = np.asarray([1, 10, 100])
----> 2 rectangular_values * three_vector

ValueError: operands could not be broadcast together with shapes (3,2) (3,) 
~~~
{: .output}

The rightmost index of `rectangular_values` (2) doesn't equal that of
`three_vector` (3), and neither is 1, so the broadcasting fails. Numpy 
doesn't try matching with the leftmost index of `rectangular_values`!

If we wanted Numpy to broadcast to this element, we would instead need 
to reshape so that the indices were correct:

~~~
three_vector_column = np.asarray([1, 10, 100]).reshape((3, 1))
rectangular_values * three_vector_column
~~~
{: .language-python}

~~~
array([[  0,   1],
       [ 20,  30],
       [400, 500]])
~~~
{: .output}

This generalises readily to more than two dimensions. Work from right to 
left index by index, and the indices must either be equal to each other,
or one must be 1.

~~~
values_4d = np.arange(120).reshape((2, 3, 4, 5))
values_4d * row_weights
~~~
{: .language-python}

~~~
array([[[[   0,    0,    0,    0,    0],
         [  50,   60,   70,   80,   90],
         [ 200,  220,  240,  260,  280],
         [ 450,  480,  510,  540,  570]],

        [[   0,    0,    0,    0,    0],
         [ 250,  260,  270,  280,  290],
         [ 600,  620,  640,  660,  680],
         [1050, 1080, 1110, 1140, 1170]],

        [[   0,    0,    0,    0,    0],
         [ 450,  460,  470,  480,  490],
         [1000, 1020, 1040, 1060, 1080],
         [1650, 1680, 1710, 1740, 1770]]],


       [[[   0,    0,    0,    0,    0],
         [ 650,  660,  670,  680,  690],
         [1400, 1420, 1440, 1460, 1480],
         [2250, 2280, 2310, 2340, 2370]],

        [[   0,    0,    0,    0,    0],
         [ 850,  860,  870,  880,  890],
         [1800, 1820, 1840, 1860, 1880],
         [2850, 2880, 2910, 2940, 2970]],

        [[   0,    0,    0,    0,    0],
         [1050, 1060, 1070, 1080, 1090],
         [2200, 2220, 2240, 2260, 2280],
         [3450, 3480, 3510, 3540, 3570]]]])
~~~
{: .output}

If we want to work on different axes, then as an alternative to 
`reshape`, we can also use `expand_dims`. For example, to use 
a $$2 \times 3$$ array to work on the leftmost two columns of
`values_4d`:

~~~
matrix_weights = np.expand_dims(np.expand_dims(
    [[0, 2, 0], [1, 0, 3]], axis=2), axis=3
)
values_4d * matrix_weights
~~~
{: .language-python}

~~~
array([[[[  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0]],

        [[ 40,  42,  44,  46,  48],
         [ 50,  52,  54,  56,  58],
         [ 60,  62,  64,  66,  68],
         [ 70,  72,  74,  76,  78]],

        [[  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0]]],


       [[[ 60,  61,  62,  63,  64],
         [ 65,  66,  67,  68,  69],
         [ 70,  71,  72,  73,  74],
         [ 75,  76,  77,  78,  79]],

        [[  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0]],

        [[300, 303, 306, 309, 312],
         [315, 318, 321, 324, 327],
         [330, 333, 336, 339, 342],
         [345, 348, 351, 354, 357]]]])
~~~
{: .output}

Another way to do this is to slice the array, and use `np.newaxis`:

~~~
matrix_weights = np.asarray(
    [[0, 2, 0], [1, 0, 3]]
)[:, :, np.newaxis, np.newaxis]
values_4d * matrix_weights
~~~
{: .language-python}

which gives the same output as the previous version.

Finally, sometimes you will need to rearrange the order of your axes to
let broadcasting work. This can be done with `swapaxes`; for example
to operate on the first two indices of `values_4d` with the
`rectangular_values` array we created earlier:

~~~
matrix_weights = np.expand_dims(np.expand_dims(
    rectangular_values, axis=2), axis=3
).swapaxes(0, 1)
values_4d * matrix_weights
~~~
{: .language-python}

> ## Array broadcasts for image manipulation
>
> Images can be represented in Numpy as three-dimensional arrays:
> the first two axes are the vertical and horizontal pixel co-ordinates,
> and the third axis gives the colour channels (red, green, blue, and
> optionally transparency, also known as alpha).
>
> The `imageio` library, included with Anaconda, provides the `imread`
> function which reads an image into an array with this format.
> (This functionality was previously provided by `scipy.misc`.)
>
> Read in an image from your hard drive with the following code
>
> ~~~
> import imageio
> from matplotlib import pyplot as plt
> image = imageio.imread('cat.jpg') / 256
> plt.imshow(image)
> plt.show()
> ~~~
> {: .language-python}
>
> (Replace `cat.jpg` with the name of your image!)
>
> Use broadcasts to apply the
> following transformations:
>
> 1. Suppress the blue channel to zero, and reduce the intensity of the
>    green channel by half.
> 2. Make the image fade to black from left to right. (I.e. all colour
>    channels are multiplied by 1 at the left edge, zero at the right
>    edge, and a number between 0 and 1 in between.)
> 3. Make the image fade to whide from top to bottom.
> 4. Use a variant of the `peaked_function` in the previous episode
>    to multiply all colour channels of the image.
> 5. Do the same, but applying it only to the red channel.
>
>> ## Solution
>>
>> 1. Here we vary with the last index, and broadcast to fill the rest:
>>
>>    ~~~
>>    plt.imshow(image * [1, 0.5, 0])
>>    plt.show()
>>    ~~~
>>    {: .language-python}
>>
>>    (You may need to add an extra 1 at the end of the list if your image
>>    has an alpha channel.)
>>
>> 2. Here we are varying the second-to-last index, so need to add an extra
>     axis
>>
>>    ~~~
>>    image2 = image * np.linspace(1, 0, image.shape[1])[:, np.newaxis]
>>    plt.imshow(image2)
>>    plt.show()
>>    ~~~
>>    {:.language-python}
>>
>> 3. To fade to white, we need to multiply the difference from 1 rather
>>    than from zero, and now we are using the leftmost index:
>>
>>    ~~~
>>    image3 = 1 - (1 - image) * np.linspace(
>>        1, 0, image.shape[0]
>>    )[:, np.newaxis, np.newaxis]
>>    plt.imshow(image3)
>>    plt.show()
>>    ~~~
>>    {: .language-python}
>>
>> 4. Now we are varying two indices. We can use `np.linspace` instead of
>>    `np.arange` to get an array of the correct size.
>>
>>    ~~~
>>    x_range = np.linspace(-30, 30, image.shape[1])
>>    y_range = np.linspace(-30, 30, image.shape[0])
>>    x_values, y_values = np.meshgrid(x_range, y_range, sparse=False)
>>    peaked_function = (np.sin(x_values**2 + y_values**2) /
>>                       (x_values**2 + y_values**2) ** 0.25)
>>    peaked_function = peaked_function[:, :, np.newaxis]
>>    plt.imshow(peaked_function * image)
>>    plt.show()
>>    ~~~
>>    {: .language-python}
>>
>> 5. Now rather than broadcasting when multiplying the image, we
>>    need to broadcast to generate the mask in the first place.
>>    First expand the rightmost axis out to three components, then
>>    make the green and blue components multiply by 1 to not affect
>>    those channels.
>>
>>    ~~~
>>    peaked_function_5 = peaked_function * [1, 0, 0] + [0, 1, 1]
>>    plt.imshow(peaked_function_5 * image)
>>    plt.show()
>>    ~~~
>>    {: .language-python}
> {: .solution}
{: .challenge}
