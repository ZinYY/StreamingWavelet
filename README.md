# Python implementation for _Streaming Wavelet Operator_

This is the python implementation for the Streaming Wavelet Operator, which **sequentially**
applies wavelet transform to a sequence efficiently in an online manner (instead of recalculation in each round).

The **speed** of the Streaming Wavelet Operator is much faster than the traditional wavelet transform for streaming data,
especially for long signals, due to its use of lazy updates and bit-wise operations in the implementation.

You can install the `StreamingWavelet` package in [https://pypi.org/project/StreamingWavelet/](https://pypi.org/project/StreamingWavelet/).

We will first introduce the structure and requirements of the code, followed by a brief instruction for a quick start.

## Reference:

- Qian et al., Efficient Non-stationary Online Learning by Wavelets with Applications to Online Distribution Shift Adaptation. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024).

## Install:

```
pip install StreamingWavelet
```

## Code Structure:

- `StreamingWavelet/StreamingWavelet.py`: The main file for the Streaming Wavelet Operator, which supports dozens types of wavelet bases.
- `StreamingWavelet/MakeCDJVFilter.py`: The file for generating the filters of different wavelet transforms, thanks to the MatLab code of _Cohen, Daubechies, Jawerth and Vial, 1992_.
- `StreamingWavelet/wavelet_coeff`: The folder for storing the different wavelet coefficients.

## Requirements:

* numpy>=1.19.0

## Quick Start & Demos:

We provide a concrete demo here.
For example, one can use the following code to generate the following Streaming Wavelet Operator
for a sequence of `dim=128`, `max_length=10000`, and using the Haar wavelets (`order=1`) as wavelet basis.

```python
import StreamingWavelet

SW = StreamingWavelet.Operator(128, 10000, 1)
```

Then, for a sequence of length 10000, one can use the following code to sequentially calculate the wavelet coefficients in an online manner:

```python
import numpy as np
import StreamingWavelet

SW = StreamingWavelet.Operator(128, 10000, 1, get_coeff=False)  # Initialize the Streaming Wavelet Operator

# Generate a sequence of length 10000
x_list = []
for i in range(10000):
    x_list.append(np.random.randn(128))  # Generate a random element of dim=128, and add it to the sequence

for i in range(10000):
    SW.add_signal(x_list[i])  # Update the wavelet coefficients by adding the new element
    current_norm = SW.get_norm()  # Get the norm of the wavelet coefficients
    print('Norm of Wavelet Coefficients of x_list[0:{}]:'.format(i), current_norm)  # Print the current norm of the wavelet coefficients
```

which will output the norm of the wavelet coefficients in each round.

---

Note that the default mode is `get_coeff=False`, which will only maintain the **2-norm** of the wavelet coefficients.
If you want to get the wavelet coefficients, you can set `get_coeff=True` when initializing the Streaming Wavelet Operator (this will take more storage):

```python
import numpy as np
import StreamingWavelet

SW = StreamingWavelet.Operator(128, 10000, 1, get_coeff=True)  # Initialize the Streaming Wavelet Operator

# Generate a sequence of length 10000
x_list = []
for i in range(10000):
    x_list.append(np.random.randn(128))  # Generate a random element of dim=128, and add it to the sequence

for i in range(10000):
    SW.add_signal(x_list[i])  # Update the wavelet coefficients by adding the new element
    current_norm = SW.get_norm()  # Get the norm of the wavelet coefficients
    print('Norm of Wavelet Coefficients of x_list[0:{}]:'.format(i), current_norm)  # Print the current norm of the wavelet coefficients
    if (i + 1) % 1000 == 0:
        print('Wavelet Coefficients of x_list[0:{}]:'.format(i), SW.all_coeff_arrs[:5])  # Print the wavelet coefficients
```

## Parameters in `StreamingWavelet.Operator`:

- `dim`: The dimension of the input signal.
- `max_length`: The maximum length of the sequence.
- `order`: The order of the wavelet transform (e.g., order=1 means Haar wavelets; order>=2 means various Daubechies wavelets).
- `get_coeff`: Whether to maintain the whole wavelet coefficients (default: False).
- `axis`: The axis to apply the wavelet transform (default: -1).
- `verbose`: Whether to print the running information (default: False).