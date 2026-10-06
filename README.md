# mylibdeguchi
A small library for computing differential operators using PyTorch.

## Installation
Clone the repository and install the package using pip:

```bash
git clone https://github.com/ShotaDeguchi/mylibdeguchi.git
cd mylibdeguchi
pip install .
```

## Usage
### MLP
You can create a multi-layer perceptron (MLP) using the `MLP` class.
```python
import torch
import mylib

BATCH_SIZE = 32
IN_DIM = 2
OUT_DIM = 1
model = mylib.MLP(
    in_dim=IN_DIM,
    out_dim=OUT_DIM,
    hid_dim=64,
    n_layers=3
)

x = torch.randn(BATCH_SIZE, IN_DIM)
y = model(x)
```

### Jacobian
You can compute the Jacobian of a function using the `jacobian` function.
```python
import torch
import mylib

def func(x):
    return torch.stack([x[0]**2, x[1]**3], dim=0)

x = torch.randn(32, 2)
jacobian_matrix = mylib.jacobian(func, x)
```

You can also compute the Jacobian of an MLP model:
```python
import torch
import mylib

BATCH_SIZE = 32
IN_DIM = 2
model = mylib.MLP(
    in_dim=IN_DIM,
    out_dim=1,
    hid_dim=64,
    n_layers=3
)
x = torch.randn(BATCH_SIZE, IN_DIM)
jacobian_matrix = mylib.jacobian(model, x)
```

## License
MIT License
