# Diamond Miner

Extract Coulomb diamond information from a provided measurement dataset. 

## Installation

To install this package you must first activate your environment (example below if you are using conda),
```shell
conda activate ENV_NAME
```
Then you can build the package using `setup.py` which will generate some folders in your repository,
```shell
python setup.py bdist_wheel sdist
```
Then using `pip` to install the package based on the build folders created from the command above,
```shell
python -m pip install . 
```

## Usage

Please see `demo.ipynb` for usage.

## Results

Below are two results that can be reproduced in the `demo.ipynb`: (1) Coulomb diamonds and estimated dot information, (2) Charge temperature information.

### Coulomb Diamonds

![alt text](photos/diamonds.svg)

```text
Summary (#0):
====================
Left Vertex: [0.135 0.   ]
Top Vertex: [0.18432432 0.00135664]
Right Vertex: [0.20898649 0.        ]
Bottom Vertex: [ 0.15966216 -0.00126573]
Elementary Charge (e): 1.60218e-19 C
Permittivity of Free Space (ϵ0): 8.85419e-12 F/m
Relative Permittivity (ϵR): 11.70000
Width: 0.07399 V
Height: 0.00262 V
Lever Arm (α): 0.01772 eV/V
Addition Voltage: 0.07399 V
Charging Voltage: 0.00131 V
Total Capacitance: 122.19267 aF
Gate Capacitance: 2.16550 aF
Dot Size: 147.44178 nm

... and more ...
```

And the statistics can also be calculated as,
```text
Average Lever Arm (α) : 0.02087 (eV/V) ± 0.00047 (eV/V)
Average Addition Voltage: 0.07004 (V) ± 0.00097 (V)
Average Charging Voltage: 0.00145 (V) ± 0.00002 (V)
Average Total Capacitance: 110.91059 (aF) ± 1.70902 (aF)
Average Gate Capacitance: 2.29876 (aF) ± 0.03264 (aF)
Average Dot Size: 133.82845 (nm) ± 2.06216 (nm)
```

### Charge Temperature

![alt text](photos/temperature.svg)