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

Please see `demo.ipynb` for usage. You can try it out on the `demo_data_(1/2).txt` prepared datasets.

## Results

Below are two results that can be reproduced in the `demo.ipynb`: (1) Coulomb diamonds and estimated dot information, (2) Charge temperature information.

### Coulomb Diamonds

![alt text](photos/demo_data_1/diamonds.svg)

```text
Summary (#0):
===================

---------
Constants
---------
Elementary Charge (e): 1.60218e-19 C
Permittivity of Free Space (ϵ0): 8.85419e-12 F/m
Relative Permittivity (ϵR): 11.70000
---------

---------
Geometry
---------
Left Vertex: [0.135 0.   ]
Top Vertex: [0.18925676 0.00121678]
Right Vertex: [0.20405405 0.        ]
Bottom Vertex: [ 0.15966216 -0.00140559]
Width: 0.06905 V
Height: 0.00262 V
---------

--------------
Dot Properties
--------------
Lever Arm (α): 0.01899 eV/V
Addition Voltage: 0.06905 V
Charging Voltage: 0.00131 V
Gate Capacitance: 2.32018 aF
Source Capacitance: 33.33012 aF
Drain Capacitance: 83.46981 aF
Total Capacitance: 122.19267 aF
Dot Size: 147.44178 nm
--------------


Summary (#1):
===================
...
```

And the statistics can also be calculated as,
```text
Average Lever Arm (α) : 0.02097 (eV/V) ± 0.00063 (eV/V)
Average Addition Voltage: 0.07004 (V) ± 0.00074 (V)
Average Charging Voltage: 0.00146 (V) ± 0.00003 (V)
Average Total Capacitance: 110.96383 (aF) ± 2.33055 (aF)
Average Gate Capacitance: 2.29400 (aF) ± 0.02470 (aF)
Average Source Capacitance: 27.89592 (aF) ± 0.93941 (aF)
Average Drain Capacitance: 77.84920 (aF) ± 1.68834 (aF)
Average Dot Size: 133.89269 (nm) ± 2.81212 (nm)
```

### Charge Temperature

![alt text](photos/demo_data_1/temperature.svg)