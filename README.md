# C++ Protoype of Stable Multiscale Finite Element Complexes


The C++ code of the *MsFEComplex* subprojects which serve as a demonstration of the method can be found in subfolders.

To set up an Eclipse project using cmake enter the folder containing the problem implementation and type
```
mkdir build/
cd build/
cmake -DDEAL_II_DIR=~/path/to/dealii -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-jN ..
```
Replace `N` with the number of cores of your machine and `/path/to/dealii` with the path containing the library.

Please do not commit to the master branch directly. If you wish to make changes open a separate branch.