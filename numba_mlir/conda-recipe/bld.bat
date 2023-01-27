copy /y "numba_mlir/conda-recipe/MKLConfig.cmake" "%CONDA_PREFIX%/Library/lib/cmake/mkl/MKLConfig.cmake"
if errorlevel 1 exit 1

"%PYTHON%" numba_mlir/setup.py install
if errorlevel 1 exit 1
