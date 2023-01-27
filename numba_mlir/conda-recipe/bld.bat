copy /y "%RECIPE_DIR%"\MKLConfig.cmake "%BUILD_PREFIX%"\Library\lib\cmake\mkl
if errorlevel 1 exit 1

"%PYTHON%" numba_mlir/setup.py install
if errorlevel 1 exit 1
