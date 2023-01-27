copy /y "%RECIPE_DIR%"\MKLConfig.cmake "%BUILD_PREFIX%"\Library\lib\cmake\mkl
if errorlevel 1 exit 1

pushd numba_mlir
"%PYTHON%" setup.py install
if errorlevel 1 exit 1
popd
