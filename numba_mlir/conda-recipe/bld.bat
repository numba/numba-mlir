copy /y "%RECIPE_DIR%"\MKLConfig.cmake "%BUILD_PREFIX%"\Library\lib\cmake\mkl
if errorlevel 1 exit 1

pushd numba_mlir
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
popd
