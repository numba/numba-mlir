echo %CONDA_PREFIX%
echo %BUILD_PREFIX%

copy /y "%RECIPE_DIR%"\MKLConfig.cmake "%CONDA_PREFIX%"\Library\lib\cmake\mkl
if errorlevel 1 exit 1

set "LIB=%CONDA_PREFIX%\Library\lib;%CONDA_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

pushd numba_mlir
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
popd
