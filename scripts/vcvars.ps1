pushd "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\"
cmd /c "vcvars64.bat & set" |
foreach {
  if ($_ -match "=") {
    $v = $_.split("=", 2); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
  }
}
popd
