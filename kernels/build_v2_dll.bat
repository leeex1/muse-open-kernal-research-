@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -O3 -arch=sm_61 -allow-unsupported-compiler --shared -Xcompiler "/MD" "C:\02_QUILLAN\kernels\quillan_cuda_sm61_v2.cu" -o "C:\02_QUILLAN\kernels\quillan_cuda_sm61_v2.dll"
echo BUILD_RC=%ERRORLEVEL%
