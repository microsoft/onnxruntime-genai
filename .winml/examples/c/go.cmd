copy ..\..\src\ort_genai.h include
copy ..\..\src\ort_genai_c.h include
copy ..\..\build\directml\win-arm64\RelWithDebInfo\onnxruntime-genai.lib lib

cmake -A arm64 -S . -B build -DPHI3=ON
cmake --build build --config Debug 

rem BUG - The build is placing the wrong WinMLBootstrap.dll

copy ..\..\build\directml\win-arm64\RelWithDebInfo\WinMLBootstrap.* Build\Debug
copy ..\..\build\directml\win-arm64\RelWithDebInfo\onnxruntime-genai.* Build\Debug
