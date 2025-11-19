@echo off
echo Compilando ResNet18...

g++ -std=c++17 -O2 -Wall -I include src/tensor.cpp src/conv2d.cpp src/main.cpp -o resnet18.exe

if %ERRORLEVEL% EQU 0 (
    echo Compilacao concluida! Executavel: resnet18.exe
) else (
    echo Erro na compilacao!
    pause
)

