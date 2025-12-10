@echo off
title Building DMT Library (Windows)
cls

echo ==============================================
echo      COMPILING WINDOWS DLL
echo ==============================================

:: 1. Создаем папку bin, если ее нет
if not exist "bin" mkdir bin

:: 2. Компиляция
:: -shared      : создать DLL
:: -D DMT_EXPORTS : сказать коду, что мы экспортируем функции
:: -o ...       : куда положить результат
g++ -shared -o bin/dmt_lib.dll cpp_core/dmt_lib.cpp -I include -O3 -mavx2 -fopenmp -D DMT_EXPORTS

:: 3. Проверка на ошибки
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation FAILED!
    echo Check the errors above.
    color 4
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Created: bin/dmt_lib.dll
echo ==============================================
color 2
pause