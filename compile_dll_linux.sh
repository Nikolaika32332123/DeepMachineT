#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=============================================="
echo "     COMPILING LINUX SHARED OBJECT (.so)"
echo "=============================================="

mkdir -p bin

# -fPIC        : Обязательно для Linux библиотек (Position Independent Code)
# libdmt.so    : Стандарт именования в Linux (префикс lib)
g++ -shared -fPIC -o bin/dmt_lib.so cpp_core/dmt_lib.cpp -I include -O3 -mavx2 -fopenmp

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Created: bin/libdmt.so${NC}"
else
    echo -e "${RED}[ERROR] Compilation FAILED!${NC}"
    exit 1
fi

echo "=============================================="