@echo off
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
title AcgML

echo.
echo  ================================
echo   AcgML — Armando Cruz
echo  ================================
echo.

:: Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python no encontrado.
    pause & exit /b 1
)

:: Verificar Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Node.js no encontrado.
    pause & exit /b 1
)

:: Cerrar procesos anteriores en los puertos usados
echo  Cerrando procesos anteriores...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173 "') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5174 "') do taskkill /PID %%a /F >nul 2>&1
timeout /t 1 /nobreak >nul

:: Instalar dependencias Python
echo  [1/4] Instalando dependencias Python...
cd /d "%ROOT%backend"
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  ERROR al instalar dependencias Python.
    pause & exit /b 1
)
echo  Dependencias Python listas.
echo.

:: Instalar dependencias Node
echo  [2/4] Instalando dependencias Node.js...
cd /d "%ROOT%frontend"
call npm install --silent
if %errorlevel% neq 0 (
    echo  ERROR al instalar dependencias Node.js.
    pause & exit /b 1
)
echo  Dependencias Node.js listas.
echo.

:: Limpiar cache de Vite
echo  [3/4] Limpiando cache y arrancando backend...
cd /d "%ROOT%frontend"
if exist "node_modules\.vite" rmdir /s /q "node_modules\.vite"

:: Iniciar backend
cd /d "%ROOT%backend"
start "Backend - AcgML" cmd /k "uvicorn api:app --reload --host localhost --port 8000"
timeout /t 2 /nobreak >nul

:: Iniciar frontend
echo  [4/4] Iniciando frontend...
cd /d "%ROOT%frontend"
start "Frontend - AcgML" cmd /k "npm run dev"
timeout /t 3 /nobreak >nul

echo.
echo  ================================
echo   Listo.
echo   http://localhost:5173
echo  ================================
echo.
pause
