@echo off
echo module will be installed
for /f %%j in (requirements.txt) do echo %%j
echo Start installing
for /f %%i in (requirements.txt) do pip install %%i
echo module installed
pip list
pause