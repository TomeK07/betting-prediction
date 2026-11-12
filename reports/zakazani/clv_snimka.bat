@echo off
rem One-off snapshot for CLV + archive (Tome 31.10), NO staking. Run by Windows Task Scheduler.
cd /d "C:\Users\tomek\OneDrive\Desktop\betting-prediction"
set PYTHONIOENCODING=utf-8
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"') do set TS=%%i
venv\Scripts\python.exe main.py fetch --league nba >> "reports\zakazani\clv_%TS%.log" 2>&1
venv\Scripts\python.exe main.py linii >> "reports\zakazani\clv_%TS%.log" 2>&1
