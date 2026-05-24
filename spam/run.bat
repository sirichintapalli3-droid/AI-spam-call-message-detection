@echo off
cd /d %~dp0
python app.py
start http://127.0.0.1:5000/
pause