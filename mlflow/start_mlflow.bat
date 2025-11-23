@echo off
cd %~dp0

if not exist artifacts (
    mkdir artifacts
)

mlflow server ^
    --backend-store-uri sqlite:///mlflow.db ^
    --default-artifact-root artifacts ^
    --host 0.0.0.0 ^
    --port 5000

pause
