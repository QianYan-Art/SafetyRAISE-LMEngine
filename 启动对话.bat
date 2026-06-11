@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title rsinfer - SafetyRAISE-TS-Qwen3
cd /d "%~dp0"

set "EXE=%CD%\target\release\rsinfer.exe"
set "PREFERRED_ARGS=--chat --interactive --max-tokens 1024 --device hybrid --quantization q8 --verbose"
set "CPU_FALLBACK_ARGS=--chat --interactive --max-tokens 1024 --verbose"
set "EXIT_CODE=0"

if "%~1" neq "" set "MODEL_PATH=%~1"
if not defined MODEL_PATH if defined RSINFER_MODEL_PATH set "MODEL_PATH=%RSINFER_MODEL_PATH%"
if not defined MODEL_PATH call :pick_model "D:\MCP_Server\root\autodl-tmp\TS-Qwen3"
if not defined MODEL_PATH call :pick_model "D:\MCP_Server\root\TS_Qwen3_Finetuned"

if not defined MODEL_PATH (
    echo [错误] 未找到可用模型目录。
    echo [提示] 可拖拽模型目录到本脚本上，或先设置环境变量 RSINFER_MODEL_PATH。
    echo [提示] 模型目录需要包含 config.json、tokenizer.json，以及 model.safetensors 或 model.safetensors.index.json。
    set "EXIT_CODE=1"
    goto :finish
)

call :ensure_release
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :finish
)

echo [信息] 模型目录: "%MODEL_PATH%"
echo [信息] 首选模式: chat + interactive + hybrid + q8 ^(resident 默认关闭^)
echo [提示] 如需启用实验 resident，可设置 RSINFER_EXTRA_ARGS=--resident
if defined RSINFER_EXTRA_ARGS (
    echo [信息] 额外参数: %RSINFER_EXTRA_ARGS%
) else (
    echo [信息] 额外参数: ^(无^)
)
echo.

"%EXE%" --model-path "%MODEL_PATH%" %PREFERRED_ARGS% %RSINFER_EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" goto :finish

echo.
echo [警告] hybrid + q8 启动失败，退出码 %EXIT_CODE%。
choice /c YN /n /m "是否自动回退到 CPU 模式重试? [Y/N] "
if errorlevel 2 goto :finish

echo.
echo [信息] 回退模式: chat + interactive + cpu
"%EXE%" --model-path "%MODEL_PATH%" %CPU_FALLBACK_ARGS% %RSINFER_EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
goto :finish

:pick_model
set "CANDIDATE=%~1"
if not exist "%CANDIDATE%" exit /b 0
if not exist "%CANDIDATE%\config.json" exit /b 0
if not exist "%CANDIDATE%\tokenizer.json" exit /b 0
if exist "%CANDIDATE%\model.safetensors" set "MODEL_PATH=%CANDIDATE%"
if exist "%CANDIDATE%\model.safetensors.index.json" set "MODEL_PATH=%CANDIDATE%"
exit /b 0

:ensure_release
if exist "%EXE%" exit /b 0
echo [信息] 未找到 release 二进制，开始执行 cargo build --release ...
cargo build --release
if errorlevel 1 (
    echo [错误] cargo build --release 失败。
    exit /b 1
)
if not exist "%EXE%" (
    echo [错误] 构建完成但仍未找到 "%EXE%"。
    exit /b 1
)
exit /b 0

:finish
echo.
if defined RSINFER_NO_PAUSE exit /b %EXIT_CODE%
if "%EXIT_CODE%"=="0" (
    echo ===== 会话结束，按任意键关闭 =====
) else (
    echo ===== 已退出，退出码 %EXIT_CODE%，按任意键关闭 =====
)
pause >nul
exit /b %EXIT_CODE%
