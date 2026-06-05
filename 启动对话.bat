@echo off
chcp 65001 >nul
title rsinfer - TS-Qwen3
"%~dp0target\release\rsinfer.exe" --model-path "D:\MCP_Server\root\autodl-tmp\TS-Qwen3" --chat -i -n 1024
echo.
echo ===== exited, press any key to close =====
pause >nul
