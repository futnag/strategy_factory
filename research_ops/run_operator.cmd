@echo off
rem strategy-factory 無人運用エントリポイント（タスクスケジューラから起動）
rem 登録: research_ops/RUNBOOK.md §3 参照。フラグは `claude --help` で要確認。
setlocal
cd /d C:\Users\futos\claude_local_sessions
if not exist research_ops\logs mkdir research_ops\logs
set LOGFILE=research_ops\logs\operator_%date:~0,4%%date:~5,2%%date:~8,2%.log

echo ==== operator run %date% %time% ==== >> %LOGFILE%
claude -p "/operator" --model opus --permission-mode bypassPermissions >> %LOGFILE% 2>&1
echo ==== exit %errorlevel% %time% ==== >> %LOGFILE%
endlocal
