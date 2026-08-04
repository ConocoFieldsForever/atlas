@echo off
title Atlas - Build All Maps
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-all-atlas-maps.ps1"
set "ATLAS_BUILD_EXIT=%ERRORLEVEL%"
echo.
if not "%ATLAS_BUILD_EXIT%"=="0" (
  echo Atlas Build All stopped with an error. Review the message above.
) else (
  echo Atlas Build All finished successfully.
)
echo.
pause
exit /b %ATLAS_BUILD_EXIT%
