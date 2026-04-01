@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================
:: Auto Training Platform - Windows 一键安装脚本
:: ============================================

title Auto Training Platform 安装程序

:: 设置颜色
color 0B

echo.
echo ============================================
echo    Auto Training Platform 环境安装程序
echo ============================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [警告] 当前未以管理员身份运行，部分操作可能失败。
    echo 建议右键点击此脚本，选择"以管理员身份运行"。
    echo.
    pause
)

:: ============================================
:: 步骤 1: 检查并安装 Node.js
:: ============================================
echo.
echo [1/7] 检查 Node.js 安装状态...

node --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%a in ('node --version') do set NODE_VERSION=%%a
    echo [✓] Node.js 已安装: !NODE_VERSION!
) else (
    echo [*] Node.js 未安装，开始安装...
    
    :: 使用 winget 安装 Node.js
    winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
    
    if %errorLevel% equ 0 (
        echo [✓] Node.js 安装成功
        :: 刷新环境变量
        call :RefreshEnv
    ) else (
        echo [✗] Node.js 安装失败，请手动安装
        echo 下载地址: https://nodejs.org/
        pause
        exit /b 1
    )
)

:: ============================================
:: 步骤 2: 检查并安装 Bun
:: ============================================
echo.
echo [2/7] 检查 Bun 安装状态...

bun --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%a in ('bun --version') do set BUN_VERSION=%%a
    echo [✓] Bun 已安装: !BUN_VERSION!
) else (
    echo [*] Bun 未安装，开始安装...
    
    :: 使用 PowerShell 安装 Bun
    powershell -Command "irm bun.sh/install.ps1 | iex"
    
    if %errorLevel% equ 0 (
        echo [✓] Bun 安装成功
        :: 刷新环境变量
        call :RefreshEnv
    ) else (
        echo [✗] Bun 安装失败，尝试使用 npm 安装...
        
        npm install -g bun
        if %errorLevel% equ 0 (
            echo [✓] Bun 安装成功 (通过 npm)
            call :RefreshEnv
        ) else (
            echo [✗] Bun 安装失败，请手动安装
            echo 参考: https://bun.sh/docs/installation
            pause
            exit /b 1
        )
    )
)

:: ============================================
:: 步骤 3: 检查并安装 Python
:: ============================================
echo.
echo [3/7] 检查 Python 安装状态...

python --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%a in ('python --version') do set PYTHON_VERSION=%%a
    echo [✓] !PYTHON_VERSION! 已安装
) else (
    echo [*] Python 未安装，开始安装 Python 3.10...
    
    :: 使用 winget 安装 Python
    winget install Python.Python.3.10 --accept-source-agreements --accept-package-agreements
    
    if %errorLevel% equ 0 (
        echo [✓] Python 3.10 安装成功
        call :RefreshEnv
    ) else (
        echo [✗] Python 安装失败，请手动安装
        echo 下载地址: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: ============================================
:: 步骤 4: 检查并安装 Git
:: ============================================
echo.
echo [4/7] 检查 Git 安装状态...

git --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%a in ('git --version') do set GIT_VERSION=%%a
    echo [✓] !GIT_VERSION! 已安装
) else (
    echo [*] Git 未安装，开始安装...
    
    winget install Git.Git --accept-source-agreements --accept-package-agreements
    
    if %errorLevel% equ 0 (
        echo [✓] Git 安装成功
        call :RefreshEnv
    ) else (
        echo [✗] Git 安装失败，请手动安装
        echo 下载地址: https://git-scm.com/download/win
        pause
        exit /b 1
    )
)

:: ============================================
:: 步骤 5: 克隆 PaddleDetection
:: ============================================
echo.
echo [5/7] 检查 PaddleDetection...

set "PADDLE_DETECTION_DIR=%USERPROFILE%\PaddleDetection"

if exist "%PADDLE_DETECTION_DIR%" (
    echo [✓] PaddleDetection 已存在于: %PADDLE_DETECTION_DIR%
) else (
    echo [*] 克隆 PaddleDetection 仓库...
    
    cd /d "%USERPROFILE%"
    git clone https://github.com/PaddlePaddle/PaddleDetection.git
    
    if exist "%PADDLE_DETECTION_DIR%" (
        echo [✓] PaddleDetection 克隆成功
        
        :: 切换到稳定版本
        cd /d "%PADDLE_DETECTION_DIR%"
        git checkout release/2.6
    ) else (
        echo [✗] PaddleDetection 克隆失败
        pause
        exit /b 1
    )
)

:: ============================================
:: 步骤 6: 安装 PaddleDetection 依赖
:: ============================================
echo.
echo [6/7] 安装 PaddleDetection 依赖...

cd /d "%PADDLE_DETECTION_DIR%"

:: 安装 requirements
echo [*] 安装 requirements.txt...
pip install -r requirements.txt

:: 安装 pycocotools-windows
echo [*] 安装 pycocotools-windows...
pip install pycocotools-windows

:: 安装 PaddlePaddle (CPU版本)
echo [*] 安装 PaddlePaddle...
pip install paddlepaddle

:: 创建必要的配置目录
echo [*] 创建配置目录...
if not exist "configs\autotrain\jobs" mkdir "configs\autotrain\jobs"
if not exist "configs\autotrain\training\default" mkdir "configs\autotrain\training\default"
if not exist "configs\autotrain\training\user" mkdir "configs\autotrain\training\user"
if not exist "configs\autotrain\models" mkdir "configs\autotrain\models"

echo [✓] PaddleDetection 依赖安装完成

:: ============================================
:: 步骤 7: 配置 Auto Training 项目
:: ============================================
echo.
echo [7/7] 配置 Auto Training 项目...

:: 检查是否已存在项目目录
if exist "autotrain" (
    echo [*] 项目目录已存在，更新代码...
    cd autotrain
    git pull
) else (
    echo [*] 克隆 Auto Training 项目...
    git clone https://github.com/HansenLYX0708/autotrain.git
    cd autotrain
)

:: 安装项目依赖
echo [*] 安装项目依赖 (bun install)...
bun install

if %errorLevel% neq 0 (
    echo [✗] 依赖安装失败
    pause
    exit /b 1
)

:: 创建环境文件
if not exist ".env.local" (
    echo [*] 创建环境配置文件...
    (
        echo DATABASE_URL="file:./db/custom.db"
        echo NEXT_PUBLIC_API_URL="http://localhost:3000"
    ) > .env.local
)

:: 初始化数据库
echo [*] 初始化数据库...
bunx prisma generate
bunx prisma db push

echo [✓] 项目配置完成

:: ============================================
:: 安装完成
:: ============================================
echo.
echo ============================================
echo    安装完成！
echo ============================================
echo.
echo 项目路径: %CD%
echo PaddleDetection 路径: %PADDLE_DETECTION_DIR%
echo.
echo 启动命令:
echo   bun run dev
echo.
echo 启动后访问: http://localhost:3000
echo.
echo 首次使用请:
echo 1. 注册账号登录系统
echo 2. 进入 Settings 页面配置路径:
echo    - Python Path: 查看下方路径
echo    - PaddleDetection Path: %PADDLE_DETECTION_DIR%
echo.

:: 显示 Python 路径
where python
echo.

pause
exit /b 0

:: ============================================
:: 子程序: 刷新环境变量
:: ============================================
:RefreshEnv
(
    echo Set objShell = CreateObject^("WScript.Shell"^)
    echo Set objEnv = objShell.Environment^("USER"^)
    echo For Each strItem in objEnv
    echo     WScript.Echo strItem
    echo Next
) > "%TEMP%\_env.vbs"

for /f "tokens=*" %%a in ('cscript //nologo "%TEMP%\_env.vbs"') do (
    set "%%a"
)
del "%TEMP%\_env.vbs" >nul 2>&1

goto :eof
