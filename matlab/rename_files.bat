@echo off
setlocal enabledelayedexpansion

cd /d "E:\pythonProject\similarity\matlab"  

rem 输出当前路径以确认是否进入正确目录
echo Current Directory: %cd%

rem 设置计数器
set count=1

rem 遍历所有 png 文件并重命名
for %%f in (*.png) do (
    rem 获取文件扩展名
    set "ext=%%~xf"
    rem 获取文件名
    set "filename=%%~nf"
    rem 如果文件名不符合"image"开头的格式，则进行重命名
    if /i not "!filename!"=="image_!count!" (
        echo Renaming "%%f" to "image!count!!ext!"
        ren "%%f" "image!count!!ext!"
    )
    set /a count=!count!+1
)

pause