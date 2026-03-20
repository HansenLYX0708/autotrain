Set WshShell = CreateObject("WScript.Shell")

projectPath = "D:\_work\projects\autoTraining\AutoTrain"
url = "http://localhost:3000"
port = "3000"

' 判断端口是否已被监听
Function IsRunning(port)
    Set exec = WshShell.Exec("cmd /c netstat -ano | findstr :" & port)
    output = exec.StdOut.ReadAll

    If InStr(output, "LISTENING") > 0 Then
        IsRunning = True
    Else
        IsRunning = False
    End If
End Function

If IsRunning(port) Then
    ' 已运行 → 只打开浏览器
    ' WshShell.Run "cmd /c start " & url, 0
Else
    ' 未运行 → 启动服务
    cmd = "cmd /c cd /d """ & projectPath & """ && bun start"
    WshShell.Run cmd, 0

    ' 等待服务启动
    WScript.Sleep 3000

    ' 打开浏览器
    ' WshShell.Run "cmd /c start " & url, 0
End If

Set WshShell = Nothing