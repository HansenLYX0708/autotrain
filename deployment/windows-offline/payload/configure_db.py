import argparse
import json
import sqlite3
import time

parser = argparse.ArgumentParser()
parser.add_argument("--db", required=True)
parser.add_argument("--root", required=True)
args = parser.parse_args()
root = args.root.replace("/", "\\").rstrip("\\")
now = int(time.time() * 1000)
paths = {
    "pythonEnvsBasePath": root + "\\runtime\\envs",
    "userConfigsPath": root + "\\data\\configs",
    "userDatabasePath": root + "\\data\\users",
    "paddleDetectionPath": root + "\\frameworks\\PaddleDetection",
    "paddleClasPath": root + "\\frameworks\\PaddleClas",
    "paddleSegPath": root + "\\frameworks\\PaddleSeg",
    "torchPath": root + "\\frameworks\\torchtrain",
}
mapping = json.dumps({
    "PaddleDetection": root + "\\runtime\\envs\\paddle\\Scripts\\python.exe",
    "PaddleClas": root + "\\runtime\\envs\\paddle\\Scripts\\python.exe",
    "PaddleSeg": root + "\\runtime\\envs\\paddle\\Scripts\\python.exe",
    "TorchDet": root + "\\runtime\\envs\\torch\\Scripts\\python.exe",
    "TorchSeg": root + "\\runtime\\envs\\torch\\Scripts\\python.exe",
    "TorchAnomaly": root + "\\runtime\\envs\\anomaly\\Scripts\\python.exe",
})
conn = sqlite3.connect(args.db)
row = conn.execute('SELECT "id" FROM "SystemConfig" ORDER BY "createdAt" LIMIT 1').fetchone()
values = [paths[key] for key in ("pythonEnvsBasePath", "userConfigsPath", "userDatabasePath", "paddleDetectionPath", "paddleClasPath", "paddleSegPath", "torchPath")]
if row:
    conn.execute('UPDATE "SystemConfig" SET "pythonEnvsBasePath"=?, "userConfigsPath"=?, "userDatabasePath"=?, "paddleDetectionPath"=?, "paddleClasPath"=?, "paddleSegPath"=?, "torchPath"=?, "frameworkPythonMappings"=?, "gpuPythonMappings"=?, "updatedAt"=? WHERE "id"=?', values + [mapping, "{}", now, row[0]])
else:
    conn.execute('INSERT INTO "SystemConfig" ("id", "condaEnv", "condaPath", "pythonEnvsBasePath", "gpuPythonMappings", "frameworkPythonMappings", "userConfigsPath", "userDatabasePath", "paddleDetectionPath", "paddleClasPath", "paddleSegPath", "torchPath", "defaultFramework", "createdAt", "updatedAt") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', ["system-config", "", "", paths["pythonEnvsBasePath"], "{}", mapping, paths["userConfigsPath"], paths["userDatabasePath"], paths["paddleDetectionPath"], paths["paddleClasPath"], paths["paddleSegPath"], paths["torchPath"], "PaddleDetection", now, now])
conn.commit()
conn.close()
