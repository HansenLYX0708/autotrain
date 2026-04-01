-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "username" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "role" TEXT NOT NULL DEFAULT 'user',
    "status" TEXT NOT NULL DEFAULT 'active',
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "SystemConfig" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "pythonPath" TEXT,
    "condaEnv" TEXT,
    "condaPath" TEXT,
    "paddleDetectionPath" TEXT,
    "paddleClasPath" TEXT,
    "defaultGpu" INTEGER NOT NULL DEFAULT 0,
    "defaultFramework" TEXT NOT NULL DEFAULT 'PaddleDetection',
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "Project" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "framework" TEXT NOT NULL DEFAULT 'PaddleDetection',
    "status" TEXT NOT NULL DEFAULT 'active',
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "userId" TEXT,
    CONSTRAINT "Project_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Dataset" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "projectId" TEXT NOT NULL,
    "format" TEXT NOT NULL DEFAULT 'COCO',
    "trainImagePath" TEXT,
    "trainAnnoPath" TEXT,
    "evalImagePath" TEXT,
    "evalAnnoPath" TEXT,
    "datasetDir" TEXT,
    "numClasses" INTEGER NOT NULL DEFAULT 0,
    "numAnnotations" INTEGER NOT NULL DEFAULT 0,
    "numTrainImages" INTEGER NOT NULL DEFAULT 0,
    "numEvalImages" INTEGER NOT NULL DEFAULT 0,
    "classStats" TEXT,
    "yamlConfig" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "userId" TEXT,
    CONSTRAINT "Dataset_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "Dataset_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Model" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "projectId" TEXT NOT NULL,
    "architecture" TEXT NOT NULL DEFAULT 'YOLOv3',
    "backbone" TEXT NOT NULL DEFAULT 'CSPResNet',
    "neck" TEXT NOT NULL DEFAULT 'CustomCSPPAN',
    "head" TEXT NOT NULL DEFAULT 'PPYOLOEHead',
    "numClasses" INTEGER NOT NULL DEFAULT 1,
    "normType" TEXT NOT NULL DEFAULT 'sync_bn',
    "useEma" BOOLEAN NOT NULL DEFAULT true,
    "emaDecay" REAL NOT NULL DEFAULT 0.9998,
    "depthMult" REAL NOT NULL DEFAULT 0.33,
    "widthMult" REAL NOT NULL DEFAULT 0.50,
    "pretrainWeights" TEXT,
    "yamlConfig" TEXT,
    "userId" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "Model_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "Model_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "TrainingConfig" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "projectId" TEXT NOT NULL,
    "epoch" INTEGER NOT NULL DEFAULT 100,
    "batchSize" INTEGER NOT NULL DEFAULT 8,
    "baseLr" REAL NOT NULL DEFAULT 0.001,
    "momentum" REAL NOT NULL DEFAULT 0.9,
    "weightDecay" REAL NOT NULL DEFAULT 0.0005,
    "scheduler" TEXT NOT NULL DEFAULT 'CosineDecay',
    "warmupEpochs" INTEGER NOT NULL DEFAULT 5,
    "maxEpochs" INTEGER NOT NULL DEFAULT 100,
    "workerNum" INTEGER NOT NULL DEFAULT 4,
    "evalHeight" INTEGER NOT NULL DEFAULT 640,
    "evalWidth" INTEGER NOT NULL DEFAULT 640,
    "useGpu" BOOLEAN NOT NULL DEFAULT true,
    "logIter" INTEGER NOT NULL DEFAULT 20,
    "saveDir" TEXT,
    "snapshotEpoch" INTEGER NOT NULL DEFAULT 1,
    "outputDir" TEXT,
    "weights" TEXT,
    "pretrainWeights" TEXT,
    "yamlConfig" TEXT,
    "userId" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "TrainingConfig_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "TrainingConfig_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "TrainingJob" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "projectId" TEXT NOT NULL,
    "datasetId" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "configId" TEXT,
    "name" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "command" TEXT,
    "evalCommand" TEXT,
    "inferCommand" TEXT,
    "configPath" TEXT,
    "errorMessage" TEXT,
    "currentEpoch" INTEGER NOT NULL DEFAULT 0,
    "totalEpochs" INTEGER NOT NULL DEFAULT 100,
    "currentLoss" REAL,
    "currentLr" REAL,
    "outputDir" TEXT,
    "weightsPath" TEXT,
    "vdlLogDir" TEXT,
    "trainingParams" TEXT,
    "yamlConfig" TEXT,
    "userId" TEXT,
    "startedAt" DATETIME,
    "completedAt" DATETIME,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "TrainingJob_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "TrainingJob_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "TrainingJob_datasetId_fkey" FOREIGN KEY ("datasetId") REFERENCES "Dataset" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "TrainingJob_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "TrainingJob_configId_fkey" FOREIGN KEY ("configId") REFERENCES "TrainingConfig" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "TrainingLog" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "jobId" TEXT NOT NULL,
    "epoch" INTEGER NOT NULL,
    "iteration" INTEGER NOT NULL,
    "totalIter" INTEGER NOT NULL,
    "loss" REAL,
    "lossCls" REAL,
    "lossIou" REAL,
    "lossDfl" REAL,
    "lossL1" REAL,
    "learningRate" REAL,
    "eta" TEXT,
    "batchCost" REAL,
    "dataCost" REAL,
    "ips" REAL,
    "memReserved" INTEGER,
    "memAllocated" INTEGER,
    "rawLog" TEXT,
    "timestamp" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "TrainingLog_jobId_fkey" FOREIGN KEY ("jobId") REFERENCES "TrainingJob" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "ValidationJob" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "projectId" TEXT NOT NULL,
    "trainingJobId" TEXT,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL DEFAULT 'eval',
    "configPath" TEXT,
    "weightsPath" TEXT,
    "datasetPath" TEXT,
    "inferInputPath" TEXT,
    "inferOutputPath" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "command" TEXT,
    "resultJson" TEXT,
    "resultPath" TEXT,
    "outputLog" TEXT,
    "userId" TEXT,
    "startedAt" DATETIME,
    "completedAt" DATETIME,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "ValidationJob_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "ValidationJob_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "GpuMetric" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "gpuId" INTEGER NOT NULL,
    "utilization" REAL NOT NULL,
    "memoryUsed" INTEGER NOT NULL,
    "memoryTotal" INTEGER NOT NULL,
    "temperature" INTEGER NOT NULL,
    "timestamp" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateIndex
CREATE UNIQUE INDEX "User_username_key" ON "User"("username");
