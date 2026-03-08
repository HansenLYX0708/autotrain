import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface GpuInfo {
  id: number;
  name: string;
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
}

async function getNvidiaGpuInfo(): Promise<GpuInfo[]> {
  try {
    // Try to get GPU info using nvidia-smi
    const { stdout } = await execAsync(
      'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits',
      { timeout: 5000 }
    );

    const lines = stdout.trim().split('\n');
    const gpus: GpuInfo[] = [];

    for (const line of lines) {
      const parts = line.split(',').map(p => p.trim());
      if (parts.length >= 6) {
        gpus.push({
          id: parseInt(parts[0]) || 0,
          name: parts[1] || 'Unknown GPU',
          utilization: parseFloat(parts[2]) || 0,
          memoryUsed: parseFloat(parts[3]) || 0,
          memoryTotal: parseFloat(parts[4]) || 0,
          temperature: parseFloat(parts[5]) || 0,
        });
      }
    }

    return gpus;
  } catch {
    // nvidia-smi not available or no NVIDIA GPU
    return [];
  }
}

async function getCpuInfo(): Promise<{ usage: number; memoryUsed: number; memoryTotal: number }> {
  try {
    const os = await import('os');
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;

    // Calculate CPU usage (simplified)
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;

    for (const cpu of cpus) {
      for (const type in cpu.times) {
        totalTick += cpu.times[type as keyof typeof cpu.times];
      }
      totalIdle += cpu.times.idle;
    }

    const usage = 100 - ~~(100 * totalIdle / totalTick);

    return {
      usage: Math.min(100, Math.max(0, usage)),
      memoryUsed: Math.round(usedMem / (1024 * 1024)), // MB
      memoryTotal: Math.round(totalMem / (1024 * 1024)), // MB
    };
  } catch {
    return { usage: 0, memoryUsed: 0, memoryTotal: 0 };
  }
}

// GET /api/system/gpu - Get GPU and system information
export async function GET() {
  try {
    const [gpus, cpu] = await Promise.all([
      getNvidiaGpuInfo(),
      getCpuInfo(),
    ]);

    // If no NVIDIA GPU found, show CPU as compute device
    const computeDevices = gpus.length > 0 ? gpus : [{
      id: 0,
      name: 'CPU (No GPU detected)',
      utilization: cpu.usage,
      memoryUsed: cpu.memoryUsed,
      memoryTotal: cpu.memoryTotal,
      temperature: 0,
    }];

    return NextResponse.json({
      success: true,
      data: {
        gpus: computeDevices,
        cpu,
        hasNvidia: gpus.length > 0,
        platform: process.platform,
      },
    });
  } catch (error) {
    console.error('Error getting system info:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to get system information',
        data: {
          gpus: [{
            id: 0,
            name: 'Unknown Device',
            utilization: 0,
            memoryUsed: 0,
            memoryTotal: 0,
            temperature: 0,
          }],
          cpu: { usage: 0, memoryUsed: 0, memoryTotal: 0 },
          hasNvidia: false,
          platform: process.platform,
        },
      },
      { status: 200 } // Return 200 with default data to not break the UI
    );
  }
}
