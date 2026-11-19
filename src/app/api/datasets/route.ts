import { NextRequest, NextResponse } from 'next/server';
import { resolveCocoValidationPaths } from '@/lib/coco-annotations';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';
import { logActivity } from '@/lib/activity-log';
import { getWorkDir, isAnomaly, isSegmentation } from '@/lib/frameworks';
import { ANOMALY_IMAGE_EXTENSIONS, generateDatasetYaml } from '@/lib/dataset-yaml';
import { resolveDatasetRoot } from '@/lib/dataset-paths';
import * as fs from 'fs';
import * as path from 'path';

// Parse COCO JSON file
async function parseCocoFile(filePath: string) {
  try {
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      console.error('File not found:', filePath);
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(content);
    if (!Array.isArray(data.images) || !Array.isArray(data.annotations) || !Array.isArray(data.categories)) {
      console.error('Invalid COCO arrays:', filePath);
      return null;
    }
    
    return {
      images: data.images || [],
      annotations: data.annotations || [],
      categories: data.categories || [],
    };
  } catch (error) {
    console.error('Error parsing COCO file:', error);
    return null;
  }
}

// Calculate statistics from COCO dataset
function calculateStats(cocoData: { images: any[]; annotations: any[]; categories: any[] }) {
  const categoryCount: Record<number, number> = {};
  const categoryImageCount: Record<number, Set<number>> = {};

  // Count annotations per category and track images per category
  for (const ann of cocoData.annotations) {
    categoryCount[ann.category_id] = (categoryCount[ann.category_id] || 0) + 1;
    
    if (!categoryImageCount[ann.category_id]) {
      categoryImageCount[ann.category_id] = new Set();
    }
    categoryImageCount[ann.category_id].add(ann.image_id);
  }

  // Build category statistics
  const classStats = cocoData.categories.map((cat: any) => ({
    id: cat.id,
    name: cat.name,
    count: categoryCount[cat.id] || 0,
    imageCount: categoryImageCount[cat.id]?.size || 0,
  }));

  return {
    numClasses: cocoData.categories.length,
    numAnnotations: cocoData.annotations.length,
    numImages: cocoData.images.length,
    classStats,
  };
}

// GET /api/datasets - Get all datasets with project info (filtered by user for non-admins)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;

    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');

    // Build where clause with user filter
    const userFilter = buildUserFilter(userId, role, 'userId');
    const where: Record<string, unknown> = { ...userFilter };
    
    if (projectId) {
      where.projectId = projectId;
    }

    const [datasets, systemConfig] = await Promise.all([
      db.dataset.findMany({
        where,
        include: {
          project: {
            select: {
              id: true,
              name: true,
              framework: true,
              status: true,
            },
          },
          user: {
            select: {
              id: true,
              username: true,
            },
          },
        },
        orderBy: {
          createdAt: 'desc',
        },
      }),
      db.systemConfig.findFirst(),
    ]);

    // resolvedDatasetDir tells the UI where a relative datasetDir actually
    // points — it is the OWNER's folder, which matters when an admin browses
    // another user's registered datasets.
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;
    const data = datasets.map((d) => ({
      ...d,
      resolvedDatasetDir: resolveDatasetRoot(d.datasetDir, userDatabasePath, d.user?.username),
    }));

    return NextResponse.json({
      success: true,
      data,
    });
  } catch (error) {
    console.error('Error fetching datasets:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch datasets',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// POST /api/datasets - Create a new dataset
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId } = auth;
    
    const body = await request.json();

    // Validate required fields
    if (!body.name || !body.projectId) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields',
          message: 'name and projectId are required',
        },
        { status: 400 }
      );
    }

    // Check if project exists and user has access
    const project = await db.project.findFirst({
      where: { 
        id: body.projectId,
        userId: userId,
      },
    });

    if (!project) {
      return NextResponse.json(
        {
          success: false,
          error: 'Project not found or access denied',
          message: `Project with id ${body.projectId} does not exist or you don't have access`,
        },
        { status: 404 }
      );
    }

    // Get system config for framework path and userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = getWorkDir((project as any).framework, systemConfig);
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;
    const owner = await db.user.findUnique({ where: { id: userId }, select: { username: true } });
    if (!owner) return NextResponse.json({ error: 'User not found' }, { status: 404 });
    body.username = owner.username;

    const framework = (project as any).framework || 'PaddleDetection';

    // Parse train annotation file if provided to get actual stats
    let numClasses = body.numClasses || 1;
    let numAnnotations = body.numAnnotations || 0;
    let numTrainImages = body.numTrainImages || 0;
    let numEvalImages = body.numEvalImages || 0;
    let classStats: { train: any[]; eval: any[] } = { train: [], eval: [] };

    // Resolve the absolute dataset directory. Absolute paths (typical for
    // PaddleSeg, e.g. H:\HAMRBackside\format) are used as-is; relative paths
    // are resolved under userDatabasePath/<username> (the COCO upload convention).
    let absoluteDatasetDir = body.datasetDir || '';
    if (body.datasetDir && !path.isAbsolute(body.datasetDir) && userDatabasePath && body.username) {
      absoluteDatasetDir = path.join(userDatabasePath, body.username, body.datasetDir);
    }
    if ((body.format || 'COCO') === 'COCO' && absoluteDatasetDir) {
      Object.assign(body, resolveCocoValidationPaths(absoluteDatasetDir, body));
    }

    if (isAnomaly(framework)) {
      // No annotations to parse: an anomaly dataset is just folders of images.
      // Count them so the list view is honest, and fail loudly when the required
      // normal directory is missing — a silently empty dataset would train a
      // model on nothing and report AUROC 0.5.
      const countImages = (rel: string | null | undefined): number => {
        if (!rel) return 0;
        const dir = path.isAbsolute(rel) ? rel : path.join(absoluteDatasetDir, rel);
        try {
          if (!fs.existsSync(dir)) {
            console.error(`[Dataset Import] AnomalyFolder directory not found: ${dir}`);
            return 0;
          }
          return fs
            .readdirSync(dir, { withFileTypes: true })
            .filter((e) => e.isFile() && ANOMALY_IMAGE_EXTENSIONS.includes(path.extname(e.name).toLowerCase()))
            .length;
        } catch (e) {
          console.error(`[Dataset Import] Failed to read ${dir}:`, e);
          return 0;
        }
      };

      const normalDir = body.normalDir || 'train_good';
      const normalCount = countImages(normalDir);
      if (normalCount === 0) {
        return NextResponse.json(
          {
            error: 'No normal images found',
            message:
              `Expected images in ${path.join(absoluteDatasetDir, normalDir)}. ` +
              `An anomaly-detection dataset trains on normal images only, so this directory cannot be empty.`,
          },
          { status: 400 }
        );
      }

      const abnormalCount = countImages(body.abnormalDir);
      const maskCount = countImages(body.maskDir);
      numTrainImages = normalCount;
      numEvalImages = countImages(body.normalTestDir) + abnormalCount;
      // Two "classes" in the anomaly sense: normal and anomalous.
      numClasses = 2;
      numAnnotations = maskCount;
      classStats = {
        train: [{ id: 0, name: 'normal', count: normalCount, imageCount: normalCount }],
        eval: [{ id: 1, name: 'anomalous', count: abnormalCount, imageCount: abnormalCount }],
      };

      if (body.maskDir && maskCount !== abnormalCount) {
        console.warn(
          `[Dataset Import] AnomalyFolder "${body.name}": ${abnormalCount} defect images but ${maskCount} masks. ` +
            'Pixel metrics need one mask per defect image, matched by file name.'
        );
      }
      console.log(
        `[Dataset Import] AnomalyFolder "${body.name}": root=${absoluteDatasetDir}, ` +
          `normal=${normalCount}, abnormal=${abnormalCount}, masks=${maskCount}`
      );
    } else if (isSegmentation(framework)) {
      // PaddleSeg datasets use list files (train.txt/val.txt) + num_classes.
      // There is no COCO JSON to parse; derive stats from the list files and
      // the optional class names file instead.
      const countListLines = (listRel: string | null | undefined): number => {
        if (!listRel) return 0;
        const listPath = path.isAbsolute(listRel) ? listRel : path.join(absoluteDatasetDir, listRel);
        try {
          if (!fs.existsSync(listPath)) {
            console.error(`[Dataset Import] PaddleSeg list file not found: ${listPath}`);
            return 0;
          }
          return fs.readFileSync(listPath, 'utf-8').split(/\r?\n/).filter(l => l.trim() !== '').length;
        } catch (e) {
          console.error(`[Dataset Import] Failed to read list file ${listPath}:`, e);
          return 0;
        }
      };

      numTrainImages = countListLines(body.trainAnnoPath);
      numEvalImages = countListLines(body.evalAnnoPath);

      // Prefer a class names file (class_names.txt / labels.txt) when present.
      let classNames: string[] = [];
      for (const candidate of ['class_names.txt', 'labels.txt']) {
        const cnPath = path.join(absoluteDatasetDir, candidate);
        if (fs.existsSync(cnPath)) {
          classNames = fs.readFileSync(cnPath, 'utf-8').split(/\r?\n/).map(l => l.trim()).filter(Boolean);
          break;
        }
      }
      if (classNames.length > 0) {
        numClasses = classNames.length;
        classStats = { train: classNames.map((name, id) => ({ id, name, count: 0, imageCount: 0 })), eval: [] };
      } else {
        numClasses = body.numClasses || numClasses;
      }

      console.log(`[Dataset Import] PaddleSeg dataset "${body.name}": root=${absoluteDatasetDir}, train=${numTrainImages}, val=${numEvalImages}, numClasses=${numClasses}`);

      // Persist class names JSON so monitoring/validation can resolve labels.
      if (classNames.length > 0 && userDatabasePath && body.username) {
        try {
          const datasetsDir = path.join(userDatabasePath, body.username, 'datasets');
          if (!fs.existsSync(datasetsDir)) fs.mkdirSync(datasetsDir, { recursive: true });
          fs.writeFileSync(
            path.join(datasetsDir, `${body.name}_classes.json`),
            JSON.stringify({ datasetName: body.name, numClasses, classNames, createdAt: new Date().toISOString() }, null, 2),
            'utf-8'
          );
        } catch (e) {
          console.error('[Dataset Import] Failed to save PaddleSeg class names:', e);
        }
      }
    } else if (body.trainAnnoPath && absoluteDatasetDir) {
      const trainAnnoFullPath = path.isAbsolute(body.trainAnnoPath) 
        ? body.trainAnnoPath 
        : path.join(absoluteDatasetDir, body.trainAnnoPath);
      
      console.log(`[Dataset Import] Reading train annotation from: ${trainAnnoFullPath}`);
      
      const cocoData = await parseCocoFile(trainAnnoFullPath);
      if (cocoData) {
        const stats = calculateStats(cocoData);
        numClasses = stats.numClasses;
        numAnnotations = stats.numAnnotations;
        numTrainImages = stats.numImages;
        classStats = { train: stats.classStats, eval: [] };
        
        // Save class names JSON to datasets folder
        if (userDatabasePath && body.username) {
          const datasetsDir = path.join(userDatabasePath, body.username, 'datasets');
          if (!fs.existsSync(datasetsDir)) {
            fs.mkdirSync(datasetsDir, { recursive: true });
          }
          
          const classNamesFile = path.join(datasetsDir, `${body.name}_classes.json`);
          const classNamesData = {
            datasetName: body.name,
            numClasses: stats.numClasses,
            classNames: stats.classStats.map((c: any) => c.name),
            classDetails: stats.classStats,
            createdAt: new Date().toISOString(),
          };
          
          fs.writeFileSync(classNamesFile, JSON.stringify(classNamesData, null, 2), 'utf-8');
          console.log(`[Dataset Import] Saved class names to: ${classNamesFile}`);
        }
        
        // Parse eval annotation if provided
        if (body.evalAnnoPath) {
          const evalAnnoFullPath = path.isAbsolute(body.evalAnnoPath)
            ? body.evalAnnoPath
            : path.join(absoluteDatasetDir, body.evalAnnoPath);
          
          console.log(`[Dataset Import] Reading eval annotation from: ${evalAnnoFullPath}`);
          
          const evalData = await parseCocoFile(evalAnnoFullPath);
          if (evalData) {
            const evalStats = calculateStats(evalData);
            numEvalImages = evalStats.numImages;
            classStats.eval = evalStats.classStats;
          } else {
            return NextResponse.json({ success: false, error: `Failed to parse eval annotations: ${evalAnnoFullPath}` }, { status: 400 });
          }
        }
      } else {
        return NextResponse.json({ success: false, error: `Failed to parse train annotations: ${trainAnnoFullPath}` }, { status: 400 });
      }
    } else {
      console.log(`[Dataset Import] Skipping annotation parsing - missing required fields:`, {
        hasTrainAnnoPath: !!body.trainAnnoPath,
        hasUserDatabasePath: !!userDatabasePath,
        hasDatasetDir: !!body.datasetDir,
        username: body.username,
      });
    }

    const datasetData = {
      name: body.name,
      format: body.format || 'COCO',
      numClasses: numClasses,
      datasetDir: absoluteDatasetDir,
      trainImagePath: body.trainImagePath,
      trainAnnoPath: body.trainAnnoPath,
      evalImagePath: body.evalImagePath,
      evalAnnoPath: body.evalAnnoPath,
      normalDir: isAnomaly(framework) ? body.normalDir : null,
      normalTestDir: isAnomaly(framework) ? body.normalTestDir : null,
      abnormalDir: isAnomaly(framework) ? body.abnormalDir : null,
      maskDir: isAnomaly(framework) ? body.maskDir : null,
      task: (project as any).task || 'detection',
      framework: (project as any).framework || 'PaddleDetection',
    };
    
    const yamlConfig = generateDatasetYaml(datasetData);

    // Create dataset in database with userId
    const dataset = await db.dataset.create({
      data: {
        name: body.name,
        description: body.description || null,
        projectId: body.projectId,
        userId: userId,
        format: body.format || 'COCO',
        trainImagePath: body.trainImagePath || null,
        trainAnnoPath: body.trainAnnoPath || null,
        evalImagePath: body.evalImagePath || null,
        evalAnnoPath: body.evalAnnoPath || null,
        // The form posts the whole state object, so these arrive even for COCO
        // projects (with their placeholder defaults). Only keep them where they
        // mean something, so the row does not claim a layout it does not have.
        normalDir: (isAnomaly(framework) && body.normalDir) || null,
        normalTestDir: (isAnomaly(framework) && body.normalTestDir) || null,
        abnormalDir: (isAnomaly(framework) && body.abnormalDir) || null,
        maskDir: (isAnomaly(framework) && body.maskDir) || null,
        // Anomaly datasets, like PaddleSeg ones, are addressed by absolute root:
        // the generated config hands `root` straight to anomalib.
        datasetDir: (isSegmentation(framework) || isAnomaly(framework) ? absoluteDatasetDir : body.datasetDir) || null,
        numClasses: numClasses,
        numAnnotations: numAnnotations,
        numTrainImages: numTrainImages,
        numEvalImages: numEvalImages,
        classStats: JSON.stringify(classStats),
        yamlConfig: yamlConfig,
      },
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
          },
        },
      },
    });

    // Save YAML file to {userConfigsPath}/{username}/datasets directory
    if (userConfigsPath && body.username) {
      const datasetName = body.name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_.-]/g, '').toLowerCase();
      const configFileName = `${datasetName}.yml`;
      const datasetsConfigDir = path.join(userConfigsPath, body.username, 'datasets');
      
      if (!fs.existsSync(datasetsConfigDir)) {
        fs.mkdirSync(datasetsConfigDir, { recursive: true });
      }
      
      const configFilePath = path.join(datasetsConfigDir, configFileName);
      fs.writeFileSync(configFilePath, yamlConfig, 'utf-8');
      console.log(`[Dataset] Saved config to: ${configFilePath}`);
    }

    // Log activity
    await logActivity(userId, {
      action: 'import_dataset',
      entityType: 'dataset',
      entityId: dataset.id,
      entityName: dataset.name,
      details: { projectId: body.projectId, projectName: project.name },
    });

    return NextResponse.json(
      {
        success: true,
        data: dataset,
        message: 'Dataset created successfully',
      },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error creating dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to create dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
