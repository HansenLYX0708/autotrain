'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import {
  BookOpen,
  ChevronLeft,
  FolderKanban,
  Database,
  Cpu,
  PlayCircle,
  ListTodo,
  Activity,
  CheckCircle2,
  Settings,
  Users,
  PencilRuler,
  ArrowRight,
  FileText,
  Upload,
  Download,
  RefreshCw,
  Eye,
  Trash2,
  Edit,
  Plus,
  Search,
  Filter,
  AlertCircle,
  CheckCircle,
  Info,
} from 'lucide-react'

interface UserManualPageProps {
  onBack?: () => void
}

interface StepSection {
  id: string
  title: string
  icon: React.ElementType
  description: string
  substeps: {
    title: string
    content: React.ReactNode
  }[]
}

const manualSections: StepSection[] = [
  {
    id: 'overview',
    title: 'Platform Overview',
    icon: BookOpen,
    description: 'Introduction to the Hawkeye+ AutoTrain Platform',
    substeps: [
      {
        title: 'System Introduction',
        content: (
          <div className="space-y-3">
            <p>
              Hawkeye+ is a comprehensive deep learning training platform built on PaddleDetection.
              It provides an end-to-end workflow for object detection model training, from dataset
              management to model deployment.
            </p>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <Card className="bg-muted/50">
                <CardContent className="p-4">
                  <h4 className="font-semibold flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Key Features
                  </h4>
                  <ul className="text-sm mt-2 space-y-1 list-disc list-inside">
                    <li>Multi-project management</li>
                    <li>Dataset format conversion (Labelme ↔ COCO)</li>
                    <li>Visual model configuration</li>
                    <li>Distributed training support</li>
                    <li>Real-time monitoring</li>
                  </ul>
                </CardContent>
              </Card>
              <Card className="bg-muted/50">
                <CardContent className="p-4">
                  <h4 className="font-semibold flex items-center gap-2">
                    <Info className="w-4 h-4 text-blue-500" />
                    System Requirements
                  </h4>
                  <ul className="text-sm mt-2 space-y-1 list-disc list-inside">
                    <li>NVIDIA GPU with CUDA support</li>
                    <li>PaddleDetection framework</li>
                    <li>Python 3.8+</li>
                    <li>Modern web browser</li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        ),
      },
      {
        title: 'Navigation Structure',
        content: (
          <div className="space-y-3">
            <p>The platform is organized into the following main modules:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { icon: FolderKanban, name: 'Projects', desc: 'Create and manage training projects' },
                { icon: Database, name: 'Datasets', desc: 'Import, convert, and manage datasets' },
                { icon: Cpu, name: 'Models', desc: 'Configure model architectures' },
                { icon: PlayCircle, name: 'Configurations', desc: 'Define training parameters' },
                { icon: ListTodo, name: 'Jobs', desc: 'Submit and manage training jobs' },
                { icon: Activity, name: 'Monitoring', desc: 'Monitor training progress in real-time' },
                { icon: CheckCircle2, name: 'Validation', desc: 'Evaluate model performance' },
                { icon: Settings, name: 'Settings', desc: 'System configuration' },
              ].map((item, idx) => (
                <div key={idx} className="flex items-start gap-3 p-3 border rounded-lg">
                  <item.icon className="w-5 h-5 text-primary mt-0.5" />
                  <div>
                    <span className="font-medium">{item.name}</span>
                    <p className="text-sm text-muted-foreground">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ),
      },
    ],
  },
  {
    id: 'projects',
    title: 'Step 1: Project Management',
    icon: FolderKanban,
    description: 'Create and organize your training projects',
    substeps: [
      {
        title: '1.1 Create a New Project',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>
                Navigate to the <strong>Projects</strong> page from the sidebar
              </li>
              <li>
                Click the <strong>"New Project"</strong> button in the top-right corner
              </li>
              <li>
                Enter a <strong>Project Name</strong> (required, unique identifier)
              </li>
              <li>
                Add an optional <strong>Description</strong> to document project details
              </li>
              <li>
                Click <strong>"Create"</strong> to finalize
              </li>
            </ol>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mt-3">
              <p className="text-sm text-amber-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                <strong>Tip:</strong> Use descriptive names like "DefectDetection_Q2_2024" for better organization.
              </p>
            </div>
          </div>
        ),
      },
      {
        title: '1.2 Manage Existing Projects',
        content: (
          <div className="space-y-3">
            <p>From the Projects page, you can perform the following actions:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>
                <strong>Search:</strong> Use the search bar to quickly find projects by name
              </li>
              <li>
                <strong>Edit:</strong> Click the edit icon to modify project name and description
              </li>
              <li>
                <strong>Delete:</strong> Remove a project (only if no datasets/models are associated)
              </li>
              <li>
                <strong>View Details:</strong> Click on a project card to see associated resources
              </li>
            </ul>
            <div className="grid grid-cols-3 gap-3 mt-4">
              <div className="flex items-center gap-2 p-2 bg-muted rounded">
                <Search className="w-4 h-4" />
                <span className="text-sm">Quick Search</span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-muted rounded">
                <Edit className="w-4 h-4" />
                <span className="text-sm">Edit Project</span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-muted rounded">
                <Trash2 className="w-4 h-4" />
                <span className="text-sm">Delete Project</span>
              </div>
            </div>
          </div>
        ),
      },
      {
        title: '1.3 Project Best Practices',
        content: (
          <div className="space-y-3">
            <ul className="list-disc list-inside space-y-2">
              <li>
                <strong>Naming Convention:</strong> Use consistent naming patterns including version or date
              </li>
              <li>
                <strong>Documentation:</strong> Always add descriptions explaining the project purpose
              </li>
              <li>
                <strong>Organization:</strong> Group related datasets and models under the same project
              </li>
              <li>
                <strong>Lifecycle:</strong> Archive old projects rather than deleting them
              </li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'datasets',
    title: 'Step 2: Dataset Management',
    icon: Database,
    description: 'Prepare and manage your training data',
    substeps: [
      {
        title: '2.1 Supported Dataset Formats',
        content: (
          <div className="space-y-3">
            <p>The platform supports two primary annotation formats:</p>
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">COCO Format</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Standard JSON format with images, annotations, and categories. 
                    Required structure: train/val images + annotation JSON files.
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Labelme Format</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Individual JSON files per image. Structure: imgs folder + jsons folder.
                    Can be converted to COCO format within the platform.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        ),
      },
      {
        title: '2.2 Upload Dataset (Chunked Upload)',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>Datasets</strong> page</li>
              <li>Click <strong>"Upload Dataset"</strong> button</li>
              <li>Select format: <strong>COCO</strong> or <strong>Labelme</strong></li>
              <li>Enter a unique <strong>Dataset Name</strong></li>
              <li>
                Click <strong>"Select Folder"</strong> and choose your dataset folder
                <ul className="list-disc list-inside ml-6 mt-1 text-sm text-muted-foreground">
                  <li>For COCO: Select folder containing train/val subdirectories</li>
                  <li>For Labelme: Select folder containing imgs/ and jsons/ folders</li>
                </ul>
              </li>
              <li>Click <strong>"Start Upload"</strong> to begin chunked transfer</li>
              <li>Monitor progress; upload supports resume if interrupted</li>
            </ol>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mt-3">
              <p className="text-sm text-blue-800 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Large datasets are automatically split into chunks for reliable transfer.
              </p>
            </div>
          </div>
        ),
      },
      {
        title: '2.3 Import Existing Dataset',
        content: (
          <div className="space-y-3">
            <p>For datasets already on the server:</p>
            <ol className="list-decimal list-inside space-y-2">
              <li>Click <strong>"Import Dataset"</strong> button</li>
              <li>Select the <strong>Project</strong> to associate with</li>
              <li>Choose from <strong>Available Datasets</strong> list (auto-detected)</li>
              <li>Verify auto-populated paths or manually enter:
                <ul className="list-disc list-inside ml-6 mt-1 text-sm">
                  <li>Training images path (e.g., <code>data/train</code>)</li>
                  <li>Training annotations path (e.g., <code>annotations/instances_train.json</code>)</li>
                  <li>Validation images path (e.g., <code>data/val</code>)</li>
                  <li>Validation annotations path</li>
                </ul>
              </li>
              <li>Click <strong>"Import"</strong> to register the dataset</li>
            </ol>
          </div>
        ),
      },
      {
        title: '2.4 Convert Labelme to COCO',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Click <strong>"Labelme → COCO"</strong> button</li>
              <li>Select the <strong>Labelme dataset</strong> from the dropdown</li>
              <li>
                Paths auto-fill based on selection:
                <ul className="list-disc list-inside ml-6 mt-1 text-sm">
                  <li>Images path: <code>/path/to/data/imgs</code></li>
                  <li>Annotations path: <code>/path/to/data/jsons</code></li>
                </ul>
              </li>
              <li>Enter the <strong>output dataset name</strong> for the COCO format result</li>
              <li>
                Configure train/val/test split ratios (must sum to 1.0):
                <ul className="list-disc list-inside ml-6 mt-1 text-sm">
                  <li>Train: 0.7 (70%)</li>
                  <li>Validation: 0.2 (20%)</li>
                  <li>Test: 0.1 (10%)</li>
                </ul>
              </li>
              <li>Click <strong>"Convert"</strong> to start conversion</li>
            </ol>
          </div>
        ),
      },
      {
        title: '2.5 Dataset Preview and Statistics',
        content: (
          <div className="space-y-3">
            <p>After importing, you can analyze your dataset:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>
                <strong>View Statistics:</strong> Click on a dataset to see annotation counts, 
                class distribution charts, and image counts
              </li>
              <li>
                <strong>Preview Images:</strong> Click <strong>"Preview"</strong> to view sample images 
                with bounding box annotations overlaid
              </li>
              <li>
                <strong>Parse Dataset:</strong> Click <strong>"Parse"</strong> to refresh statistics 
                and verify annotation integrity
              </li>
              <li>
                <strong>Filter by Class:</strong> In preview mode, filter images by specific categories
              </li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'models',
    title: 'Step 3: Model Configuration',
    icon: Cpu,
    description: 'Define and configure model architectures',
    substeps: [
      {
        title: '3.1 Create Model Configuration',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>Models</strong> page</li>
              <li>Click <strong>"Import Model"</strong> (for YAML configs) or configure manually</li>
              <li>Select the target <strong>Project</strong></li>
              <li>Enter a <strong>Model Name</strong></li>
              <li>
                Choose configuration source:
                <ul className="list-disc list-inside ml-6 mt-1 text-sm">
                  <li><strong>Default Configs:</strong> Pre-defined templates from PaddleDetection</li>
                  <li><strong>User Configs:</strong> Your previously saved configurations</li>
                  <li><strong>Custom YAML:</strong> Paste your own YAML configuration</li>
                </ul>
              </li>
              <li>Click <strong>"Import Model"</strong> to save</li>
            </ol>
          </div>
        ),
      },
      {
        title: '3.2 Architecture Components',
        content: (
          <div className="space-y-3">
            <p>Configure the three main architecture components:</p>
            <div className="space-y-3">
              <Card className="border-l-4 border-l-blue-500">
                <CardContent className="p-4">
                  <h4 className="font-semibold">Backbone (Feature Extractor)</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Options: CSPResNet, MobileNetV3, ResNet. Affects speed/accuracy trade-off.
                  </p>
                </CardContent>
              </Card>
              <Card className="border-l-4 border-l-green-500">
                <CardContent className="p-4">
                  <h4 className="font-semibold">Neck (Feature Fusion)</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Options: CustomCSPPAN, FPN, YOLOv3FPN. Combines multi-scale features.
                  </p>
                </CardContent>
              </Card>
              <Card className="border-l-4 border-l-purple-500">
                <CardContent className="p-4">
                  <h4 className="font-semibold">Head (Detection Head)</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Options: PPYOLOEHead, RetinaHead, DINOHead. Performs final predictions.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        ),
      },
      {
        title: '3.3 Advanced Parameters',
        content: (
          <div className="space-y-3">
            <p>Fine-tune model behavior with these parameters:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>
                <strong>Normalization Type:</strong> sync_bn (multi-GPU), bn (single GPU), gn (group norm)
              </li>
              <li>
                <strong>EMA (Exponential Moving Average):</strong> Enable for more stable training
              </li>
              <li>
                <strong>Depth/Width Multiplier:</strong> Scale model size (0.33 = small, 1.0 = full)
              </li>
              <li>
                <strong>Pretrained Weights:</strong> URL to pretrained checkpoint for transfer learning
              </li>
              <li>
                <strong>Number of Classes:</strong> Must match your dataset's class count
              </li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'training',
    title: 'Step 4: Training Configuration',
    icon: PlayCircle,
    description: 'Set up training parameters and hyperparameters',
    substeps: [
      {
        title: '4.1 Create Training Config',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>Configurations</strong> page</li>
              <li>Click <strong>"New Configuration"</strong></li>
              <li>Select <strong>Project</strong>, <strong>Dataset</strong>, and <strong>Model</strong></li>
              <li>Enter a <strong>Configuration Name</strong></li>
              <li>Set the base <strong>Configuration File</strong> (optimizer, scheduler settings)</li>
              <li>Click <strong>"Create Configuration"</strong></li>
            </ol>
          </div>
        ),
      },
      {
        title: '4.2 Configure Training Parameters',
        content: (
          <div className="space-y-3">
            <p>Edit the configuration to set training parameters:</p>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-sm">Basic Settings</h4>
                <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                  <li>Epochs: Total training iterations</li>
                  <li>Batch Size: Images per training step</li>
                  <li>Learning Rate: Initial LR for optimizer</li>
                  <li>Image Size: Input resolution (e.g., 640×640)</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm">Advanced Settings</h4>
                <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                  <li>Optimizer: SGD, Adam, AdamW</li>
                  <li>Scheduler: Piecewise, Cosine Annealing</li>
                  <li>Warmup: Steps for LR warmup</li>
                  <li>Augmentations: Mosaic, MixUp, CutMix</li>
                </ul>
              </div>
            </div>
          </div>
        ),
      },
      {
        title: '4.3 GPU Assignment Strategy',
        content: (
          <div className="space-y-3">
            <p>Configure multi-GPU training with flexible assignment:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>
                <strong>Auto (Recommended):</strong> System automatically assigns available GPUs
              </li>
              <li>
                <strong>Manual:</strong> Specify GPU indices (e.g., "0,1,2" for GPUs 0, 1, and 2)
              </li>
              <li>
                <strong>Smart Exclusion:</strong> System excludes GPUs that are:
                <ul className="list-disc list-inside ml-6 text-sm text-muted-foreground">
                  <li>Currently occupied by other training jobs</li>
                  <li>Have insufficient memory</li>
                  <li>Below compute capability requirements</li>
                </ul>
              </li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'jobs',
    title: 'Step 5: Training Jobs',
    icon: ListTodo,
    description: 'Submit and manage training jobs',
    substeps: [
      {
        title: '5.1 Submit Training Job',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>Jobs</strong> page</li>
              <li>Click <strong>"New Job"</strong> button</li>
              <li>Enter a <strong>Job Name</strong> (descriptive identifier)</li>
              <li>Select the <strong>Training Configuration</strong> to use</li>
              <li>
                Choose <strong>GPU Strategy</strong>:
                <ul className="list-disc list-inside ml-6 text-sm">
                  <li>Auto: Let system pick available GPUs</li>
                  <li>Manual: Specify GPU IDs explicitly</li>
                </ul>
              </li>
              <li>Review configuration summary</li>
              <li>Click <strong>"Start Training"</strong> to submit</li>
            </ol>
          </div>
        ),
      },
      {
        title: '5.2 Job Lifecycle Management',
        content: (
          <div className="space-y-3">
            <p>Monitor and control training jobs:</p>
            <div className="flex flex-wrap gap-2 mb-3">
              <Badge variant="outline" className="bg-yellow-100 text-yellow-800">Pending</Badge>
              <Badge variant="outline" className="bg-blue-100 text-blue-800">Running</Badge>
              <Badge variant="outline" className="bg-green-100 text-green-800">Completed</Badge>
              <Badge variant="outline" className="bg-red-100 text-red-800">Failed</Badge>
              <Badge variant="outline" className="bg-gray-100 text-gray-800">Stopped</Badge>
            </div>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Start:</strong> Begin a pending job</li>
              <li><strong>Stop:</strong> Gracefully terminate a running job</li>
              <li><strong>Restart:</strong> Resume from last checkpoint</li>
              <li><strong>Delete:</strong> Remove job record (preserves outputs)</li>
              <li><strong>View Log:</strong> Access real-time training logs</li>
            </ul>
          </div>
        ),
      },
      {
        title: '5.3 Export Trained Model',
        content: (
          <div className="space-y-3">
            <p>After training completes, export models for deployment:</p>
            <ol className="list-decimal list-inside space-y-2">
              <li>Click on a <strong>Completed</strong> job</li>
              <li>Click <strong>"Export Model"</strong> in the details panel</li>
              <li>Select export format:
                <ul className="list-disc list-inside ml-6 text-sm">
                  <li>Paddle Inference (default)</li>
                  <li>ONNX (for cross-platform deployment)</li>
                </ul>
              </li>
              <li>Choose output directory</li>
              <li>Click <strong>"Export"</strong> to generate deployment files</li>
            </ol>
          </div>
        ),
      },
    ],
  },
  {
    id: 'monitoring',
    title: 'Step 6: Training Monitoring',
    icon: Activity,
    description: 'Monitor training progress and resource usage',
    substeps: [
      {
        title: '6.1 Real-time Metrics',
        content: (
          <div className="space-y-3">
            <p>The Monitoring page displays live training metrics:</p>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Loss Curves:</strong> Training and validation loss over time</li>
              <li><strong>Learning Rate:</strong> LR schedule visualization</li>
              <li><strong>mAP Metrics:</strong> Mean Average Precision for object detection</li>
              <li><strong>Speed:</strong> Iterations per second, estimated completion time</li>
              <li><strong>GPU Utilization:</strong> Memory and compute usage per GPU</li>
            </ul>
          </div>
        ),
      },
      {
        title: '6.2 Log Analysis',
        content: (
          <div className="space-y-3">
            <p>Access detailed training logs:</p>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Live Log:</strong> Auto-updating training output</li>
              <li><strong>Log Files:</strong> Download complete training logs</li>
              <li><strong>Error Detection:</strong> Automatic highlighting of error messages</li>
              <li><strong>Search:</strong> Find specific events in log history</li>
            </ul>
          </div>
        ),
      },
      {
        title: '6.3 Checkpoint Management',
        content: (
          <div className="space-y-3">
            <p>Manage training checkpoints:</p>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Auto-save:</strong> Checkpoints saved every N epochs (configurable)</li>
              <li><strong>Best Model:</strong> Automatically tracks highest mAP checkpoint</li>
              <li><strong>Resume:</strong> Continue training from any saved checkpoint</li>
              <li><strong>Cleanup:</strong> Remove old checkpoints to save disk space</li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'validation',
    title: 'Step 7: Model Validation',
    icon: CheckCircle2,
    description: 'Evaluate model performance on test data',
    substeps: [
      {
        title: '7.1 Run Validation',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>Validation</strong> page</li>
              <li>Click <strong>"New Validation"</strong></li>
              <li>Select the <strong>Dataset</strong> for evaluation</li>
              <li>Choose the <strong>Model/Checkpoint</strong> to validate</li>
              <li>Configure validation parameters:
                <ul className="list-disc list-inside ml-6 text-sm">
                  <li>Confidence threshold (default: 0.5)</li>
                  <li>IoU threshold for NMS (default: 0.5)</li>
                  <li>Image size for inference</li>
                </ul>
              </li>
              <li>Click <strong>"Start Validation"</strong></li>
            </ol>
          </div>
        ),
      },
      {
        title: '7.2 Analyze Results',
        content: (
          <div className="space-y-3">
            <p>Review validation metrics and visualizations:</p>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>mAP@0.5:</strong> Mean Average Precision at IoU=0.5</li>
              <li><strong>mAP@0.5:0.95:</strong> COCO-style mAP across IoU thresholds</li>
              <li><strong>Per-class AP:</strong> Individual class performance</li>
              <li><strong>PR Curves:</strong> Precision-Recall curves per class</li>
              <li><strong>Confusion Matrix:</strong> Detection confusion analysis</li>
              <li><strong>Sample Visualizations:</strong> Annotated validation images</li>
            </ul>
          </div>
        ),
      },
      {
        title: '7.3 Batch Validation',
        content: (
          <div className="space-y-3">
            <p>Compare multiple models:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>Select multiple jobs/configurations</li>
              <li>Run validation on same test set</li>
              <li>Side-by-side metric comparison</li>
              <li>Export comparison report as CSV/PDF</li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'annotation',
    title: 'Step 8: Data Annotation (Admin)',
    icon: PencilRuler,
    description: 'Built-in annotation tool for dataset preparation',
    substeps: [
      {
        title: '8.1 Access Annotation Tool',
        content: (
          <div className="space-y-3">
            <p><strong>Note:</strong> Annotation feature is admin-only.</p>
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>Annotation</strong> page</li>
              <li>Select the <strong>Dataset</strong> to annotate</li>
              <li>Choose annotation mode:
                <ul className="list-disc list-inside ml-6 text-sm">
                  <li>Create new annotations</li>
                  <li>Edit existing annotations</li>
                  <li>Review and validate</li>
                </ul>
              </li>
            </ol>
          </div>
        ),
      },
      {
        title: '8.2 Annotation Workflow',
        content: (
          <div className="space-y-3">
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Select Class:</strong> Choose from predefined categories</li>
              <li><strong>Draw Bounding Box:</strong> Click and drag to create rectangles</li>
              <li><strong>Edit:</strong> Resize by dragging edges, move by dragging center</li>
              <li><strong>Delete:</strong> Select box and press Delete key</li>
              <li><strong>Navigate:</strong> Arrow keys or buttons for next/previous image</li>
              <li><strong>Save:</strong> Auto-saves every 30 seconds, manual save available</li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'users',
    title: 'Step 9: User Management (Admin)',
    icon: Users,
    description: 'Manage platform users and permissions',
    substeps: [
      {
        title: '9.1 Create Users',
        content: (
          <div className="space-y-3">
            <ol className="list-decimal list-inside space-y-2">
              <li>Navigate to <strong>User Management</strong> page</li>
              <li>Click <strong>"New User"</strong></li>
              <li>Enter <strong>Username</strong> (unique, no spaces)</li>
              <li>Set temporary <strong>Password</strong> (user must change on first login)</li>
              <li>Select <strong>Role</strong>:
                <ul className="list-disc list-inside ml-6 text-sm">
                  <li><strong>Admin:</strong> Full access to all features</li>
                  <li><strong>User:</strong> Standard training workflow access</li>
                </ul>
              </li>
              <li>Click <strong>"Create User"</strong></li>
            </ol>
          </div>
        ),
      },
      {
        title: '9.2 Manage Users',
        content: (
          <div className="space-y-3">
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Edit:</strong> Modify user details and role</li>
              <li><strong>Reset Password:</strong> Generate new temporary password</li>
              <li><strong>Deactivate:</strong> Temporarily disable account</li>
              <li><strong>Delete:</strong> Permanently remove user (requires confirmation)</li>
              <li><strong>Activity Log:</strong> View user's job and project history</li>
            </ul>
          </div>
        ),
      },
    ],
  },
  {
    id: 'settings',
    title: 'Step 10: System Settings (Admin)',
    icon: Settings,
    description: 'Configure platform-wide settings (Admin only)',
    substeps: [
      {
        title: '10.1 Basic Settings',
        content: (
          <div className="space-y-3">
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Python Path:</strong> Path to Python interpreter with PaddleDetection</li>
              <li><strong>Data Directory:</strong> Root directory for datasets and outputs</li>
              <li><strong>Default Framework:</strong> Set default deep learning framework</li>
              <li><strong>Theme:</strong> Light/Dark mode preference</li>
            </ul>
          </div>
        ),
      },
      {
        title: '10.2 Advanced Settings',
        content: (
          <div className="space-y-3">
            <ul className="list-disc list-inside space-y-2">
              <li><strong>GPU Memory Threshold:</strong> Define when GPU is considered "occupied"</li>
              <li><strong>Max Concurrent Jobs:</strong> Limit simultaneous training jobs</li>
              <li><strong>Auto-cleanup:</strong> Automatically remove old logs/checkpoints</li>
              <li><strong>Notification:</strong> Configure email alerts for job completion/failure</li>
            </ul>
          </div>
        ),
      },
      {
        title: '10.3 Environment Check',
        content: (
          <div className="space-y-3">
            <p>Verify system health:</p>
            <ul className="list-disc list-inside space-y-2">
              <li>Click <strong>"Check Environment"</strong> in Settings</li>
              <li>Review component status:
                <ul className="list-disc list-inside ml-6 text-sm">
                  <li>Python environment with required packages</li>
                  <li>CUDA and GPU drivers</li>
                  <li>PaddleDetection installation</li>
                  <li>Disk space availability</li>
                </ul>
              </li>
              <li>Follow repair suggestions if issues detected</li>
            </ul>
          </div>
        ),
      },
    ],
  },
]

export function UserManualPage({ onBack }: UserManualPageProps) {
  const [selectedSection, setSelectedSection] = useState<string | null>(null)

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="shrink-0 mb-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            {onBack && (
              <Button variant="outline" size="sm" onClick={onBack}>
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back to App
              </Button>
            )}
            <div>
              <h1 className="text-2xl font-bold">User Manual</h1>
              <p className="text-sm text-muted-foreground">
                Hawkeye+ AutoTrain Platform - Complete Guide
              </p>
            </div>
          </div>
          <Badge variant="outline" className="text-xs">English Version</Badge>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 flex gap-6 overflow-hidden">
        {/* Sidebar - Main Steps */}
        <Card className="w-80 flex flex-col overflow-hidden">
          <CardHeader className="pb-3 shrink-0">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              Table of Contents
            </CardTitle>
          </CardHeader>

          <CardContent className="flex-1 flex flex-col p-0 min-h-0">
            <div className="flex-1 min-h-0 overflow-y-auto px-4 pb-4 scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-gray-300 scrollbar-track-transparent">
              <div className="space-y-1">
                {manualSections.map((section) => {
                  const Icon = section.icon
                  const isActive = selectedSection === section.id

                  return (
                    <button
                      key={section.id}
                      onClick={() => setSelectedSection(section.id)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-muted'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <Icon className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <div className="min-w-0">
                          <p className={`text-sm font-medium ${isActive ? 'text-primary-foreground' : ''}`}>
                            {section.title}
                          </p>
                          <p className={`text-xs mt-0.5 truncate ${
                            isActive ? 'text-primary-foreground/80' : 'text-muted-foreground'
                          }`}>
                            {section.description}
                          </p>
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Content Area - Detailed Steps */}
        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardContent className="flex-1 min-h-0 overflow-y-auto p-6">
            {selectedSection ? (
              <div className="max-w-3xl">
                {manualSections
                  .filter((s) => s.id === selectedSection)
                  .map((section) => {
                    const Icon = section.icon
                    return (
                      <div key={section.id}>
                        <div className="flex items-center gap-3 mb-4">
                          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                            <Icon className="w-5 h-5 text-primary" />
                          </div>
                          <div>
                            <h2 className="text-xl font-bold">{section.title}</h2>
                            <p className="text-sm text-muted-foreground">
                              {section.description}
                            </p>
                          </div>
                        </div>

                        <Separator className="my-4" />

                        <Accordion type="multiple" defaultValue={[section.substeps[0]?.title]} className="space-y-2">
                          {section.substeps.map((substep, idx) => (
                            <AccordionItem
                              key={idx}
                              value={substep.title}
                              className="border rounded-lg px-4 data-[state=open]:bg-muted/50"
                            >
                              <AccordionTrigger className="text-sm font-medium hover:no-underline py-3">
                                <div className="flex items-center gap-2">
                                  <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center">
                                    {idx + 1}
                                  </span>
                                  {substep.title}
                                </div>
                              </AccordionTrigger>
                              <AccordionContent className="pb-4 text-sm">
                                {substep.content}
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </div>
                    )
                  })}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <BookOpen className="w-16 h-16 text-muted-foreground/30 mb-4" />
                <h3 className="text-lg font-medium text-muted-foreground">
                  Select a section from the sidebar
                </h3>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  Click on any step to view detailed instructions
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>


  )
}
