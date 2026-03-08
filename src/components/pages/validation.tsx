'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  CheckCircle2, 
  Upload, 
  Play, 
  FileSearch,
  AlertCircle 
} from 'lucide-react'

export function ValidationPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Validation</h1>
          <p className="text-muted-foreground">
            Evaluate trained models and run inference
          </p>
        </div>
        <Button>
          <Play className="w-4 h-4 mr-2" />
          New Validation Job
        </Button>
      </div>

      {/* Placeholder Content */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5" />
              Model Evaluation
            </CardTitle>
            <CardDescription>
              Evaluate model performance on validation dataset
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="text-sm font-medium">Select Model</label>
                <p className="text-xs text-muted-foreground">
                  Choose a trained model to evaluate
                </p>
              </div>
              <Button variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Browse
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="text-sm font-medium">Validation Dataset</label>
                <p className="text-xs text-muted-foreground">
                  Dataset to use for evaluation
                </p>
              </div>
              <Button variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Browse
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileSearch className="w-5 h-5" />
              Inference
            </CardTitle>
            <CardDescription>
              Run inference on images or videos
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="text-sm font-medium">Input Source</label>
                <p className="text-xs text-muted-foreground">
                  Image, video, or folder path
                </p>
              </div>
              <Button variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Browse
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="text-sm font-medium">Output Directory</label>
                <p className="text-xs text-muted-foreground">
                  Where to save results
                </p>
              </div>
              <Button variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Browse
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Jobs */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Validation Jobs</CardTitle>
          <CardDescription>
            View and manage your validation jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">No validation jobs yet</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Run your first validation job to see results here
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
