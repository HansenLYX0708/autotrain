import { db } from '@/lib/db'

export type ActivityAction =
  | 'create_project'
  | 'update_project'
  | 'delete_project'
  | 'import_dataset'
  | 'update_dataset'
  | 'delete_dataset'
  | 'import_model'
  | 'update_model'
  | 'delete_model'
  | 'import_config'
  | 'update_config'
  | 'delete_config'
  | 'create_job'
  | 'start_training'
  | 'stop_training'
  | 'restart_training'
  | 'delete_job'
  | 'create_validation'
  | 'start_validation'
  | 'export_model'
  | 'login'
  | 'logout'

export interface ActivityLogData {
  action: ActivityAction
  entityType: string
  entityId?: string
  entityName?: string
  details?: Record<string, unknown>
}

// Helper to log user activity
export async function logActivity(userId: string, data: ActivityLogData) {
  try {
    await db.activityLog.create({
      data: {
        userId,
        action: data.action,
        entityType: data.entityType,
        entityId: data.entityId,
        entityName: data.entityName,
        details: data.details ? JSON.stringify(data.details) : null,
      },
    })
  } catch (error) {
    console.error('Failed to log activity:', error)
    // Don't throw - activity logging should not break main functionality
  }
}

// Get activity display name
export function getActivityDisplayName(action: ActivityAction): string {
  const displayNames: Record<ActivityAction, string> = {
    create_project: 'Created Project',
    update_project: 'Updated Project',
    delete_project: 'Deleted Project',
    import_dataset: 'Imported Dataset',
    update_dataset: 'Updated Dataset',
    delete_dataset: 'Deleted Dataset',
    import_model: 'Imported Model',
    update_model: 'Updated Model',
    delete_model: 'Deleted Model',
    import_config: 'Imported Config',
    update_config: 'Updated Config',
    delete_config: 'Deleted Config',
    create_job: 'Created Training Job',
    start_training: 'Started Training',
    stop_training: 'Stopped Training',
    restart_training: 'Restarted Training',
    delete_job: 'Deleted Job',
    create_validation: 'Created Validation',
    start_validation: 'Started Validation',
    export_model: 'Exported Model to TensorRT',
    login: 'Logged In',
    logout: 'Logged Out',
  }
  return displayNames[action] || action
}

// Get activity icon color
export function getActivityColor(action: ActivityAction): string {
  if (action.includes('create') || action.includes('import')) {
    return 'text-emerald-500 bg-emerald-50'
  }
  if (action.includes('update')) {
    return 'text-blue-500 bg-blue-50'
  }
  if (action.includes('delete')) {
    return 'text-red-500 bg-red-50'
  }
  if (action.includes('start')) {
    return 'text-green-500 bg-green-50'
  }
  if (action.includes('stop')) {
    return 'text-orange-500 bg-orange-50'
  }
  return 'text-gray-500 bg-gray-50'
}
