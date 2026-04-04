import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { getCurrentUser } from '@/lib/auth'

// GET /api/activity-logs - Get current user's recent activity logs
export async function GET(request: NextRequest) {
  try {
    const user = await getCurrentUser(request)
    if (!user) {
      return NextResponse.json(
        { success: false, error: 'Unauthorized' },
        { status: 401 }
      )
    }

    // Get last 10 activities for the user
    const activities = await db.activityLog.findMany({
      where: { userId: user.userId },
      orderBy: { createdAt: 'desc' },
      take: 10,
    })

    return NextResponse.json({
      success: true,
      data: activities,
    })
  } catch (error) {
    console.error('Error fetching activity logs:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch activity logs' },
      { status: 500 }
    )
  }
}

// POST /api/activity-logs - Create a new activity log (internal use)
export async function POST(request: NextRequest) {
  try {
    const user = await getCurrentUser(request)
    if (!user) {
      return NextResponse.json(
        { success: false, error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const body = await request.json()
    const { action, entityType, entityId, entityName, details } = body

    const activity = await db.activityLog.create({
      data: {
        userId: user.userId,
        action,
        entityType,
        entityId,
        entityName,
        details: details ? JSON.stringify(details) : null,
      },
    })

    return NextResponse.json({
      success: true,
      data: activity,
    })
  } catch (error) {
    console.error('Error creating activity log:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to create activity log' },
      { status: 500 }
    )
  }
}
