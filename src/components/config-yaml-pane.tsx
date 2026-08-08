'use client'

/**
 * Shared "generated YAML / hand-edited YAML" pane used by the model and
 * training config dialogs.
 *
 * The contract that makes the advanced editor trustworthy: whatever this pane
 * displays is exactly what gets persisted and trained on. In `auto` mode the
 * YAML is re-derived from the form on every keystroke; in `manual` mode the
 * form stops driving it and the textarea becomes the source of truth. Switching
 * back to `auto` is explicitly destructive and says so.
 */

import { useMemo } from 'react'
import { parseDocument } from 'yaml'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle, Code, Copy, Pencil, RotateCcw, TriangleAlert } from 'lucide-react'
import { toast } from '@/hooks/use-toast'

export interface ConfigIssue {
  level: 'error' | 'warning'
  message: string
}

interface ConfigYamlPaneProps {
  /** The effective YAML: generated from the form, or the user's manual text. */
  yaml: string
  /** True when the user has taken manual control of the YAML. */
  manual: boolean
  onManualYamlChange: (value: string) => void
  /** Enter manual mode, seeding the textarea with the current generated YAML. */
  onEnterManual: () => void
  /** Leave manual mode, discarding hand edits. */
  onLeaveManual: () => void
  /** Semantic issues contributed by the caller (e.g. loss/logits mismatch). */
  issues?: ConfigIssue[]
  rows?: number
}

export function ConfigYamlPane({
  yaml,
  manual,
  onManualYamlChange,
  onEnterManual,
  onLeaveManual,
  issues = [],
  rows = 20,
}: ConfigYamlPaneProps) {
  // Validate as the user types rather than on submit, so a YAML typo is visible
  // immediately instead of after the request round-trips. Parents gate their
  // save button with the exported `yamlSyntaxError` on the same input.
  const syntaxError = useMemo(() => yamlSyntaxError(yaml), [yaml])

  const errors = issues.filter((i) => i.level === 'error')
  const warnings = issues.filter((i) => i.level === 'warning')

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Code className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium">Generated YAML</span>
          {manual ? (
            <Badge variant="secondary">Manually edited</Badge>
          ) : (
            <Badge variant="outline">Synced with form</Badge>
          )}
        </div>
        <div className="flex gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => {
              navigator.clipboard.writeText(yaml)
              toast({ title: 'YAML copied to clipboard' })
            }}
          >
            <Copy className="w-4 h-4 mr-1" />
            Copy
          </Button>
          {manual ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                if (confirm('Discard your manual YAML edits and regenerate from the form?')) {
                  onLeaveManual()
                }
              }}
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              Regenerate from form
            </Button>
          ) : (
            <Button type="button" variant="outline" size="sm" onClick={onEnterManual}>
              <Pencil className="w-4 h-4 mr-1" />
              Edit manually
            </Button>
          )}
        </div>
      </div>

      {manual ? (
        <Textarea
          value={yaml}
          onChange={(e) => onManualYamlChange(e.target.value)}
          rows={rows}
          spellCheck={false}
          className="font-mono text-xs leading-relaxed max-h-[62vh] min-h-[320px] resize-y"
        />
      ) : (
        <pre className="p-3 rounded-lg bg-muted/50 text-xs overflow-auto max-h-[62vh] min-h-[320px] font-mono leading-relaxed whitespace-pre">
          {yaml}
        </pre>
      )}

      {manual && (
        <p className="text-xs text-muted-foreground">
          The form controls no longer drive this YAML. It is saved exactly as written.
        </p>
      )}

      {syntaxError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-xs whitespace-pre-wrap">
            YAML syntax error: {syntaxError}
          </AlertDescription>
        </Alert>
      )}

      {!syntaxError &&
        errors.map((issue, i) => (
          <Alert variant="destructive" key={`e${i}`}>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs">{issue.message}</AlertDescription>
          </Alert>
        ))}

      {!syntaxError &&
        warnings.map((issue, i) => (
          <Alert key={`w${i}`}>
            <TriangleAlert className="h-4 w-4" />
            <AlertDescription className="text-xs">{issue.message}</AlertDescription>
          </Alert>
        ))}
    </div>
  )
}

/** Same validation the pane performs, for parents that gate the save button. */
export function yamlSyntaxError(yaml: string): string | null {
  if (!yaml.trim()) return 'Configuration is empty.'
  try {
    const doc = parseDocument(yaml, { logLevel: 'silent' })
    if (doc.errors.length > 0) return doc.errors[0].message
    if (doc.contents === null) return 'Configuration is empty.'
    return null
  } catch (error) {
    return error instanceof Error ? error.message : 'Invalid YAML'
  }
}
