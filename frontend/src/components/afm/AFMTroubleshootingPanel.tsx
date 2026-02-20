import { useState } from 'react'
import { AlertCircle, CheckCircle, Loader2, Wrench, ChevronDown, ChevronRight } from 'lucide-react'
import axios from 'axios'

interface IssueCause {
  cause: string
  probability: number
  description: string
  checks: string[]
  fixes: string[]
}

interface DiagnosticResult {
  issue_type: string
  causes: IssueCause[]
  system_checks: any
}

export default function AFMTroubleshootingPanel() {
  const [issueDescription, setIssueDescription] = useState('')
  const [isDiagnosing, setIsDiagnosing] = useState(false)
  const [results, setResults] = useState<DiagnosticResult | null>(null)
  const [expandedCause, setExpandedCause] = useState<string | null>(null)
  const [appliedFixes, setAppliedFixes] = useState<Set<string>>(new Set())

  const exampleIssues = [
    'The signal is very weak',
    'CCD noise is too high',
    'Peak position is incorrect',
    'Signal is saturated',
    'No signal detected',
    'Focus problems'
  ]

  const handleDiagnose = async () => {
    if (!issueDescription.trim()) return

    setIsDiagnosing(true)
    try {
      // Stub: Use knowledge from MD files
      // CCD cooling, Filter-angle issues, Laser power misalignment, Grating calibration offset
      const response = await axios.get('/api/troubleshoot', {
        params: { issue: issueDescription }
      })
      setResults(response.data)
    } catch (error) {
      console.error('Error diagnosing:', error)
      // Fallback to dummy data
      setResults({
        issue_type: 'weak_signal',
        causes: [
          {
            cause: 'CCD temperature not stable',
            probability: 0.85,
            description: 'CCD cooling system may not have reached target temperature',
            checks: [
              'Check CCD temperature: Should be -70°C ± 0.5°C',
              'Verify cooling system status',
              'Check for temperature fluctuations'
            ],
            fixes: [
              'Wait for CCD to reach stable temperature',
              'Check cooling system connections',
              'Verify cooling fluid level'
            ]
          },
          {
            cause: 'Laser power too low',
            probability: 0.75,
            description: 'Laser power may be set below optimal range',
            checks: [
              'Check current laser power setting',
              'Verify laser is enabled',
              'Check for laser power drift'
            ],
            fixes: [
              'Increase laser power gradually',
              'Verify laser is turned on',
              'Check laser power calibration'
            ]
          },
          {
            cause: 'Filter angle misalignment',
            probability: 0.65,
            description: 'Notch filter angle may not be optimized',
            checks: [
              'Check filter angle setting',
              'Verify filter calibration',
              'Check for mechanical drift'
            ],
            fixes: [
              'Adjust filter angle to optimal position',
              'Recalibrate filter angle',
              'Check filter mount stability'
            ]
          }
        ],
        system_checks: {
          ccd_temperature: -70.2,
          ccd_stable: true,
          laser_power: 25.0,
          laser_enabled: true,
          filter_angle: 0.0
        }
      })
    } finally {
      setIsDiagnosing(false)
    }
  }

  const handleApplyFix = (cause: string, fixIndex: number) => {
    // Stub: Apply fix
    console.log(`Stub: Applying fix ${fixIndex} for ${cause}`)
    setAppliedFixes(prev => new Set([...prev, `${cause}-${fixIndex}`]))
    alert(`Stub: Fix applied. In production, this would execute: ${results?.causes.find(c => c.cause === cause)?.fixes[fixIndex]}`)
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header - Gemini Style */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-light text-gray-900 tracking-tight">
          Troubleshooting Agent
        </h1>
        <p className="text-gray-600 font-light max-w-2xl mx-auto">
          Intelligent diagnostic system for Raman spectroscopy issues.
        </p>
      </div>

      {/* Issue Input */}
      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Describe the Issue
        </label>
        <div className="flex gap-2">
          <textarea
            value={issueDescription}
            onChange={(e) => setIssueDescription(e.target.value)}
            placeholder="e.g., The signal is very weak, what should I check?"
            rows={3}
            className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500 resize-none"
          />
          <button
            onClick={handleDiagnose}
            disabled={isDiagnosing || !issueDescription.trim()}
            className="px-6 py-3 bg-afm-primary-500 text-white rounded-xl hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-medium"
          >
            {isDiagnosing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Diagnosing...
              </>
            ) : (
              <>
                <AlertCircle className="w-5 h-5" />
                Diagnose
              </>
            )}
          </button>
        </div>
      </div>

      {/* Example Issues */}
      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
        <p className="text-sm font-medium text-gray-700 mb-3">Example Issues:</p>
        <div className="flex flex-wrap gap-2">
          {exampleIssues.map((issue, idx) => (
            <button
              key={idx}
              onClick={() => setIssueDescription(issue)}
              className="px-3 py-1.5 bg-afm-primary-50 border border-afm-primary-200 text-afm-primary-700 text-sm rounded-lg hover:bg-afm-primary-100 transition-colors"
            >
              {issue}
            </button>
          ))}
        </div>
      </div>

      {/* Diagnostic Results */}
      {results && (
        <div className="space-y-4">
          {/* Issue Type */}
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="w-6 h-6 text-afm-primary-500" />
              <h3 className="text-lg font-semibold text-gray-900">
                Issue Type: {results.issue_type.replace('_', ' ').toUpperCase()}
              </h3>
            </div>
          </div>

          {/* Detected Causes */}
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <h3 className="font-semibold text-gray-900 mb-4">Detected Causes</h3>
            <div className="space-y-3">
              {results.causes.map((cause, idx) => (
                <div
                  key={idx}
                  className="border-2 border-gray-200 rounded-xl overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedCause(expandedCause === cause.cause ? null : cause.cause)}
                    className="w-full p-4 bg-gray-50 hover:bg-gray-100 transition-colors flex items-center justify-between"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-afm-primary-500 text-white rounded-full flex items-center justify-center font-bold">
                        {idx + 1}
                      </div>
                      <div className="text-left">
                        <p className="font-semibold text-gray-900">{cause.cause}</p>
                        <p className="text-sm text-gray-600">{cause.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <p className="text-sm font-medium text-afm-primary-700">
                          {(cause.probability * 100).toFixed(0)}% probability
                        </p>
                      </div>
                      {expandedCause === cause.cause ? (
                        <ChevronDown className="w-5 h-5 text-gray-600" />
                      ) : (
                        <ChevronRight className="w-5 h-5 text-gray-600" />
                      )}
                    </div>
                  </button>

                  {expandedCause === cause.cause && (
                    <div className="p-4 bg-white border-t border-gray-200">
                      {/* Checks */}
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold text-gray-900 mb-2">System Checks:</h4>
                        <ul className="space-y-2">
                          {cause.checks.map((check, checkIdx) => (
                            <li key={checkIdx} className="flex items-start gap-2 text-sm text-gray-700">
                              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                              <span>{check}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Fixes */}
                      <div>
                        <h4 className="text-sm font-semibold text-gray-900 mb-2">Recommended Fixes:</h4>
                        <div className="space-y-2">
                          {cause.fixes.map((fix, fixIdx) => (
                            <div
                              key={fixIdx}
                              className="flex items-center justify-between p-3 bg-afm-primary-50 border border-afm-primary-200 rounded-lg"
                            >
                              <p className="text-sm text-gray-700 flex-1">{fix}</p>
                              <button
                                onClick={() => handleApplyFix(cause.cause, fixIdx)}
                                disabled={appliedFixes.has(`${cause.cause}-${fixIdx}`)}
                                className="ml-3 px-4 py-2 bg-afm-primary-500 text-white rounded-lg hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm font-medium flex items-center gap-2"
                              >
                                {appliedFixes.has(`${cause.cause}-${fixIdx}`) ? (
                                  <>
                                    <CheckCircle className="w-4 h-4" />
                                    Applied
                                  </>
                                ) : (
                                  <>
                                    <Wrench className="w-4 h-4" />
                                    Apply Fix
                                  </>
                                )}
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* System Status */}
          {results.system_checks && (
            <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <h3 className="font-semibold text-gray-900 mb-4">System Status</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">CCD Temperature</p>
                  <p className="text-lg font-bold text-gray-900">
                    {results.system_checks.ccd_temperature?.toFixed(1)}°C
                  </p>
                  {results.system_checks.ccd_stable && (
                    <span className="text-xs text-green-600">✓ Stable</span>
                  )}
                </div>
                <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Laser Power</p>
                  <p className="text-lg font-bold text-gray-900">
                    {results.system_checks.laser_power?.toFixed(1)} mW
                  </p>
                  {results.system_checks.laser_enabled && (
                    <span className="text-xs text-green-600">✓ Enabled</span>
                  )}
                </div>
                <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Filter Angle</p>
                  <p className="text-lg font-bold text-gray-900">
                    {results.system_checks.filter_angle?.toFixed(1)}°
                  </p>
                </div>
                <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Grating</p>
                  <p className="text-lg font-bold text-gray-900">
                    1200 gr/mm
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {!results && !isDiagnosing && (
        <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-12 text-center shadow-sm">
          <AlertCircle className="w-16 h-16 text-gray-300 mx-auto mb-3" />
          <p className="text-sm text-gray-500">No diagnosis yet</p>
          <p className="text-xs text-gray-400 mt-1">Describe an issue and click Diagnose</p>
        </div>
      )}
    </div>
  )
}

