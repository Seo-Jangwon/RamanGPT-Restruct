import { useState } from 'react'
import { AlertCircle, Search, Loader2, CheckCircle, AlertTriangle } from 'lucide-react'
import axios from 'axios'

export default function TroubleshootingPanel() {
  const [issue, setIssue] = useState('')
  const [isDiagnosing, setIsDiagnosing] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [file, setFile] = useState<File | null>(null)
  const [filePreview, setFilePreview] = useState<string | null>(null)

  const commonIssues = [
    { label: 'Weak Signal', value: 'The signal is very weak' },
    { label: 'CCD Noise', value: 'Getting a lot of noise in spectra' },
    { label: 'Peak Position Wrong', value: 'Peak positions look incorrect' },
    { label: 'Saturation', value: 'Signal is saturated' },
    { label: 'No Signal', value: 'Not getting any signal' },
    { label: 'Focus Problems', value: 'Cannot get good focus' }
  ]

  const handleDiagnose = async () => {
    if (!issue.trim() && !file) return

    setIsDiagnosing(true)
    try {
      let response

      if (file) {
        const formData = new FormData()
        formData.append('file', file)
        if (issue.trim()) {
          formData.append('description', issue)
        }

        response = await axios.post('/api/troubleshoot/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      } else {
        response = await axios.get('/api/troubleshoot', {
          params: { issue },
        })
      }

      setResults(response.data)
    } catch (error) {
      console.error('Error diagnosing:', error)
      alert('Failed to diagnose issue. Make sure the backend is running.')
    } finally {
      setIsDiagnosing(false)
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0]
    if (!selected) {
      setFile(null)
      setFilePreview(null)
      return
    }

    setFile(selected)

    if (selected.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setFilePreview(e.target?.result as string)
      }
      reader.readAsDataURL(selected)
    } else {
      setFilePreview(null)
    }
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h3 className="font-semibold text-red-900 mb-2">Troubleshooting Agent</h3>
        <p className="text-sm text-red-800">
          Intelligent diagnostic system that analyzes issues, checks system status, and provides
          step-by-step solutions with probability-ranked causes.
        </p>
      </div>

      {/* Issue Input */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Describe the Issue
        </label>
        <textarea
          value={issue}
          onChange={(e) => setIssue(e.target.value)}
          placeholder="e.g., The signal is very weak and I can barely see any peaks..."
          rows={3}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none"
        />
      </div>

      {/* File Upload */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Attach Data (optional)
        </label>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-red-400 transition-colors">
          <input
            type="file"
            id="troubleshoot-file"
            accept=".csv,.xls,.xlsx,image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <label htmlFor="troubleshoot-file" className="cursor-pointer block">
            <p className="text-sm text-gray-700 font-medium mb-1">
              Upload spectrum image, CSV, or Excel log
            </p>
            <p className="text-xs text-gray-500">
              Supported: images (PNG/JPG), spreadsheets (.csv, .xls, .xlsx)
            </p>
          </label>

          {file && (
            <div className="mt-3 text-left text-xs text-gray-600">
              <p className="font-medium">Selected file:</p>
              <p>{file.name} ({Math.round(file.size / 1024)} kB)</p>
            </div>
          )}

          {filePreview && (
            <div className="mt-3 flex justify-center">
              <img
                src={filePreview}
                alt="Uploaded preview"
                className="max-h-40 rounded border border-gray-200"
              />
            </div>
          )}
        </div>
        <p className="mt-1 text-xs text-gray-500">
          You can describe the issue, attach a file, or do both. The system will attempt to infer
          the problem from the uploaded data even if you are not familiar with Raman terminology.
        </p>
      </div>

      {/* Common Issues */}
      <div>
        <p className="text-sm font-medium text-gray-700 mb-2">Common Issues:</p>
        <div className="grid grid-cols-2 gap-2">
          {commonIssues.map((item, idx) => (
            <button
              key={idx}
              onClick={() => setIssue(item.value)}
              className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-lg transition-colors text-left"
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Diagnose Button */}
      <button
        onClick={handleDiagnose}
        disabled={isDiagnosing || (!issue.trim() && !file)}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        {isDiagnosing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Diagnosing...
          </>
        ) : (
          <>
            <Search className="w-5 h-5" />
            Diagnose Issue
          </>
        )}
      </button>

      {/* Results */}
      {results && (
        <div className="bg-gray-50 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-gray-900">Diagnostic Report</h4>
            <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full">
              {results.diagnosis?.num_possible_causes} causes found
            </span>
          </div>

          {/* Summary */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-5 h-5 text-blue-600" />
              <p className="font-semibold text-blue-900">Issue Identified</p>
            </div>
            <p className="text-sm text-blue-800">{results.diagnosis?.summary}</p>
          </div>

          {/* Possible Causes */}
          <div className="space-y-3">
            <p className="text-sm font-semibold text-gray-900">
              Possible Causes (ranked by probability):
            </p>

            {results.recommendations?.slice(0, 3).map((rec: any, idx: number) => (
              <div
                key={idx}
                className="bg-white rounded-lg border border-gray-200 p-4 space-y-3"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1">
                    <span className="flex items-center justify-center w-6 h-6 rounded-full bg-raman-100 text-raman-700 text-sm font-bold flex-shrink-0">
                      {idx + 1}
                    </span>
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900">{rec.cause}</p>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-red-500 to-orange-500 h-2 rounded-full"
                            style={{ width: `${rec.probability * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-gray-600">
                          {(rec.probability * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Diagnostics */}
                <div>
                  <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-500" />
                    Check these:
                  </p>
                  <ul className="space-y-1">
                    {rec.diagnostics?.map((diag: string, didx: number) => (
                      <li key={didx} className="text-sm text-gray-600 flex gap-2">
                        <span className="text-blue-500">•</span>
                        <span>{diag}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Fixes */}
                <div>
                  <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-green-500" />
                    How to fix:
                  </p>
                  <ul className="space-y-1">
                    {rec.fixes?.map((fix: string, fidx: number) => (
                      <li key={fidx} className="text-sm text-gray-600 flex gap-2">
                        <span className="text-green-500">✓</span>
                        <span>{fix}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Automated Check */}
                {rec.automated_check && Object.keys(rec.automated_check).length > 0 && (
                  <div className="bg-gray-50 rounded p-3 border border-gray-200">
                    <p className="text-xs font-semibold text-gray-700 mb-2">
                      Automated System Check:
                    </p>
                    {Object.entries(rec.automated_check).map(([key, value]: [string, any]) => (
                      <div key={key} className="text-xs text-gray-600">
                        <strong>{key}:</strong> {JSON.stringify(value, null, 2)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* System Status */}
          {results.system_status && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <p className="text-sm font-semibold text-gray-900 mb-3">Current System Status</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-600">Laser:</span>
                  <span className={`ml-2 font-semibold ${
                    results.system_status.raonspec?.spectrometer?.laser_on
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}>
                    {results.system_status.raonspec?.spectrometer?.laser_on ? 'ON' : 'OFF'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">CCD Temp:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    {results.system_status.raonspec?.ccd?.temperature}°C
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Laser Power:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    {results.system_status.raonspec?.spectrometer?.laser_power} mW
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">System:</span>
                  <span className={`ml-2 font-semibold ${
                    results.system_status.raonspec?.initialized
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}>
                    {results.system_status.raonspec?.initialized ? 'Ready' : 'Not Ready'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

