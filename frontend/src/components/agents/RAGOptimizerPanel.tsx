import { useState } from 'react'
import { Sparkles, Loader2, Zap, BookOpen } from 'lucide-react'
import axios from 'axios'

export default function RAGOptimizerPanel() {
  const [sampleType, setSampleType] = useState('cell')
  const [purpose, setPurpose] = useState('qualitative')
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [results, setResults] = useState<any>(null)

  const handleOptimize = async () => {
    setIsOptimizing(true)
    try {
      const response = await axios.post('/api/optimize-parameters', {
        sample_type: sampleType,
        purpose: purpose
      })
      setResults(response.data)
    } catch (error) {
      console.error('Error optimizing:', error)
      alert('Failed to optimize parameters. Make sure the backend is running.')
    } finally {
      setIsOptimizing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h3 className="font-semibold text-purple-900 mb-2">RAG Optimization Agent</h3>
        <p className="text-sm text-purple-800">
          AI-powered parameter recommendations based on scientific literature, sample type,
          and measurement purpose. Uses retrieval-augmented generation for intelligent suggestions.
        </p>
      </div>

      {/* Input Form */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Sample Type
          </label>
          <select
            value={sampleType}
            onChange={(e) => setSampleType(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-raman-500 focus:border-transparent"
          >
            <option value="cell">Cell (biological)</option>
            <option value="sers_substrate">SERS Substrate</option>
            <option value="tissue">Tissue</option>
            <option value="polymer">Polymer</option>
            <option value="graphene">Graphene / Carbon Materials</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Measurement Purpose
          </label>
          <select
            value={purpose}
            onChange={(e) => setPurpose(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-raman-500 focus:border-transparent"
          >
            <option value="qualitative">Qualitative (Quick identification)</option>
            <option value="quantitative">Quantitative (Precise measurements)</option>
            <option value="mapping">Mapping (Large area scan)</option>
          </select>
        </div>
      </div>

      {/* Action Button */}
      <button
        onClick={handleOptimize}
        disabled={isOptimizing}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-raman-500 to-sers-500 text-white rounded-lg hover:from-raman-600 hover:to-sers-600 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all"
      >
        {isOptimizing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Optimizing...
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            Get Recommendations
          </>
        )}
      </button>

      {/* Results */}
      {results && (
        <div className="bg-gray-50 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-gray-900">Recommended Parameters</h4>
            <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
              Optimized
            </span>
          </div>

          {/* Quick Summary */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-blue-600" />
              <p className="text-sm font-semibold text-blue-900">Quick Summary</p>
            </div>
            <p className="text-sm text-blue-800">{results.summary}</p>
          </div>

          {/* Detailed Parameters */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-600">Laser Power</p>
              <p className="text-lg font-bold text-raman-500">
                {results.spectrometer_settings?.laser_power_mw} mW
              </p>
            </div>
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-600">Exposure Time</p>
              <p className="text-lg font-bold text-raman-500">
                {results.ccd_settings?.exposure_time_s} s
              </p>
            </div>
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-600">Accumulations</p>
              <p className="text-lg font-bold text-raman-500">
                {results.ccd_settings?.num_accumulations}
              </p>
            </div>
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-600">Grating</p>
              <p className="text-lg font-bold text-raman-500">
                {results.spectrometer_settings?.grating} gr/mm
              </p>
            </div>
          </div>

          {/* Expected Peaks */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <p className="text-sm font-semibold text-gray-900 mb-2">Expected Peaks</p>
            <div className="flex flex-wrap gap-2">
              {results.expected_peaks?.map((peak: number, idx: number) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                >
                  {peak} cm⁻¹
                </span>
              ))}
            </div>
          </div>

          {/* Reasoning */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <div className="flex items-center gap-2 mb-3">
              <BookOpen className="w-4 h-4 text-gray-600" />
              <p className="text-sm font-semibold text-gray-900">AI Reasoning</p>
            </div>
            <ul className="space-y-2">
              {results.reasoning?.map((reason: string, idx: number) => (
                <li key={idx} className="text-sm text-gray-700 flex gap-2">
                  <span className="text-raman-500">•</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Note */}
          {results.notes && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <p className="text-sm text-yellow-900">
                <strong>Note:</strong> {results.notes}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

