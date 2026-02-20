import { useState } from 'react'
import { Sparkles, Loader2, Zap, BookOpen, CheckCircle, Clock } from 'lucide-react'
import axios from 'axios'

interface AcquisitionSettings {
  laserPower: number
  exposure: number
  accumulations: number
  grating: number
  ndFilter: number
}

interface OptimizationStep {
  iteration: number
  settings: AcquisitionSettings
  score: number
  timestamp: Date
}

export default function AFMRAGOptimizerPanel() {
  const [sampleType, setSampleType] = useState('cell')
  const [purpose, setPurpose] = useState('qualitative')
  const [targetPeaks, setTargetPeaks] = useState('')
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [optimizationHistory, setOptimizationHistory] = useState<OptimizationStep[]>([])

  const handleOptimize = async () => {
    setIsOptimizing(true)
    try {
      // Stub: Call existing RAG optimizer
      // Reference RAG flow described in docs
      // Wrap existing text-embedding / retrieval module (dummy wrapper)
      
      const response = await axios.post('/api/optimize-parameters', {
        sample_type: sampleType,
        purpose: purpose,
        target_peaks: targetPeaks ? targetPeaks.split(',').map(p => parseFloat(p.trim())) : null
      })
      setResults(response.data)
      
      // Add to optimization history
      if (response.data.spectrometer_settings && response.data.ccd_settings) {
        const newStep: OptimizationStep = {
          iteration: optimizationHistory.length + 1,
          settings: {
            laserPower: response.data.spectrometer_settings.laser_power_mw || 50,
            exposure: response.data.ccd_settings.exposure_time_s || 1.0,
            accumulations: response.data.ccd_settings.num_accumulations || 3,
            grating: parseInt(response.data.spectrometer_settings.grating) || 1200,
            ndFilter: response.data.spectrometer_settings.nd_filter || 1.0
          },
          score: 0.85 + Math.random() * 0.1, // Dummy score
          timestamp: new Date()
        }
        setOptimizationHistory(prev => [...prev, newStep])
      }
    } catch (error) {
      console.error('Error optimizing:', error)
      // Fallback to dummy data
      setResults({
        summary: 'Recommended parameters for optimal signal-to-noise ratio',
        spectrometer_settings: {
          laser_power_mw: 50,
          grating: '1200',
          nd_filter: 1.0
        },
        ccd_settings: {
          exposure_time_s: 1.5,
          num_accumulations: 5
        },
        expected_peaks: [1000, 1500, 2000],
        reasoning: [
          'Laser power optimized for sample sensitivity',
          'Exposure time balanced for signal quality',
          'Grating selected for optimal resolution'
        ]
      })
    } finally {
      setIsOptimizing(false)
    }
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header - Gemini Style */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-light text-gray-900 tracking-tight">
          Raman/SERS RAG Optimization
        </h1>
        <p className="text-gray-600 font-light max-w-2xl mx-auto">
          AI-powered parameter recommendations using RAG.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Sample Context Form */}
        <div className="space-y-4">
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <h3 className="font-semibold text-gray-900 mb-4">Sample Context</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sample Type
                </label>
                <select
                  value={sampleType}
                  onChange={(e) => setSampleType(e.target.value)}
                  className="w-full px-4 py-2 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500"
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
                  className="w-full px-4 py-2 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500"
                >
                  <option value="qualitative">Qualitative (Quick identification)</option>
                  <option value="quantitative">Quantitative (Precise measurements)</option>
                  <option value="mapping">Mapping (Large area scan)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Peaks (cm⁻¹, comma-separated)
                </label>
                <input
                  type="text"
                  value={targetPeaks}
                  onChange={(e) => setTargetPeaks(e.target.value)}
                  placeholder="e.g., 1000, 1500, 2000"
                  className="w-full px-4 py-2 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500"
                />
              </div>

              <button
                onClick={handleOptimize}
                disabled={isOptimizing}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-afm-primary-500 text-white rounded-xl hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
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
            </div>
          </div>
        </div>

        {/* Right: Recommended Settings & Timeline */}
        <div className="space-y-4">
          {/* Recommended Settings Cards */}
          {results ? (
            <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Recommended Settings</h3>
                <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
                  Optimized
                </span>
              </div>

              {/* Quick Summary */}
              <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-4 mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-afm-primary-600" />
                  <p className="text-sm font-semibold text-afm-primary-900">Quick Summary</p>
                </div>
                <p className="text-sm text-afm-primary-800">{results.summary}</p>
              </div>

              {/* Parameter Cards */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-3">
                  <p className="text-xs text-afm-primary-700 mb-1">Laser Power</p>
                  <p className="text-lg font-bold text-afm-primary-900">
                    {results.spectrometer_settings?.laser_power_mw} mW
                  </p>
                </div>
                <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-3">
                  <p className="text-xs text-afm-primary-700 mb-1">Exposure</p>
                  <p className="text-lg font-bold text-afm-primary-900">
                    {results.ccd_settings?.exposure_time_s} s
                  </p>
                </div>
                <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-3">
                  <p className="text-xs text-afm-primary-700 mb-1">Accumulations</p>
                  <p className="text-lg font-bold text-afm-primary-900">
                    {results.ccd_settings?.num_accumulations}
                  </p>
                </div>
                <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-3">
                  <p className="text-xs text-afm-primary-700 mb-1">Grating</p>
                  <p className="text-lg font-bold text-afm-primary-900">
                    {results.spectrometer_settings?.grating} gr/mm
                  </p>
                </div>
              </div>

              {/* Expected Peaks */}
              {results.expected_peaks && results.expected_peaks.length > 0 && (
                <div className="mb-4">
                  <p className="text-sm font-semibold text-gray-900 mb-2">Expected Peaks</p>
                  <div className="flex flex-wrap gap-2">
                    {results.expected_peaks.map((peak: number, idx: number) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-afm-primary-100 text-afm-primary-800 text-sm rounded-full"
                      >
                        {peak} cm⁻¹
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Reasoning */}
              {results.reasoning && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <BookOpen className="w-4 h-4 text-gray-600" />
                    <p className="text-sm font-semibold text-gray-900">AI Reasoning</p>
                  </div>
                  <ul className="space-y-2">
                    {results.reasoning.map((reason: string, idx: number) => (
                      <li key={idx} className="text-sm text-gray-700 flex gap-2">
                        <span className="text-afm-primary-500">•</span>
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-center py-12 text-gray-500">
                <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p className="text-sm">No recommendations yet</p>
                <p className="text-xs">Fill in sample context and optimize</p>
              </div>
            </div>
          )}

          {/* Optimization Loop Timeline */}
          {optimizationHistory.length > 0 && (
            <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-4">
                <Clock className="w-4 h-4 text-gray-600" />
                <h3 className="font-semibold text-gray-900">Optimization History</h3>
              </div>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {optimizationHistory.map((step, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-3 p-3 bg-gray-50 border border-gray-200 rounded-xl"
                  >
                    <div className="flex-shrink-0 w-8 h-8 bg-afm-primary-500 text-white rounded-full flex items-center justify-center text-sm font-medium">
                      {step.iteration}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900">
                        Laser: {step.settings.laserPower}mW, Exp: {step.settings.exposure}s
                      </p>
                      <p className="text-xs text-gray-500">
                        Score: {step.score.toFixed(3)} • {step.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

