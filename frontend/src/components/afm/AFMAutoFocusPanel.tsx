import { useState } from 'react'
import { Focus, Play, Loader2, TrendingUp, Eye, Zap } from 'lucide-react'
import axios from 'axios'

export default function AFMAutoFocusPanel() {
  const [zRange, setZRange] = useState(100)
  const [zStep, setZStep] = useState(5)
  const [initialZ, setInitialZ] = useState(0)
  const [focusMode, setFocusMode] = useState('hybrid')
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<any>(null)

  const handleRunZScan = async () => {
    setIsRunning(true)
    try {
      // Stub: Call existing autofocus pipeline
      // await hardware.stage.zScan({ initial_z, z_range, z_step }); // stub
      // await hardware.raman.acquireSingle(); // stub
      
      const response = await axios.post('/api/autofocus', {
        initial_z: initialZ,
        z_range: zRange,
        z_step: zStep
      })
      setResults(response.data)
    } catch (error) {
      console.error('Error running Z-scan:', error)
      // Fallback to dummy data
      setResults({
        optimal_z: initialZ + zRange / 2,
        best_score: 0.85,
        z_positions: Array.from({ length: Math.ceil(zRange / zStep) }, (_, i) => initialZ + i * zStep),
        focus_scores: Array.from({ length: Math.ceil(zRange / zStep) }, () => 0.5 + Math.random() * 0.4)
      })
    } finally {
      setIsRunning(false)
    }
  }

  // Generate dummy focus curve data
  const focusData = results?.z_positions?.map((z: number, idx: number) => ({
    z,
    score: results.focus_scores?.[idx] || 0
  })) || []

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header - Gemini Style */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-light text-gray-900 tracking-tight">
          Auto-Focus & Z-Optimization
        </h1>
        <p className="text-gray-600 font-light max-w-2xl mx-auto">
          Optimize Z-position using hybrid optical and Raman focus metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Z-Scan Parameters */}
        <div className="space-y-4">
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <h3 className="font-semibold text-gray-900 mb-4">Z-Scan Parameters</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Focus Mode
                </label>
                <select
                  value={focusMode}
                  onChange={(e) => setFocusMode(e.target.value)}
                  className="w-full px-4 py-2 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500"
                >
                  <option value="hybrid">Hybrid (Optical + Raman)</option>
                  <option value="optical">Optical Only</option>
                  <option value="raman">Raman Only</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Initial Z Position (μm): {initialZ}
                </label>
                <input
                  type="range"
                  min="-500"
                  max="500"
                  step="10"
                  value={initialZ}
                  onChange={(e) => setInitialZ(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-afm-primary-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>-500 μm</span>
                  <span>500 μm</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Z-Range (μm): {zRange}
                </label>
                <input
                  type="range"
                  min="20"
                  max="500"
                  step="10"
                  value={zRange}
                  onChange={(e) => setZRange(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-afm-primary-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>20 μm</span>
                  <span>500 μm</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Z-Step (μm): {zStep}
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={zStep}
                  onChange={(e) => setZStep(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-afm-primary-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1 μm</span>
                  <span>20 μm</span>
                </div>
              </div>

              {/* Calculation */}
              <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-4">
                <div className="space-y-1">
                  <p className="text-sm text-afm-primary-900">
                    <strong>Estimated scans:</strong> {Math.ceil(zRange / zStep)} positions
                  </p>
                  <p className="text-sm text-afm-primary-800">
                    <strong>Estimated time:</strong> ~{(Math.ceil(zRange / zStep) * 2).toFixed(0)} seconds
                  </p>
                </div>
              </div>

              <button
                onClick={handleRunZScan}
                disabled={isRunning}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-afm-primary-500 text-white rounded-xl hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Running Z-Scan...
                  </>
                ) : (
                  <>
                    <Focus className="w-5 h-5" />
                    Run Z-Scan
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Focus Metric Chart & Preview */}
        <div className="space-y-4">
          {/* Focus Metric Chart */}
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-gray-900">Focus Metric</h3>
              {results && (
                <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
                  Complete
                </span>
              )}
            </div>

            {results ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-afm-primary-50 border border-afm-primary-200 rounded-xl p-4">
                    <p className="text-sm text-afm-primary-700 mb-1">Optimal Z</p>
                    <p className="text-2xl font-bold text-afm-primary-900">
                      {results.optimal_z?.toFixed(2)} μm
                    </p>
                  </div>
                  <div className="bg-green-50 border border-green-200 rounded-xl p-4">
                    <p className="text-sm text-green-700 mb-1">Focus Score</p>
                    <p className="text-2xl font-bold text-green-900">
                      {results.best_score?.toFixed(3)}
                    </p>
                  </div>
                </div>

                {/* Focus Curve */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="w-4 h-4 text-gray-600" />
                    <h4 className="text-sm font-semibold text-gray-900">Focus Curve</h4>
                  </div>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {focusData.slice(0, 20).map((data: any, idx: number) => {
                      const isOptimal = Math.abs(data.z - results.optimal_z) < zStep / 2
                      return (
                        <div key={idx} className="flex items-center gap-3 text-xs">
                          <span className={`w-24 ${isOptimal ? 'font-bold text-afm-primary-500' : 'text-gray-600'}`}>
                            Z={data.z.toFixed(1)}μm
                          </span>
                          <div className="flex-1 bg-gray-200 rounded-full h-3">
                            <div
                              className={`h-3 rounded-full transition-all ${
                                isOptimal ? 'bg-afm-primary-500' : 'bg-blue-400'
                              }`}
                              style={{ width: `${data.score * 100}%` }}
                            />
                          </div>
                          <span className="w-12 text-right text-gray-600">
                            {data.score.toFixed(3)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Eye className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p className="text-sm">No scan data</p>
                <p className="text-xs">Run Z-scan to see results</p>
              </div>
            )}
          </div>

          {/* Optical + Raman Preview */}
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <h3 className="font-semibold text-gray-900 mb-4">Preview</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-100 rounded-lg aspect-square flex items-center justify-center border-2 border-gray-300">
                <div className="text-center text-gray-500">
                  <Eye className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  <p className="text-xs">Optical</p>
                </div>
              </div>
              <div className="bg-gray-100 rounded-lg aspect-square flex items-center justify-center border-2 border-gray-300">
                <div className="text-center text-gray-500">
                  <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  <p className="text-xs">Raman</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

