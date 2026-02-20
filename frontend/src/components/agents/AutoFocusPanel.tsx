import { useState } from 'react'
import { Focus, Play, Loader2, TrendingUp } from 'lucide-react'
import axios from 'axios'

export default function AutoFocusPanel() {
  const [zRange, setZRange] = useState(100)
  const [zStep, setZStep] = useState(5)
  const [focusMode, setFocusMode] = useState('hybrid')
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<any>(null)

  const handleRunAutoFocus = async () => {
    setIsRunning(true)
    try {
      const response = await axios.post('/api/autofocus', {
        initial_z: 0,
        z_range: zRange,
        z_step: zStep
      })
      setResults(response.data)
    } catch (error) {
      console.error('Error running autofocus:', error)
      alert('Failed to run autofocus. Make sure the backend is running.')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
        <h3 className="font-semibold text-orange-900 mb-2">Auto-Focus Agent</h3>
        <p className="text-sm text-orange-800">
          Hybrid focus optimization combining optical image sharpness and Raman peak intensity.
          Ensures perfect Z-position for high-quality spectrum acquisition.
        </p>
      </div>

      {/* Settings */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Focus Mode
          </label>
          <select
            value={focusMode}
            onChange={(e) => setFocusMode(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-sers-500 focus:border-transparent"
          >
            <option value="hybrid">Hybrid (Optical + Raman)</option>
            <option value="optical">Optical Only</option>
            <option value="raman">Raman Only</option>
          </select>
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
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-sers-500"
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
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-sers-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1 μm</span>
            <span>20 μm</span>
          </div>
        </div>
      </div>

      {/* Calculation */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-900">
          <strong>Estimated scans:</strong> {Math.ceil(zRange / zStep)} positions
        </p>
        <p className="text-sm text-blue-800 mt-1">
          <strong>Estimated time:</strong> ~{(Math.ceil(zRange / zStep) * 2).toFixed(0)} seconds
        </p>
      </div>

      {/* Action Button */}
      <button
        onClick={handleRunAutoFocus}
        disabled={isRunning}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-sers-500 text-white rounded-lg hover:bg-sers-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Optimizing Focus...
          </>
        ) : (
          <>
            <Focus className="w-5 h-5" />
            Run Auto-Focus
          </>
        )}
      </button>

      {/* Results */}
      {results && (
        <div className="bg-gray-50 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-gray-900">Focus Results</h4>
            <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
              Complete
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-sm text-gray-600">Optimal Z Position</p>
              <p className="text-2xl font-bold text-raman-500">
                {results.optimal_z?.toFixed(2)} μm
              </p>
            </div>
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-sm text-gray-600">Focus Score</p>
              <p className="text-2xl font-bold text-green-500">
                {results.best_score?.toFixed(3)}
              </p>
            </div>
          </div>

          {/* Focus Curve */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-gray-600" />
              <h5 className="text-sm font-semibold text-gray-900">Focus Curve</h5>
            </div>
            <div className="space-y-1">
              {results.z_positions?.slice(0, 10).map((z: number, idx: number) => {
                const score = results.focus_scores?.[idx] || 0
                const isOptimal = Math.abs(z - results.optimal_z) < 0.1
                return (
                  <div key={idx} className="flex items-center gap-2 text-xs">
                    <span className={`w-20 ${isOptimal ? 'font-bold text-raman-500' : 'text-gray-600'}`}>
                      Z={z.toFixed(1)}μm
                    </span>
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${isOptimal ? 'bg-raman-500' : 'bg-blue-400'}`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <span className="w-12 text-right text-gray-600">
                      {score.toFixed(3)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

