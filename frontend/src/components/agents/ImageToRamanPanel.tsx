import { useState } from 'react'
import { Camera, Play, MapPin, Loader2, Radio, Square } from 'lucide-react'
import axios from 'axios'

interface ROI {
  x: number
  y: number
  confidence: number
  type: string
  image_coords: [number, number]
}

export default function ImageToRamanPanel() {
  const [detectionMode, setDetectionMode] = useState('sers_hotspots')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [isLive, setIsLive] = useState(false)

  const handleCaptureImage = async () => {
    setIsAnalyzing(true)
    try {
      const response = await axios.post('/api/chat', {
        message: `Analyze current microscopy image and detect ${detectionMode.replace('_', ' ')}`,
        agent: 'image-raman'
      })
      setResults(response.data.data)
    } catch (error) {
      console.error('Error analyzing image:', error)
      alert('Failed to analyze image. Make sure the backend is running.')
    } finally {
      setIsAnalyzing(false)
    }
  }


  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="font-semibold text-blue-900 mb-2">Image-to-Raman Agent</h3>
        <p className="text-sm text-blue-800">
          Automatically detects regions of interest from microscopy images using AI-powered
          segmentation. Perfect for identifying SERS hotspots, cells, or tissue structures.
        </p>
      </div>

      {/* Detection Mode Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Detection Mode
        </label>
        <select
          value={detectionMode}
          onChange={(e) => setDetectionMode(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-raman-500 focus:border-transparent"
        >
          <option value="sers_hotspots">SERS Hotspots</option>
          <option value="cells">Cells</option>
          <option value="tissue">Tissue Regions</option>
          <option value="auto">Auto-detect</option>
        </select>
      </div>

      {/* Live Microscope Feed (CCD) */}
      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow">
        {/* Card Header */}
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-900">Live Microscope Feed (CCD)</h3>
          {/* Status Pill - Top Right */}
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${
            isLive 
              ? 'bg-green-100 text-green-700 border border-green-300' 
              : 'bg-gray-100 text-gray-600 border border-gray-300'
          }`}>
            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            <span>{isLive ? 'Live' : 'Disconnected'}</span>
          </div>
        </div>

        {/* Placeholder Video Stream Area */}
        <div className="relative bg-gray-100 border-2 border-dashed border-gray-300 rounded-lg aspect-square flex items-center justify-center overflow-hidden">
          {/* Centered Text */}
          <div className="text-center space-y-2 z-10">
            <p className="text-lg font-bold text-gray-700">
              Live CCD Microscopy View
            </p>
            <p className="text-sm text-gray-500">
              Streaming preview from Olympus BX image camera (dummy feed)
            </p>
          </div>

          {/* LIVE Overlay - Top Left Corner (when live) */}
          {isLive && (
            <div className="absolute top-3 left-3 bg-red-500 text-white px-3 py-1 rounded-md text-xs font-bold shadow-lg z-20">
              LIVE
            </div>
          )}
        </div>

        {/* Control Buttons - Bottom */}
        <div className="mt-4 flex gap-3">
          <button
            onClick={() => {
              setIsLive(true)
              console.log('Stub: Starting live CCD feed')
            }}
            disabled={isLive}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-colors ${
              isLive
                ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                : 'bg-raman-500 text-white hover:bg-raman-600'
            }`}
          >
            <Radio className="w-4 h-4" />
            Start Live View
          </button>
          <button
            onClick={() => {
              setIsLive(false)
              console.log('Stub: Stopping live CCD feed')
            }}
            disabled={!isLive}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-colors ${
              !isLive
                ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                : 'bg-red-500 text-white hover:bg-red-600'
            }`}
          >
            <Square className="w-4 h-4" />
            Stop Live View
          </button>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={handleCaptureImage}
          disabled={isAnalyzing || !isLive}
          className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-raman-500 text-white rounded-lg hover:bg-raman-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Camera className="w-5 h-5" />
              Capture & Analyze
            </>
          )}
        </button>
      </div>

      {/* Results */}
      {results && (
        <div className="bg-gray-50 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-gray-900">Detection Results</h4>
            <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
              {results.num_rois} ROIs found
            </span>
          </div>

          {results.rois && results.rois.length > 0 ? (
            <div className="space-y-2">
              {results.rois.slice(0, 10).map((roi: ROI, idx: number) => (
                <div
                  key={idx}
                  className="bg-white border border-gray-200 rounded-lg p-3 flex items-center justify-between hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center gap-3">
                    <MapPin className="w-5 h-5 text-raman-500" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {roi.type.replace('_', ' ')}
                      </p>
                      <p className="text-xs text-gray-500">
                        Position: ({roi.x.toFixed(2)}, {roi.y.toFixed(2)}) μm
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">
                      {(roi.confidence * 100).toFixed(0)}%
                    </p>
                    <p className="text-xs text-gray-500">confidence</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-600">No ROIs detected. Try a different mode.</p>
          )}

          <button
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-sers-500 text-white rounded-lg hover:bg-sers-600 transition-colors"
          >
            <Play className="w-4 h-4" />
            Acquire Spectra at ROIs
          </button>
        </div>
      )}
    </div>
  )
}

