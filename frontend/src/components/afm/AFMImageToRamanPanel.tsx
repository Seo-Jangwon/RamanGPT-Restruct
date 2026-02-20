import { useState } from 'react'
import { Camera, Play, MapPin, Loader2, CheckCircle, X, Target, Radio, Square } from 'lucide-react'
import axios from 'axios'

interface ROI {
  x: number
  y: number
  confidence: number
  type: string
  image_coords: [number, number]
}

interface MappingQueueItem {
  id: string
  type: 'single' | 'accumulate' | 'mapping'
  rois: ROI[]
  status: 'pending' | 'running' | 'completed' | 'error'
}

export default function AFMImageToRamanPanel() {
  const [detectionMode, setDetectionMode] = useState('sers_hotspots')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [selectedROIs, setSelectedROIs] = useState<ROI[]>([])
  const [mappingQueue, setMappingQueue] = useState<MappingQueueItem[]>([])
  const [isLive, setIsLive] = useState(false)

  const handleCaptureAndAnalyze = async () => {
    setIsAnalyzing(true)
    try {
      // Stub: Call existing pipeline with dummy adapter
      const response = await axios.post('/api/chat', {
        message: `Analyze current microscopy image and detect ${detectionMode.replace('_', ' ')}`,
        agent: 'image-raman'
      })
      setResults(response.data.data)
      setSelectedROIs(response.data.data?.rois || [])
    } catch (error) {
      console.error('Error analyzing image:', error)
      // Fallback to dummy data
      setResults({
        num_rois: 5,
        rois: Array.from({ length: 5 }, (_, i) => ({
          x: 100 + i * 50,
          y: 100 + i * 30,
          confidence: 0.7 + Math.random() * 0.3,
          type: detectionMode,
          image_coords: [512 + i * 100, 512 + i * 60]
        }))
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleAddToQueue = (type: 'single' | 'accumulate' | 'mapping') => {
    if (selectedROIs.length === 0) {
      alert('Please select ROIs first')
      return
    }

    const newItem: MappingQueueItem = {
      id: Date.now().toString(),
      type,
      rois: selectedROIs,
      status: 'pending'
    }

    setMappingQueue(prev => [...prev, newItem])
  }

  const handleAcquireAtROIs = async () => {
    if (selectedROIs.length === 0) return

    // Stub: Call existing acquisition pipeline
    // await hardware.stage.moveTo(x, y, z); // stub
    // await hardware.raman.acquireSingle(); // stub
    
    console.log('Stub: Acquiring spectra at ROIs:', selectedROIs)
    alert(`Stub: Would acquire ${selectedROIs.length} spectra using existing pipeline`)
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header - Modern Gemini Style */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-light text-gray-900 tracking-tight">
          Image-to-Raman Agent
        </h1>
        <p className="text-gray-600 font-light max-w-2xl mx-auto">
          Analyze microscopy images, detect regions of interest, and acquire spectra at precise coordinates.
        </p>
      </div>
      
      {/* Content with modern card styling */}
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Live Microscope Feed (CCD) */}
          <div className="space-y-4">
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
                      : 'bg-afm-primary-500 text-white hover:bg-afm-primary-600'
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

            {/* Detection Mode & Capture Controls */}
            <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow">
              <div className="space-y-3">
                {/* Detection Mode */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Detection Mode
                  </label>
                  <select
                    value={detectionMode}
                    onChange={(e) => setDetectionMode(e.target.value)}
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500"
                  >
                    <option value="sers_hotspots">SERS Hotspots</option>
                    <option value="cells">Cells</option>
                    <option value="tissue">Tissue Regions</option>
                    <option value="auto">Auto-detect</option>
                  </select>
                </div>

                {/* Capture & Analyze Button */}
                <button
                  onClick={handleCaptureAndAnalyze}
                  disabled={isAnalyzing || !isLive}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-afm-primary-500 text-white rounded-xl hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
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
            </div>
          </div>

          {/* Right: ROI Table & Mapping Queue */}
          <div className="space-y-4">
            {/* ROI Table */}
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-900">Detected ROIs</h3>
              {results && (
                <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
                  {results.num_rois} found
                </span>
              )}
            </div>

            {results?.rois && results.rois.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {results.rois.map((roi: ROI, idx: number) => (
                  <div
                    key={idx}
                    className={`p-3 border-2 rounded-xl transition-all cursor-pointer ${
                      selectedROIs.includes(roi)
                        ? 'border-afm-primary-500 bg-afm-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => {
                      if (selectedROIs.includes(roi)) {
                        setSelectedROIs(prev => prev.filter(r => r !== roi))
                      } else {
                        setSelectedROIs(prev => [...prev, roi])
                      }
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <MapPin className="w-5 h-5 text-afm-primary-500" />
                        <div>
                          <p className="text-sm font-medium text-gray-900">
                            ROI {idx + 1} - {roi.type.replace('_', ' ')}
                          </p>
                          <p className="text-xs text-gray-500">
                            Stage: ({roi.x.toFixed(2)}, {roi.y.toFixed(2)}) μm
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">
                          {(roi.confidence * 100).toFixed(0)}%
                        </p>
                        {selectedROIs.includes(roi) && (
                          <CheckCircle className="w-4 h-4 text-afm-primary-500 mt-1" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Target className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p className="text-sm">No ROIs detected</p>
                <p className="text-xs">Click "Capture & Analyze" to start</p>
              </div>
            )}
          </div>

          {/* Mapping Queue */}
          <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow">
            <h3 className="font-semibold text-gray-900 mb-3">Mapping Queue</h3>
            
            <div className="space-y-2 mb-3">
              <button
                onClick={() => handleAddToQueue('single')}
                disabled={selectedROIs.length === 0}
                className="w-full px-4 py-2 bg-afm-primary-50 border-2 border-afm-primary-200 text-afm-primary-700 rounded-xl hover:bg-afm-primary-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
              >
                Add Single Acquisition
              </button>
              <button
                onClick={() => handleAddToQueue('accumulate')}
                disabled={selectedROIs.length === 0}
                className="w-full px-4 py-2 bg-afm-primary-50 border-2 border-afm-primary-200 text-afm-primary-700 rounded-xl hover:bg-afm-primary-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
              >
                Add Accumulate Mode
              </button>
              <button
                onClick={() => handleAddToQueue('mapping')}
                disabled={selectedROIs.length === 0}
                className="w-full px-4 py-2 bg-afm-primary-50 border-2 border-afm-primary-200 text-afm-primary-700 rounded-xl hover:bg-afm-primary-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
              >
                Add Mapping Mode
              </button>
            </div>

            {mappingQueue.length > 0 ? (
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {mappingQueue.map((item) => (
                  <div
                    key={item.id}
                    className="p-3 bg-gray-50 border border-gray-200 rounded-lg flex items-center justify-between"
                  >
                    <div>
                      <p className="text-sm font-medium text-gray-900 capitalize">
                        {item.type}
                      </p>
                      <p className="text-xs text-gray-500">
                        {item.rois.length} ROIs
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        item.status === 'completed' ? 'bg-green-100 text-green-800' :
                        item.status === 'running' ? 'bg-blue-100 text-blue-800' :
                        item.status === 'error' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {item.status}
                      </span>
                      <button
                        onClick={() => setMappingQueue(prev => prev.filter(i => i.id !== item.id))}
                        className="p-1 hover:bg-gray-200 rounded"
                      >
                        <X className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">
                Queue is empty
              </p>
            )}

            <button
              onClick={handleAcquireAtROIs}
              disabled={selectedROIs.length === 0}
              className="w-full mt-3 px-4 py-3 bg-afm-primary-500 text-white rounded-xl hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              Acquire Spectra at Selected ROIs
            </button>
          </div>
          </div>
        </div>
      </div>
    </div>
  )
}

