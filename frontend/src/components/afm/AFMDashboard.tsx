import { useState, useEffect } from 'react'
import { Camera, Focus, Sparkles, Terminal, AlertCircle, ArrowLeft, Menu } from 'lucide-react'
import AFMImageToRamanPanel from './AFMImageToRamanPanel'
import AFMAutoFocusPanel from './AFMAutoFocusPanel'
import AFMRAGOptimizerPanel from './AFMRAGOptimizerPanel'
import AFMHardwareControlPanel from './AFMHardwareControlPanel'
import AFMTroubleshootingPanel from './AFMTroubleshootingPanel'

type AFMModule = 'overview' | 'image-raman' | 'autofocus' | 'optimization' | 'hardware' | 'troubleshooting'

interface AFMDashboardProps {
  initialModule?: AFMModule
}

export default function AFMDashboard({ initialModule = 'overview' }: AFMDashboardProps) {
  const [activeModule, setActiveModule] = useState<AFMModule>(initialModule)

  // Update active module when initialModule prop changes (e.g., from URL/routing)
  useEffect(() => {
    if (initialModule) {
      setActiveModule(initialModule)
    }
  }, [initialModule])

  const modules = [
    { id: 'image-raman' as AFMModule, name: 'Image-to-Raman Agent', icon: Camera, description: 'ROI detection and spectrum acquisition' },
    { id: 'autofocus' as AFMModule, name: 'Auto-Focus & Z-Optimization', icon: Focus, description: 'Z-scan and focus optimization' },
    { id: 'optimization' as AFMModule, name: 'Raman/SERS RAG Optimization', icon: Sparkles, description: 'AI-powered parameter recommendations' },
    { id: 'hardware' as AFMModule, name: 'LLM-Based Hardware Control', icon: Terminal, description: 'Natural language hardware commands' },
    { id: 'troubleshooting' as AFMModule, name: 'Troubleshooting Agent', icon: AlertCircle, description: 'Diagnostic and problem-solving' },
  ]

  const renderModule = () => {
    switch (activeModule) {
      case 'image-raman':
        return <AFMImageToRamanPanel />
      case 'autofocus':
        return <AFMAutoFocusPanel />
      case 'optimization':
        return <AFMRAGOptimizerPanel />
      case 'hardware':
        return <AFMHardwareControlPanel />
      case 'troubleshooting':
        return <AFMTroubleshootingPanel />
      case 'overview':
      default:
        return (
          <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-12">
            {/* Centered Title */}
            <div className="text-center space-y-4 max-w-2xl">
              <h1 className="text-5xl sm:text-6xl font-light text-gray-900 tracking-tight">
                AFM Agent
              </h1>
              <p className="text-lg text-gray-600 font-light">
                Integrated Raman spectroscopy control system with AI-powered agents
              </p>
            </div>

            {/* Centered Module Cards - Modern Gemini Style */}
            <div className="w-full max-w-5xl">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {modules.map((module) => {
                  const Icon = module.icon
                  return (
                    <button
                      key={module.id}
                      onClick={() => setActiveModule(module.id)}
                      className="group relative bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-2xl p-8 hover:border-afm-primary-500/50 hover:shadow-2xl hover:shadow-afm-primary-500/10 transition-all duration-300 text-left hover:scale-[1.02] hover:bg-white/90"
                    >
                      {/* Subtle gradient overlay on hover */}
                      <div className="absolute inset-0 bg-gradient-to-br from-afm-primary-500/5 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      
                      <div className="relative flex flex-col items-start space-y-4">
                        <div className="p-4 bg-gradient-to-br from-afm-primary-50 to-afm-primary-100/50 rounded-xl group-hover:from-afm-primary-100 group-hover:to-afm-primary-200/50 transition-all duration-300 shadow-sm">
                          <Icon className="w-8 h-8 text-afm-primary-500" />
                        </div>
                        <div className="space-y-2">
                          <h3 className="text-xl font-medium text-gray-900 group-hover:text-afm-primary-700 transition-colors">
                            {module.name}
                          </h3>
                          <p className="text-sm text-gray-600 leading-relaxed">
                            {module.description}
                          </p>
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-afm-primary-50 via-afm-primary-50/80 to-afm-primary-100/40 relative">
      {/* Modern background pattern overlay with pastel key color */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(113,20,30,0.04),transparent_50%)] pointer-events-none" />
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(113,20,30,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(113,20,30,0.03)_1px,transparent_1px)] bg-[size:4rem_4rem] pointer-events-none opacity-30" />
      
      {/* Minimal Top Navigation - Gemini Style */}
      {activeModule !== 'overview' && (
        <div className="sticky top-0 z-10 bg-white/60 backdrop-blur-md border-b border-gray-200/50 shadow-sm">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <button
              onClick={() => setActiveModule('overview')}
              className="flex items-center gap-2 text-sm text-gray-700 hover:text-afm-primary-500 transition-colors group"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
              Back to Overview
            </button>
          </div>
        </div>
      )}

      {/* Centered Content Area - Gemini Style */}
      <div className="relative max-w-7xl mx-auto px-6 py-12">
        {renderModule()}
      </div>
    </div>
  )
}

