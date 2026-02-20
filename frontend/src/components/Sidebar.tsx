import {
  Sparkles,
  Clock,
  Settings,
  X,
  Camera,
  Focus,
  Sparkles as SparklesIcon,
  Terminal,
  AlertCircle,
} from 'lucide-react'
import type { PageId } from '../App'
import logoImage from '../logo/logo.png'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onPageSelect: (id: PageId) => void
}

export default function Sidebar({ isOpen, onClose, onPageSelect }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-64 bg-white/60 backdrop-blur-md border-r border-gray-200/50
          transform transition-transform duration-200 ease-in-out
          flex flex-col shadow-sm
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
        aria-label="Navigation sidebar"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img 
              src={logoImage} 
              alt="Raman-GPT Logo" 
              className="w-6 h-6 object-contain"
            />
            <h1 className="text-xl font-semibold text-gray-900">Raman-GPT</h1>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-2 hover:bg-gray-200 rounded-lg"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* AFM Agent section - Moved to top */}
        <div className="p-3 border-b border-gray-200">
          <h2 className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            AFM Agent
          </h2>
          <nav className="space-y-1" aria-label="AFM Agent">
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-afm-primary-500 hover:bg-afm-primary-50 rounded-lg focus:ring-2 focus:ring-afm-primary-500 font-medium"
              onClick={() => {
                onPageSelect('afm')
                onClose()
              }}
            >
              <Sparkles className="w-4 h-4 text-afm-primary-500" />
              <span>AFM Dashboard</span>
            </button>
          </nav>
        </div>

        {/* Agents section */}
        <div className="flex-1 overflow-y-auto p-3">
          <h2 className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Agents
          </h2>
          <nav className="space-y-1" aria-label="Agents">
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-gray-800 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
              onClick={() => {
                onPageSelect('afm-image-raman')
                onClose()
              }}
            >
              <Camera className="w-4 h-4 text-raman-500" />
              Image-to-Raman
            </button>
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-gray-800 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
              onClick={() => {
                onPageSelect('afm-autofocus')
                onClose()
              }}
            >
              <Focus className="w-4 h-4 text-sers-500" />
              Auto-Focus
            </button>
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-gray-800 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
              onClick={() => {
                onPageSelect('afm-optimization')
                onClose()
              }}
            >
              <SparklesIcon className="w-4 h-4 text-raman-500" />
              RAG Optimization
            </button>
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-gray-800 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
              onClick={() => {
                onPageSelect('afm-hardware')
                onClose()
              }}
            >
              <Terminal className="w-4 h-4 text-gray-700" />
              Hardware Control
            </button>
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-gray-800 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
              onClick={() => {
                onPageSelect('afm-troubleshooting')
                onClose()
              }}
            >
              <AlertCircle className="w-4 h-4 text-red-500" />
              Troubleshooting
            </button>
          </nav>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-200 space-y-1">
          <button
            className="w-full flex items-center gap-3 px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
            aria-label="View activity"
          >
            <Clock className="w-5 h-5" />
            Activity
          </button>
          
          <button
            className="w-full flex items-center gap-3 px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
            aria-label="Open settings"
          >
            <Settings className="w-5 h-5" />
            Settings
          </button>
        </div>
      </aside>
    </>
  )
}

