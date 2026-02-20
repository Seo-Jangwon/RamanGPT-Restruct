import { useState, useRef, useEffect } from 'react'
import { Wrench, Camera, Focus, Sparkles, Terminal, AlertCircle } from 'lucide-react'

const tools = [
  { id: 'image-raman', name: 'Image-to-Raman', icon: Camera, description: 'Analyze microscopy images' },
  { id: 'autofocus', name: 'Auto-Focus', icon: Focus, description: 'Z-optimization agent' },
  { id: 'optimization', name: 'RAG Optimizer', icon: Sparkles, description: 'Parameter recommendations' },
  { id: 'hardware', name: 'Hardware Control', icon: Terminal, description: 'Natural language commands' },
  { id: 'troubleshoot', name: 'Troubleshooting', icon: AlertCircle, description: 'Diagnostic assistant' },
]

export default function ToolsMenu() {
  const [isOpen, setIsOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      document.addEventListener('keydown', handleEscape)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen])

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 hover:bg-gray-100 rounded-full"
        aria-label="Select tool"
        aria-expanded={isOpen}
      >
        <Wrench className="w-5 h-5 text-gray-600" />
      </button>

      {isOpen && (
        <div className="absolute right-0 bottom-full mb-2 w-72 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden">
          <div className="p-2">
            <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Available Agents
            </div>
            {tools.map((tool) => {
              const Icon = tool.icon
              return (
                <button
                  key={tool.id}
                  onClick={() => {
                    console.log('Selected tool:', tool.id)
                    setIsOpen(false)
                  }}
                  className="w-full text-left px-3 py-3 rounded-lg hover:bg-gray-50 flex items-start gap-3"
                >
                  <Icon className="w-5 h-5 text-raman-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-gray-900 text-sm">{tool.name}</div>
                    <div className="text-xs text-gray-500">{tool.description}</div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

