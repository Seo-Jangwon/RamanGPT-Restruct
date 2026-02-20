import { useState, useRef, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'

const models = [
  { id: 'gpt4-vision', name: 'GPT-4 Vision', description: 'Best for image analysis' },
  { id: 'gpt4', name: 'GPT-4', description: 'Advanced reasoning' },
  { id: 'gpt35', name: 'GPT-3.5', description: 'Fast and efficient' },
  { id: 'claude-3', name: 'Claude 3', description: 'Long context' },
]

export default function ModelSelector() {
  const [isOpen, setIsOpen] = useState(false)
  const [selected, setSelected] = useState(models[0])
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
        className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg"
        aria-label="Select model"
        aria-expanded={isOpen}
      >
        <span className="hidden sm:inline">{selected.name}</span>
        <ChevronDown className="w-4 h-4" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden">
          <div className="p-2">
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => {
                  setSelected(model)
                  setIsOpen(false)
                }}
                className={`
                  w-full text-left px-4 py-3 rounded-lg
                  ${selected.id === model.id ? 'bg-raman-50' : 'hover:bg-gray-50'}
                `}
              >
                <div className="font-medium text-gray-900">{model.name}</div>
                <div className="text-xs text-gray-500">{model.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

