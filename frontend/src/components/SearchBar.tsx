import { useState, useRef, useEffect } from 'react'
import { Plus, Mic, Send } from 'lucide-react'
import ModelSelector from './ModelSelector'
import ToolsMenu from './ToolsMenu'

export default function SearchBar() {
  const [value, setValue] = useState('')
  const [focused, setFocused] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Focus shortcut: press '/'
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (value.trim()) {
      console.log('Submitting:', value)
      // TODO: Send to backend
      setValue('')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div
        className={`
          relative flex items-center gap-2 p-4
          bg-white border-2 rounded-3xl
          transition-all duration-200
          ${focused 
            ? 'border-raman-500 shadow-lg shadow-raman-100' 
            : 'border-gray-300 hover:border-gray-400 shadow-sm'
          }
        `}
      >
        {/* Plus button */}
        <button
          type="button"
          className="p-2 hover:bg-gray-100 rounded-full flex-shrink-0"
          aria-label="Attach file"
        >
          <Plus className="w-5 h-5 text-gray-600" />
        </button>

        {/* Text input */}
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask Raman-GPT anything... (e.g., 'Map this 10×10 μm region with 0.5 μm steps')"
          className="flex-1 resize-none outline-none text-gray-900 placeholder-gray-500 min-h-[24px] max-h-[200px]"
          rows={1}
          onInput={(e) => {
            const target = e.target as HTMLTextAreaElement
            target.style.height = 'auto'
            target.style.height = target.scrollHeight + 'px'
          }}
        />

        {/* Right side buttons */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <ToolsMenu />
          <ModelSelector />
          
          {value.trim() ? (
            <button
              type="submit"
              className="p-2 bg-raman-500 hover:bg-raman-600 text-white rounded-full"
              aria-label="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          ) : (
            <button
              type="button"
              className="p-2 hover:bg-gray-100 rounded-full"
              aria-label="Voice input"
            >
              <Mic className="w-5 h-5 text-gray-600" />
            </button>
          )}
        </div>
      </div>

      {/* Auto-suggestions */}
      {focused && !value && (
        <div className="mt-2 p-4 bg-white border border-gray-200 rounded-xl shadow-lg">
          <p className="text-xs text-gray-500 mb-2">Try asking:</p>
          <div className="space-y-1">
            {[
              'Analyze this microscopy image and select SERS hotspots',
              'Optimize laser power for cell imaging',
              'Troubleshoot weak signal issue',
              'Auto-focus and acquire spectrum at this point'
            ].map((suggestion, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setValue(suggestion)}
                className="block w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
    </form>
  )
}

