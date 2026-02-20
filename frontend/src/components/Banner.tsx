import { X, Zap } from 'lucide-react'

interface BannerProps {
  onDismiss: () => void
}

export default function Banner({ onDismiss }: BannerProps) {
  return (
    <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-30 max-w-2xl w-full mx-4">
      <div className="bg-gradient-to-r from-raman-500 to-sers-500 text-white rounded-2xl shadow-2xl p-4 flex items-center gap-4">
        <Zap className="w-6 h-6 flex-shrink-0" />
        
        <div className="flex-1">
          <p className="font-medium">
            New: Real-time SERS optimization with GPT-4 Vision
          </p>
          <p className="text-sm text-white/90">
            Automatically detect hotspots and optimize acquisition parameters
          </p>
        </div>

        <button
          onClick={() => console.log('Learn more clicked')}
          className="px-4 py-2 bg-white text-raman-600 rounded-lg font-medium hover:bg-gray-100 flex-shrink-0"
        >
          Learn more
        </button>

        <button
          onClick={onDismiss}
          className="p-1 hover:bg-white/20 rounded-lg flex-shrink-0"
          aria-label="Dismiss banner"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
    </div>
  )
}

