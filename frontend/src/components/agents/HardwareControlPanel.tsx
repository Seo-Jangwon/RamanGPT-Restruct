import { useState } from 'react'
import { Terminal, Send, Loader2, CheckCircle, XCircle } from 'lucide-react'
import axios from 'axios'

interface CommandHistory {
  command: string
  response: string
  success: boolean
  timestamp: Date
}

export default function HardwareControlPanel() {
  const [command, setCommand] = useState('')
  const [isExecuting, setIsExecuting] = useState(false)
  const [history, setHistory] = useState<CommandHistory[]>([])

  const exampleCommands = [
    'Move stage to X=100, Y=200',
    'Set laser power to 10 mW',
    'Map this 10×10 μm region with 1 μm steps',
    'Acquire spectrum with 2 second exposure',
    'Move stage 50 microns to the right',
    'Turn laser on',
    'Set grating to 1200 gr/mm'
  ]

  const handleExecute = async () => {
    if (!command.trim()) return

    setIsExecuting(true)
    const startTime = new Date()

    try {
      const response = await axios.post('/api/hardware-command', {
        command: command
      })

      setHistory(prev => [{
        command,
        response: response.data.message || 'Command executed successfully',
        success: true,
        timestamp: startTime
      }, ...prev])

      setCommand('')
    } catch (error: any) {
      setHistory(prev => [{
        command,
        response: error.response?.data?.detail || 'Failed to execute command',
        success: false,
        timestamp: startTime
      }, ...prev])
    } finally {
      setIsExecuting(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleExecute()
    }
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h3 className="font-semibold text-green-900 mb-2">Hardware Control Agent</h3>
        <p className="text-sm text-green-800">
          Control the Raman system using natural language commands. Just type what you want
          to do, and the AI will translate it into hardware API calls.
        </p>
      </div>

      {/* Command Input */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Natural Language Command
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="e.g., Move stage to X=100, Y=200..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-raman-500 focus:border-transparent"
          />
          <button
            onClick={handleExecute}
            disabled={isExecuting || !command.trim()}
            className="px-6 py-3 bg-raman-500 text-white rounded-lg hover:bg-raman-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isExecuting ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Example Commands */}
      <div>
        <p className="text-sm font-medium text-gray-700 mb-2">Example Commands:</p>
        <div className="flex flex-wrap gap-2">
          {exampleCommands.map((cmd, idx) => (
            <button
              key={idx}
              onClick={() => setCommand(cmd)}
              className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-lg transition-colors"
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>

      {/* Command History */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Terminal className="w-4 h-4 text-gray-600" />
          <h4 className="text-sm font-semibold text-gray-900">Command History</h4>
        </div>

        {history.length === 0 ? (
          <div className="bg-gray-50 rounded-lg p-8 text-center">
            <Terminal className="w-12 h-12 text-gray-300 mx-auto mb-2" />
            <p className="text-sm text-gray-500">No commands executed yet</p>
            <p className="text-xs text-gray-400 mt-1">Try one of the example commands above</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {history.map((item, idx) => (
              <div
                key={idx}
                className={`rounded-lg p-4 border ${
                  item.success
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex items-start gap-3">
                  {item.success ? (
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 mb-1">
                      $ {item.command}
                    </p>
                    <p className={`text-sm ${
                      item.success ? 'text-green-800' : 'text-red-800'
                    }`}>
                      {item.response}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {item.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm font-semibold text-blue-900 mb-2">Quick Actions</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setCommand('Turn laser on')}
            className="px-3 py-2 bg-white border border-blue-200 text-blue-900 text-sm rounded-lg hover:bg-blue-100 transition-colors"
          >
            🔦 Laser On
          </button>
          <button
            onClick={() => setCommand('Turn laser off')}
            className="px-3 py-2 bg-white border border-blue-200 text-blue-900 text-sm rounded-lg hover:bg-blue-100 transition-colors"
          >
            🔦 Laser Off
          </button>
          <button
            onClick={() => setCommand('Move to home position')}
            className="px-3 py-2 bg-white border border-blue-200 text-blue-900 text-sm rounded-lg hover:bg-blue-100 transition-colors"
          >
            🏠 Home
          </button>
          <button
            onClick={() => setCommand('Acquire single spectrum')}
            className="px-3 py-2 bg-white border border-blue-200 text-blue-900 text-sm rounded-lg hover:bg-blue-100 transition-colors"
          >
            📊 Acquire
          </button>
        </div>
      </div>
    </div>
  )
}

