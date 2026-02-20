import { useEffect, useState } from 'react'
import axios from 'axios'
import { Thermometer } from 'lucide-react'

interface CCDStatus {
  connected: boolean
  temperature: number | null
  temp_status: string | null
}

type StatusKey =
  | 'STABILIZED'
  | 'NOT_STABILIZED'
  | 'NOT_REACHED'
  | 'DRIFT'
  | 'OFF'
  | 'NOT_SUPPORTED'
  | 'disconnected'

const STATUS_STYLES: Record<
  StatusKey,
  { bg: string; dot: string; label: string; text: string }
> = {
  STABILIZED: {
    bg: 'bg-emerald-950',
    dot: 'bg-emerald-400',
    label: 'Stabilized',
    text: 'text-emerald-400',
  },
  NOT_STABILIZED: {
    bg: 'bg-sky-950',
    dot: 'bg-sky-400 animate-pulse',
    label: 'Stabilizing…',
    text: 'text-sky-400',
  },
  NOT_REACHED: {
    bg: 'bg-sky-950',
    dot: 'bg-sky-400 animate-pulse',
    label: 'Cooling…',
    text: 'text-sky-400',
  },
  DRIFT: {
    bg: 'bg-amber-950',
    dot: 'bg-amber-400 animate-pulse',
    label: 'Drift',
    text: 'text-amber-400',
  },
  OFF: {
    bg: 'bg-gray-900',
    dot: 'bg-gray-500',
    label: 'Cooler Off',
    text: 'text-gray-400',
  },
  NOT_SUPPORTED: {
    bg: 'bg-gray-900',
    dot: 'bg-gray-500',
    label: 'Not Supported',
    text: 'text-gray-400',
  },
  disconnected: {
    bg: 'bg-gray-900',
    dot: 'bg-gray-700',
    label: 'Not connected',
    text: 'text-gray-600',
  },
}

export default function CCDStatusBar() {
  const [status, setStatus] = useState<CCDStatus>({
    connected: false,
    temperature: null,
    temp_status: null,
  })

  useEffect(() => {
    const poll = async () => {
      try {
        const { data } = await axios.get<CCDStatus>('/api/ccd/status', {
          timeout: 1500,
        })
        setStatus(data)
      } catch {
        setStatus({ connected: false, temperature: null, temp_status: null })
      }
    }

    poll()
    const id = setInterval(poll, 2000)
    return () => clearInterval(id)
  }, [])

  const key: StatusKey =
    !status.connected
      ? 'disconnected'
      : ((status.temp_status as StatusKey | null) ?? 'OFF')

  const cfg = STATUS_STYLES[key] ?? STATUS_STYLES.disconnected

  return (
    <div
      className={`w-full flex items-center justify-between px-4 py-1 shrink-0 transition-colors duration-700 ${cfg.bg}`}
    >
      {/* 왼쪽: 온도 + 상태 */}
      <div className="flex items-center gap-2.5 text-xs font-mono">
        <Thermometer className={`w-3.5 h-3.5 shrink-0 ${cfg.text}`} />
        <span className="text-gray-600 uppercase tracking-widest text-[10px]">CCD</span>
        <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${cfg.dot}`} />
        {status.connected ? (
          <>
            <span className={`font-semibold ${cfg.text}`}>
              {status.temperature !== null ? `${status.temperature} °C` : '—'}
            </span>
            <span className={`opacity-60 ${cfg.text}`}>{cfg.label}</span>
          </>
        ) : (
          <span className={cfg.text}>{cfg.label}</span>
        )}
      </div>

      {/* 오른쪽: 기기 식별 */}
      <span className="text-[10px] text-gray-700 font-mono tracking-wider uppercase select-none">
        Andor EMCCD
      </span>
    </div>
  )
}
