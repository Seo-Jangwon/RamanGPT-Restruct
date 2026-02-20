import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import { Camera, Layers, Zap, Cpu, CheckCircle2, XCircle, Loader2, Circle, RotateCcw } from 'lucide-react'

type DeviceStatus = 'idle' | 'connecting' | 'connected' | 'failed' | 'skipped'

interface DeviceState {
  status: DeviceStatus
  error?: string
}

interface Props {
  onComplete: () => void
}

function StatusIcon({ status }: { status: DeviceStatus }) {
  switch (status) {
    case 'connecting': return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
    case 'connected':  return <CheckCircle2 className="w-4 h-4 text-green-500" />
    case 'failed':     return <XCircle className="w-4 h-4 text-red-400" />
    default:           return <Circle className="w-4 h-4 text-gray-300" />
  }
}

function statusText(status: DeviceStatus): string {
  switch (status) {
    case 'idle':       return '대기 중'
    case 'connecting': return '연결 중...'
    case 'connected':  return '연결됨'
    case 'failed':     return '연결 실패'
    case 'skipped':    return '건너뜀'
  }
}

function statusColor(status: DeviceStatus): string {
  switch (status) {
    case 'connecting': return 'text-blue-500'
    case 'connected':  return 'text-green-600'
    case 'failed':     return 'text-red-400'
    default:           return 'text-gray-400'
  }
}

function cardBg(status: DeviceStatus): string {
  switch (status) {
    case 'connecting': return 'border-blue-200 bg-blue-50/60'
    case 'connected':  return 'border-green-200 bg-green-50/60'
    case 'failed':     return 'border-red-200 bg-red-50/40'
    default:           return 'border-gray-200 bg-gray-50/60'
  }
}

interface DeviceCardProps {
  icon: React.ReactNode
  name: string
  detail: string
  state: DeviceState
  onRetry?: () => void
  children?: React.ReactNode
}

function DeviceCard({ icon, name, detail, state, onRetry, children }: DeviceCardProps) {
  return (
    <div className={`flex items-center gap-4 px-5 py-4 rounded-xl border transition-all duration-300 ${cardBg(state.status)}`}>
      <div className={`flex-shrink-0 ${statusColor(state.status)}`}>
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-800">{name}</div>
        <div className="text-xs text-gray-400 mt-0.5">{detail}</div>
        {children}
        {state.error && (
          <div className="text-xs text-red-400 mt-1 truncate" title={state.error}>
            {state.error}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className={`text-xs font-medium ${statusColor(state.status)}`}>
          {statusText(state.status)}
        </span>
        <StatusIcon status={state.status} />
        {onRetry && state.status === 'failed' && (
          <button
            onClick={onRetry}
            title="재시도"
            className="ml-1 p-1 rounded-md text-gray-400 hover:text-red-500 hover:bg-red-50 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  )
}

export default function SystemInitModal({ onComplete }: Props) {
  const [camera, setCamera] = useState<DeviceState>({ status: 'idle' })
  const [stage,  setStage]  = useState<DeviceState>({ status: 'idle' })
  const [laser,  setLaser]  = useState<DeviceState>({ status: 'idle' })
  const [ccd,    setCcd]    = useState<DeviceState>({ status: 'idle' })
  const [laserPort, setLaserPort] = useState('COM4')
  const [phase, setPhase]   = useState<'running' | 'done'>('running')
  const started = useRef(false)

  const connectCamera = useCallback(async () => {
    setCamera({ status: 'connecting' })
    try {
      await axios.post('/api/camera/connect', { exposure_ms: 10.0 })
      setCamera({ status: 'connected' })
    } catch (e: any) {
      setCamera({ status: 'failed', error: e.response?.data?.detail ?? e.message })
    }
  }, [])

  const connectStage = useCallback(async () => {
    setStage({ status: 'connecting' })
    try {
      await axios.post('/api/stage/connect', { dll_path: '' })
      setStage({ status: 'connected' })
    } catch (e: any) {
      setStage({ status: 'failed', error: e.response?.data?.detail ?? e.message })
    }
  }, [])

  const connectLaser = useCallback(async (port: string) => {
    setLaser({ status: 'connecting' })
    try {
      await axios.post('/api/laser/connect', { port, baud: 115200 })
      setLaser({ status: 'connected' })
    } catch (e: any) {
      setLaser({ status: 'failed', error: e.response?.data?.detail ?? e.message })
    }
  }, [])

  const connectCcd = useCallback(async () => {
    setCcd({ status: 'connecting' })
    try {
      // 서버 startup에서 이미 연결됐을 수 있으므로 status 먼저 확인
      const statusRes = await axios.get('/api/ccd/status')
      if (statusRes.data.connected) {
        setCcd({ status: 'connected' })
        return
      }
      await axios.post('/api/ccd/connect', { target_temp: -40 })
      setCcd({ status: 'connected' })
    } catch (e: any) {
      setCcd({ status: 'failed', error: e.response?.data?.detail ?? e.message })
    }
  }, [])

  const runSequence = useCallback(async (port: string) => {
    setPhase('running')
    await connectCamera()
    await connectStage()
    await connectLaser(port)
    await connectCcd()
    setPhase('done')
  }, [connectCamera, connectStage, connectLaser, connectCcd])

  useEffect(() => {
    if (started.current) return
    started.current = true
    runSequence(laserPort)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const connectedCount = [camera, stage, laser, ccd].filter(d => d.status === 'connected').length
  const isDone = phase === 'done'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-white">
      <div className="w-full max-w-md px-8 py-10">

        {/* 헤더 */}
        <div className="text-center mb-10">
          <div className="text-2xl font-bold tracking-tight text-gray-900 mb-1">RamanGPT</div>
          <div className="text-sm text-gray-400">시스템 초기화 중...</div>
        </div>

        {/* 디바이스 카드들 */}
        <div className="space-y-2.5 mb-8">

          {/* Camera */}
          <DeviceCard
            icon={<Camera className="w-5 h-5" />}
            name="카메라 (TUCam)"
            detail="노출: 10.0 ms"
            state={camera}
            onRetry={isDone ? connectCamera : undefined}
          />

          {/* Stage */}
          <DeviceCard
            icon={<Layers className="w-5 h-5" />}
            name="스테이지 (Tango)"
            detail="Tango_DLL.dll"
            state={stage}
            onRetry={isDone ? connectStage : undefined}
          />

          {/* Laser */}
          <DeviceCard
            icon={<Zap className="w-5 h-5" />}
            name="레이저"
            detail={`포트: ${laserPort}  |  Baud: 115200`}
            state={laser}
            onRetry={isDone ? () => connectLaser(laserPort) : undefined}
          >
            {/* 포트 입력 - idle / failed 상태에서만 편집 가능 */}
            {(laser.status === 'idle' || laser.status === 'failed') && isDone && (
              <div className="flex items-center gap-1.5 mt-1.5">
                <span className="text-xs text-gray-400">포트:</span>
                <input
                  type="text"
                  value={laserPort}
                  onChange={e => setLaserPort(e.target.value.toUpperCase())}
                  className="text-xs border border-gray-200 rounded px-1.5 py-0.5 w-20 focus:outline-none focus:border-blue-300"
                  placeholder="COM4"
                />
              </div>
            )}
          </DeviceCard>

          {/* CCD */}
          <DeviceCard
            icon={<Cpu className="w-5 h-5" />}
            name="CCD (Andor EMCCD)"
            detail="목표 온도: -40°C  |  쿨러 ON"
            state={ccd}
            onRetry={isDone ? connectCcd : undefined}
          />
        </div>

        {/* 연결 진행 상태 */}
        <div className="flex items-center gap-2 mb-6">
          <div className="flex-1 h-1 rounded-full bg-gray-100 overflow-hidden">
            <div
              className="h-full bg-green-400 rounded-full transition-all duration-500"
              style={{ width: `${(connectedCount / 4) * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-400 flex-shrink-0">{connectedCount} / 4 연결됨</span>
        </div>

        {/* 액션 버튼 */}
        <div className="flex gap-2.5">
          <button
            onClick={onComplete}
            className="flex-1 py-2.5 rounded-xl border border-gray-200 text-sm text-gray-500 hover:bg-gray-50 transition"
          >
            건너뛰기
          </button>
          <button
            onClick={onComplete}
            disabled={!isDone}
            className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition ${
              isDone
                ? 'bg-afm-primary-700 text-white hover:bg-afm-primary-800'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
          >
            시스템 진입 →
          </button>
        </div>

      </div>
    </div>
  )
}
