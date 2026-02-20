import Chip from './Chip'

const actions = [
  { id: 'analyze-image', label: 'Analyze Image', color: 'raman' as const },
  { id: 'optimize', label: 'Optimize Parameters', color: 'sers' as const },
  { id: 'map', label: 'Create Map', color: 'raman' as const },
  { id: 'focus', label: 'Auto-Focus', color: 'sers' as const },
  { id: 'troubleshoot', label: 'Troubleshoot', color: 'raman' as const },
]

interface ActionChipsProps {
  onActionClick: (actionId: string) => void
}

export default function ActionChips({ onActionClick }: ActionChipsProps) {
  return (
    <div className="flex flex-wrap justify-center gap-3">
      {actions.map((action) => (
        <Chip
          key={action.id}
          label={action.label}
          onClick={() => onActionClick(action.id)}
          color={action.color}
        />
      ))}
    </div>
  )
}

