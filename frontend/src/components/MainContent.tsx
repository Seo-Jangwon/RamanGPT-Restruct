import { Menu } from 'lucide-react'
import SearchBar from './SearchBar'
import ActionChips from './ActionChips'
import IconButton from './IconButton'
import AFMDashboard from './afm/AFMDashboard'
import type { PageId } from '../App'

interface MainContentProps {
  onMenuClick: () => void
  sidebarOpen: boolean
  activePage: PageId
  onPageSelect: (id: PageId) => void
}

export default function MainContent({
  onMenuClick,
  sidebarOpen,
  activePage,
  onPageSelect,
}: MainContentProps) {
  const handleActionClick = (actionId: string) => {
    switch (actionId) {
      case 'analyze-image':
        onPageSelect('afm-image-raman')
        break
      case 'focus':
        onPageSelect('afm-autofocus')
        break
      case 'optimize':
        onPageSelect('afm-optimization')
        break
      case 'map':
        onPageSelect('afm-hardware')
        break
      case 'troubleshoot':
        onPageSelect('afm-troubleshooting')
        break
      default:
        onPageSelect('home')
    }
  }
  // Map page IDs to AFM module names
  const getAFMModule = (pageId: PageId): 'overview' | 'image-raman' | 'autofocus' | 'optimization' | 'hardware' | 'troubleshooting' => {
    switch (pageId) {
      case 'image-raman':
      case 'afm-image-raman':
        return 'image-raman'
      case 'autofocus':
      case 'afm-autofocus':
        return 'autofocus'
      case 'optimization':
      case 'afm-optimization':
        return 'optimization'
      case 'hardware':
      case 'afm-hardware':
        return 'hardware'
      case 'troubleshooting':
      case 'afm-troubleshooting':
        return 'troubleshooting'
      case 'afm':
        return 'overview'
      default:
        return 'overview'
    }
  }

  const renderPage = () => {
    switch (activePage) {
      // All agent pages now use AFM Dashboard with unified UI
      case 'image-raman':
      case 'autofocus':
      case 'optimization':
      case 'hardware':
      case 'troubleshooting':
      case 'afm':
      case 'afm-image-raman':
      case 'afm-autofocus':
      case 'afm-optimization':
      case 'afm-hardware':
      case 'afm-troubleshooting':
        return (
          <div className="flex-1 flex flex-col w-full">
            <AFMDashboard initialModule={getAFMModule(activePage)} />
          </div>
        )
      case 'home':
      default:
        return (
          <div className="flex-1 flex flex-col items-center justify-center px-4 py-8 max-w-4xl mx-auto w-full">
            {/* Logo/Title */}
            <div className="text-center mb-12">
              <h1 className="text-5xl sm:text-6xl font-light text-gray-900 mb-4">
                Raman<span className="text-raman-500">-GPT</span>
              </h1>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                AI-powered Raman spectroscopy system with automated imaging,
                optimization, and intelligent troubleshooting
              </p>
            </div>

            {/* Search bar */}
            <div className="w-full mb-8">
              <SearchBar />
            </div>

            {/* Action chips */}
            <ActionChips onActionClick={handleActionClick} />

            {/* Feature highlights */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-12 w-full">
              <FeatureCard
                title="Image-to-Raman"
                description="Auto-detect hotspots and acquire spectra at precise coordinates"
                color="raman"
              />
              <FeatureCard
                title="Auto-Focus"
                description="Hybrid optical + Raman intensity optimization for perfect Z-position"
                color="sers"
              />
              <FeatureCard
                title="RAG Optimization"
                description="AI-recommended parameters based on sample type and literature"
                color="raman"
              />
              <FeatureCard
                title="Natural Language Control"
                description="Control hardware with simple commands like 'Map this 10×10 μm region'"
                color="sers"
              />
            </div>
          </div>
        )
    }
  }

  const isAFMPage = activePage.startsWith('afm')

  return (
    <div className="flex-1 flex flex-col overflow-y-auto">
      {/* Header - Minimal for AFM pages */}
      <header className={`sticky top-0 z-10 border-b transition-all ${
        isAFMPage 
          ? 'bg-white/40 backdrop-blur-md border-gray-200/50' 
          : 'bg-white/80 backdrop-blur-sm border-gray-200'
      }`}>
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            {!sidebarOpen && (
              <IconButton
                icon={Menu}
                onClick={onMenuClick}
                label="Open sidebar"
              />
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className={`text-sm hidden sm:inline ${
              isAFMPage ? 'text-gray-700' : 'text-gray-600'
            }`}>
              Olympus BX + RAONSpec
            </span>
            <div className="w-2 h-2 rounded-full bg-green-500" title="System online" />
          </div>
        </div>
      </header>

      {/* Page content */}
      {renderPage()}
    </div>
  )
}

interface FeatureCardProps {
  title: string
  description: string
  color: 'raman' | 'sers'
}

function FeatureCard({ title, description, color }: FeatureCardProps) {
  const colorClasses = color === 'raman' 
    ? 'border-raman-200 bg-raman-50 hover:bg-raman-100' 
    : 'border-sers-200 bg-sers-50 hover:bg-sers-100'
  
  return (
    <div className={`p-6 border-2 rounded-xl ${colorClasses} transition-all hover:shadow-md`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-sm text-gray-600">{description}</p>
    </div>
  )
}

