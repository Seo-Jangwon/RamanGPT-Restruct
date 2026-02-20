import { useState } from 'react'
import Sidebar from './components/Sidebar'
import MainContent from './components/MainContent'
import Banner from './components/Banner'
import SystemInitModal from './components/SystemInitModal'

export type PageId =
  | 'home'
  | 'image-raman'
  | 'autofocus'
  | 'optimization'
  | 'hardware'
  | 'troubleshooting'
  | 'afm'
  | 'afm-image-raman'
  | 'afm-autofocus'
  | 'afm-optimization'
  | 'afm-hardware'
  | 'afm-troubleshooting'

function App() {
  const [initialized, setInitialized] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [bannerVisible, setBannerVisible] = useState(true)
  const [activePage, setActivePage] = useState<PageId>('home')

  // Keep sidebar visible but make it more subtle for AFM pages
  const isAFMPage = activePage.startsWith('afm')
  const shouldShowSidebar = sidebarOpen

  if (!initialized) {
    return <SystemInitModal onComplete={() => setInitialized(true)} />
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-raman-500 focus:text-white focus:rounded"
      >
        Skip to content
      </a>

      {/* Sidebar - Always visible but subtle for AFM pages */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onPageSelect={(id) => setActivePage(id)}
      />

      {/* Main content area */}
      <main
        id="main-content"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <MainContent
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={shouldShowSidebar}
          activePage={activePage}
          onPageSelect={setActivePage}
        />
      </main>

      {/* Bottom banner - Hidden for AFM pages */}
      {bannerVisible && !isAFMPage && (
        <Banner onDismiss={() => setBannerVisible(false)} />
      )}
    </div>
  )
}

export default App

