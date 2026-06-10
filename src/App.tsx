import { Navigate, Route, Routes } from 'react-router-dom'
import { TopBar } from './components/TopBar'
import { IntroPage } from './pages/IntroPage'
import { SurveyPage } from './pages/SurveyPage'
import { ResultPage } from './pages/ResultPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { ExperimenterPage } from './pages/ExperimenterPage'
import { ColorOrderPage } from './pages/ColorOrderPage'

export default function App() {
  return (
    <div className="min-h-screen">
      <TopBar />
      <Routes>
        <Route path="/" element={<IntroPage />} />
        <Route path="/survey" element={<SurveyPage />} />
        <Route path="/result" element={<ResultPage />} />

        {/* 실험자용 Stroop 기준 안내 */}
        <Route path="/experimenter" element={<ExperimenterPage />} />

        {/* 실험자용 색깔 순서 랜덤 제시 */}
        <Route path="/color-order" element={<ColorOrderPage />} />

        <Route path="/home" element={<Navigate to="/" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
      <div className="mx-auto w-full max-w-3xl px-4 pb-10 text-xs text-zinc-400">
        © {new Date().getFullYear()} VELA · Research tooling UI
      </div>
    </div>
  )
}