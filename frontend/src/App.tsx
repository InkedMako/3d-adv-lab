import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigPage } from '@/pages/ConfigPage';
import { ExperimentPage } from '@/pages/ExperimentPage';
import { HistoryPage } from '@/pages/HistoryPage';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ConfigPage />} />
        <Route path="/experiment" element={<ExperimentPage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </Router>
  );
}