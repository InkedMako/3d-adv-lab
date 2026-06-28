import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHistory, getHistoryDetail, ExperimentResult } from '@/api/experiment';
import { Clock, Image, Layers, CheckCircle, XCircle, BarChart3, X, Calendar, Target, Shield } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// 详情模态框组件
function DetailModal({ result, onClose }: { result: ExperimentResult | null; onClose: () => void }) {
  if (!result) return null;

  // 准备置信度图表数据
  const chartData = [
    { name: '原始', value: result.predictions.original?.confidence || 0, color: '#00d9ff' },
    { name: '攻击后', value: result.predictions.attacked?.confidence || 0, color: '#ff4444' },
    { name: '防御后', value: result.predictions.defended?.confidence || 0, color: '#44ff44' },
  ];

  return (
    <div 
      className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[#1a1a2e] rounded-2xl max-w-5xl w-full max-h-[90vh] overflow-y-auto border border-cyan-500/30 shadow-2xl">
        {/* 模态框头部 */}
        <div className="sticky top-0 bg-[#1a1a2e] border-b border-cyan-500/20 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-cyan-500/20 flex items-center justify-center">
              <Target className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold">实验详情</h2>
              <p className="text-sm text-gray-400">样本 {result.sample_id}</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-[#0a0a1a] hover:bg-red-500/20 flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5 text-gray-400 hover:text-red-400" />
          </button>
        </div>

        {/* 模态框内容 */}
        <div className="p-6 space-y-6">
          {/* 基本信息 */}
          <div className="bg-[#0a0a1a] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Calendar className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold">实验参数</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-[#1a1a2e] rounded-lg p-3">
                <div className="text-gray-400 mb-1">样本编号</div>
                <div className="text-cyan-400 font-medium">{result.sample_id}</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3">
                <div className="text-gray-400 mb-1">实验时间</div>
                <div className="text-cyan-400 font-medium">{result.timestamp || result.id.split('_').slice(-2).join('_')}</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3">
                <div className="text-gray-400 mb-1">攻击方法</div>
                <div className="text-cyan-400 font-medium">{result.attack_method}</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3">
                <div className="text-gray-400 mb-1">防御方法</div>
                <div className="text-cyan-400 font-medium">{result.defense_method}</div>
              </div>
            </div>
          </div>

          {/* 置信度分析 */}
          <div className="bg-[#0a0a1a] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold">置信度分析</h3>
            </div>
            
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-gradient-to-br from-cyan-500/20 to-transparent rounded-xl p-5 text-center border border-cyan-500/30">
                <div className="text-sm text-gray-400 mb-2">原始识别</div>
                <div className="text-3xl font-bold text-cyan-400 mb-1">{result.predictions.original?.class || '-'}</div>
                <div className="text-lg text-white">{((result.predictions.original?.confidence || 0) * 100).toFixed(1)}%</div>
              </div>
              <div className="bg-gradient-to-br from-red-500/20 to-transparent rounded-xl p-5 text-center border border-red-500/30">
                <div className="text-sm text-gray-400 mb-2">攻击后识别</div>
                <div className="text-3xl font-bold text-red-400 mb-1">{result.predictions.attacked?.class || '-'}</div>
                <div className="text-lg text-white">{((result.predictions.attacked?.confidence || 0) * 100).toFixed(1)}%</div>
              </div>
              <div className="bg-gradient-to-br from-green-500/20 to-transparent rounded-xl p-5 text-center border border-green-500/30">
                <div className="text-sm text-gray-400 mb-2">防御后识别</div>
                <div className="text-3xl font-bold text-green-400 mb-1">{result.predictions.defended?.class || '-'}</div>
                <div className="text-lg text-white">{((result.predictions.defended?.confidence || 0) * 100).toFixed(1)}%</div>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" />
                <XAxis dataKey="name" stroke="#888" />
                <YAxis domain={[0, 1]} stroke="#888" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #00d9ff', borderRadius: '8px' }}
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 图像对比 */}
          <div className="bg-[#0a0a1a] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Image className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold">图像对比</h3>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-cyan-500/10">
                <div className="text-sm text-cyan-400 mb-2 text-center font-medium">原始</div>
                {result.images.original ? (
                  <img
                    src={`/files/${result.images.original}`}
                    alt="原始"
                    className="w-full rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400 text-center py-8">暂无图像</div>
                )}
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-red-500/10">
                <div className="text-sm text-red-400 mb-2 text-center font-medium">攻击后</div>
                {result.images.attacked ? (
                  <img
                    src={`/files/${result.images.attacked}`}
                    alt="攻击后"
                    className="w-full rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400 text-center py-8">暂无图像</div>
                )}
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-green-500/10">
                <div className="text-sm text-green-400 mb-2 text-center font-medium">防御后</div>
                {result.images.defended ? (
                  <img
                    src={`/files/${result.images.defended}`}
                    alt="防御后"
                    className="w-full rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400 text-center py-8">暂无图像</div>
                )}
              </div>
            </div>
          </div>

          {/* BEV对比 */}
          <div className="bg-[#0a0a1a] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold">BEV对比</h3>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-cyan-500/10">
                <div className="text-sm text-cyan-400 mb-2 text-center font-medium">原始</div>
                {result.bevs.original ? (
                  <img
                    src={`/files/${result.bevs.original}`}
                    alt="原始BEV"
                    className="w-full rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400 text-center py-8">暂无BEV</div>
                )}
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-red-500/10">
                <div className="text-sm text-red-400 mb-2 text-center font-medium">攻击后</div>
                {result.bevs.attacked ? (
                  <img
                    src={`/files/${result.bevs.attacked}`}
                    alt="攻击后BEV"
                    className="w-full rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400 text-center py-8">暂无BEV</div>
                )}
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-green-500/10">
                <div className="text-sm text-green-400 mb-2 text-center font-medium">防御后</div>
                {result.bevs.defended ? (
                  <img
                    src={`/files/${result.bevs.defended}`}
                    alt="防御后BEV"
                    className="w-full rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400 text-center py-8">暂无BEV</div>
                )}
              </div>
            </div>
          </div>

          {/* 综合对比图 */}
          {result.images.comparison && (
            <div className="bg-[#0a0a1a] rounded-xl p-5 border border-cyan-500/20">
              <h3 className="text-lg font-semibold mb-4">综合对比图</h3>
              <img
                src={`/files/${result.images.comparison}`}
                alt="综合对比"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function HistoryPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<ExperimentResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    setLoading(true);
    getHistory()
      .then((data) => {
        // 按时间排序（最新在前）
        const sorted = [...data].sort((a, b) => {
          // 从id中提取时间戳 (格式: sampleid_YYYYMMDD_HHMMSS)
          const timeA = extractTimestamp(a.id, a.timestamp);
          const timeB = extractTimestamp(b.id, b.timestamp);
          return timeB - timeA; // 最新的在前
        });
        setHistory(sorted);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // 从id或timestamp提取可比较的时间戳
  const extractTimestamp = (id: string, timestamp: string): number => {
    if (timestamp) {
      const clean = timestamp.replace(/-/g, '').replace(/_/g, '').replace(/:/g, '').replace(/ /g, '');
      const parsed = parseInt(clean);
      if (!isNaN(parsed)) return parsed;
    }
    const idParts = id.split('_');
    const datePart = idParts[idParts.length - 2];
    const timePart = idParts[idParts.length - 1];
    if (datePart && timePart && /^\d+$/.test(datePart) && /^\d+$/.test(timePart)) {
      return parseInt(datePart + timePart);
    }
    return 0;
  };

  const handleSelect = async (id: string) => {
    const detail = await getHistoryDetail(id);
    setSelectedResult(detail);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setSelectedResult(null);
  };

  // 格式化显示时间
  const formatDisplayTime = (id: string, timestamp: string): string => {
    if (timestamp) {
      if (timestamp.includes('-') && timestamp.includes(':')) {
        return timestamp;
      }
      const parts = timestamp.split('_');
      if (parts.length === 2) {
        const date = parts[0];
        const time = parts[1];
        return `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)} ${time.slice(0, 2)}:${time.slice(2, 4)}:${time.slice(4, 6)}`;
      }
      return timestamp;
    }
    const idParts = id.split('_');
    const datePart = idParts[idParts.length - 2];
    const timePart = idParts[idParts.length - 1];
    if (datePart && timePart && /^\d+$/.test(datePart) && /^\d+$/.test(timePart)) {
      return `${datePart.slice(0, 4)}-${datePart.slice(4, 6)}-${datePart.slice(6, 8)} ${timePart.slice(0, 2)}:${timePart.slice(2, 4)}:${timePart.slice(4, 6)}`;
    }
    return id;
  };

  return (
    <div className="min-h-screen bg-[#0a0a1a] text-white">
      {/* 顶部导航 */}
      <header className="border-b border-cyan-500/30 bg-[#1a1a2e]/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-wider" style={{ fontFamily: 'Orbitron, sans-serif' }}>
            <span className="text-cyan-400">3D</span> 对抗鲁棒性研究平台
          </h1>
          <nav className="flex gap-6">
            <button onClick={() => navigate('/')} className="text-gray-400 hover:text-cyan-400 transition-colors">实验配置</button>
            <button onClick={() => navigate('/experiment')} className="text-gray-400 hover:text-cyan-400 transition-colors">实验运行</button>
            <button onClick={() => navigate('/history')} className="text-cyan-400 hover:text-cyan-300 transition-colors">历史记录</button>
          </nav>
        </div>
      </header>

      {/* 主内容区 */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* 页面标题 */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/20 flex items-center justify-center">
            <Clock className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold">实验历史</h2>
            <p className="text-gray-400">按时间排序，最新实验在前</p>
          </div>
        </div>

        {/* 实验记录网格 */}
        {loading ? (
          <div className="text-gray-400 animate-pulse py-16 text-center">加载中...</div>
        ) : history.length === 0 ? (
          <div className="text-gray-400 py-16 text-center">暂无实验记录</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {history.map((exp) => (
              <div
                key={exp.id}
                onClick={() => handleSelect(exp.id)}
                className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20 hover:border-cyan-500/40 hover:bg-cyan-500/5 transition-all cursor-pointer group"
              >
                {/* 时间戳 */}
                <div className="flex items-center gap-2 mb-3">
                  <Calendar className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm text-cyan-400 font-medium">
                    {formatDisplayTime(exp.id, exp.timestamp)}
                  </span>
                </div>

                {/* 样本和方法 */}
                <div className="mb-3">
                  <div className="text-xl font-bold mb-2">样本 {exp.sample_id}</div>
                  <div className="flex items-center gap-3 text-sm text-gray-400">
                    <div className="flex items-center gap-1">
                      <Target className="w-4 h-4" />
                      <span>{exp.attack_method}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Shield className="w-4 h-4" />
                      <span>{exp.defense_method}</span>
                    </div>
                  </div>
                </div>

                {/* 成功状态 */}
                <div className="flex items-center gap-3">
                  <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
                    exp.attack_success 
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                      : 'bg-red-500/20 text-red-400 border border-red-500/30'
                  }`}>
                    {exp.attack_success ? (
                      <CheckCircle className="w-3.5 h-3.5" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5" />
                    )}
                    <span>{exp.attack_success ? '攻击成功' : '攻击失败'}</span>
                  </div>
                  <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
                    exp.defense_success 
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                      : 'bg-red-500/20 text-red-400 border border-red-500/30'
                  }`}>
                    {exp.defense_success ? (
                      <CheckCircle className="w-3.5 h-3.5" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5" />
                    )}
                    <span>{exp.defense_success ? '防御成功' : '防御失败'}</span>
                  </div>
                </div>

                {/* 点击提示 */}
                <div className="mt-4 text-center text-sm text-gray-500 group-hover:text-cyan-400 transition-colors">
                  点击查看详情
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 详情模态框 */}
      <DetailModal result={selectedResult} onClose={handleCloseModal} />
    </div>
  );
}