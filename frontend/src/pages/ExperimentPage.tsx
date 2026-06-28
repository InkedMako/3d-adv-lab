import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store/useAppStore';
import { runExperiment, getExperimentProgress, getExperimentResult } from '@/api/experiment';
import { Play, Pause, Image, Layers, BarChart3, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export function ExperimentPage() {
  const navigate = useNavigate();
  const {
    currentSample,
    attackMethod,
    attackParams,
    defenseMethod,
    defenseParams,
    experimentRunning,
    experimentProgress,
    experimentResult,
    setExperimentRunning,
    setExperimentProgress,
    setExperimentResult,
    addToHistory,
  } = useAppStore();

  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');

  const handleRun = async () => {
    setStatus('running');
    setExperimentRunning(true);
    setExperimentProgress(0);
    setExperimentResult(null);

    try {
      // 运行实验
      const result = await runExperiment({
        sample_id: currentSample,
        attack_method: attackMethod,
        attack_params: attackParams,
        defense_method: defenseMethod,
        defense_params: defenseParams,
      });
      setExperimentId(result.experiment_id);

      // 监控进度
      const pollProgress = async () => {
        try {
          const progress = await getExperimentProgress(result.experiment_id);
          setExperimentProgress(progress.progress);
          setStatus(progress.status as 'running' | 'completed' | 'failed');
          
          if (progress.status === 'running') {
            setTimeout(pollProgress, 1000);
          } else if (progress.status === 'completed') {
            // 获取结果
            const expResult = await getExperimentResult(result.experiment_id);
            setExperimentResult(expResult);
            addToHistory(expResult);
            setExperimentRunning(false);
          } else {
            setExperimentRunning(false);
          }
        } catch (error) {
          console.error('进度查询错误:', error);
          setTimeout(pollProgress, 2000);
        }
      };
      pollProgress();
    } catch (error) {
      console.error(error);
      setStatus('failed');
      setExperimentRunning(false);
    }
  };

  // 准备置信度图表数据
  const chartData = experimentResult?.predictions ? [
    { name: '原始', value: experimentResult.predictions.original?.confidence || 0, color: '#00d9ff' },
    { name: '攻击后', value: experimentResult.predictions.attacked?.confidence || 0, color: '#ff4444' },
    { name: '防御后', value: experimentResult.predictions.defended?.confidence || 0, color: '#44ff44' },
  ] : [];

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
            <button onClick={() => navigate('/experiment')} className="text-cyan-400 hover:text-cyan-300 transition-colors">实验运行</button>
            <button onClick={() => navigate('/history')} className="text-gray-400 hover:text-cyan-400 transition-colors">历史记录</button>
          </nav>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* 控制面板 */}
        <div className="bg-[#1a1a2e] rounded-xl p-6 border border-cyan-500/20 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold mb-2">实验状态</h2>
              <div className="flex items-center gap-4">
                <span className="text-gray-400">样本: <span className="text-cyan-400">{currentSample}</span></span>
                <span className="text-gray-400">攻击: <span className="text-cyan-400">{attackMethod}</span></span>
                <span className="text-gray-400">防御: <span className="text-cyan-400">{defenseMethod}</span></span>
              </div>
            </div>
            <button
              onClick={handleRun}
              disabled={experimentRunning}
              className="bg-cyan-500 hover:bg-cyan-400 disabled:bg-gray-600 text-black font-semibold rounded-xl px-6 py-3 flex items-center gap-2 transition-all"
            >
              {experimentRunning ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  运行中...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  开始实验
                </>
              )}
            </button>
          </div>

          {/* 进度条 */}
          {status !== 'idle' && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">进度</span>
                <span className="text-sm text-cyan-400">{experimentProgress}%</span>
              </div>
              <div className="h-2 bg-[#0a0a1a] rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 transition-all duration-500"
                  style={{ width: `${experimentProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* 结果展示 */}
        {experimentResult && (
          <div className="space-y-8">
            {/* 识别结果 */}
            <div className="bg-[#1a1a2e] rounded-xl p-6 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-6">
                <BarChart3 className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-semibold">识别结果</h2>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-[#0a0a1a] rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-cyan-400">{experimentResult.predictions.original?.class || '-'}</div>
                  <div className="text-lg">{((experimentResult.predictions.original?.confidence || 0) * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400 mt-1">原始预测</div>
                </div>
                <div className="bg-[#0a0a1a] rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-400">{experimentResult.predictions.attacked?.class || '-'}</div>
                  <div className="text-lg">{((experimentResult.predictions.attacked?.confidence || 0) * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400 mt-1">攻击后预测</div>
                </div>
                <div className="bg-[#0a0a1a] rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-400">{experimentResult.predictions.defended?.class || '-'}</div>
                  <div className="text-lg">{((experimentResult.predictions.defended?.confidence || 0) * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-400 mt-1">防御后预测</div>
                </div>
              </div>

              {/* 置信度图表 */}
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" />
                  <XAxis dataKey="name" stroke="#888" />
                  <YAxis domain={[0, 1]} stroke="#888" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #00d9ff' }}
                    formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* 成功状态 */}
              <div className="flex items-center justify-center gap-8 mt-6">
                <div className="flex items-center gap-2">
                  {experimentResult.attack_success ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-400" />
                  )}
                  <span>攻击: {experimentResult.attack_success ? '成功' : '失败'}</span>
                </div>
                <div className="flex items-center gap-2">
                  {experimentResult.defense_success ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-400" />
                  )}
                  <span>防御: {experimentResult.defense_success ? '成功' : '失败'}</span>
                </div>
              </div>
            </div>

            {/* 图像对比 */}
            <div className="bg-[#1a1a2e] rounded-xl p-6 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-6">
                <Image className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-semibold">图像对比</h2>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-gray-400 mb-2 text-center">原始图像</div>
                  <img
                    src={`/files/${experimentResult.images.original}`}
                    alt="原始图像"
                    className="w-full rounded-lg border border-cyan-500/20"
                  />
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-2 text-center">攻击后图像</div>
                  <img
                    src={`/files/${experimentResult.images.attacked}`}
                    alt="攻击后图像"
                    className="w-full rounded-lg border border-cyan-500/20"
                  />
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-2 text-center">防御后图像</div>
                  <img
                    src={`/files/${experimentResult.images.defended}`}
                    alt="防御后图像"
                    className="w-full rounded-lg border border-cyan-500/20"
                  />
                </div>
              </div>
            </div>

            {/* BEV对比 */}
            <div className="bg-[#1a1a2e] rounded-xl p-6 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-6">
                <Layers className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-semibold">点云BEV对比</h2>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-gray-400 mb-2 text-center">原始BEV</div>
                  <img
                    src={`/files/${experimentResult.bevs.original}`}
                    alt="原始BEV"
                    className="w-full rounded-lg border border-cyan-500/20"
                  />
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-2 text-center">攻击后BEV</div>
                  <img
                    src={`/files/${experimentResult.bevs.attacked}`}
                    alt="攻击后BEV"
                    className="w-full rounded-lg border border-cyan-500/20"
                  />
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-2 text-center">防御后BEV</div>
                  <img
                    src={`/files/${experimentResult.bevs.defended}`}
                    alt="防御后BEV"
                    className="w-full rounded-lg border border-cyan-500/20"
                  />
                </div>
              </div>
            </div>

            {/* 综合对比图 */}
            <div className="bg-[#1a1a2e] rounded-xl p-6 border border-cyan-500/20">
              <h2 className="text-lg font-semibold mb-6">综合对比图</h2>
              <img
                src={`/files/${experimentResult.images.comparison}`}
                alt="综合对比图"
                className="w-full rounded-lg border border-cyan-500/20"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}