import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store/useAppStore';
import { getSamples, getSamplePreview } from '@/api/experiment';
import { Sliders, Image, Layers, Play, Settings } from 'lucide-react';

export function ConfigPage() {
  const navigate = useNavigate();
  const {
    currentSample,
    samplePreview,
    attackMethod,
    attackParams,
    defenseMethod,
    defenseParams,
    setCurrentSample,
    setSamplePreview,
    setAttackMethod,
    setAttackParams,
    setDefenseMethod,
    setDefenseParams,
  } = useAppStore();

  const [samples, setSamples] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  // 加载样本列表
  useEffect(() => {
    getSamples().then(setSamples).catch(console.error);
  }, []);

  // 加载样本预览
  useEffect(() => {
    if (currentSample) {
      setLoading(true);
      getSamplePreview(currentSample)
        .then(setSamplePreview)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [currentSample]);

  const handleRunExperiment = () => {
    // 跳转到实验运行页面
    navigate('/experiment');
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
            <button onClick={() => navigate('/')} className="text-cyan-400 hover:text-cyan-300 transition-colors">实验配置</button>
            <button onClick={() => navigate('/experiment')} className="text-gray-400 hover:text-cyan-400 transition-colors">实验运行</button>
            <button onClick={() => navigate('/history')} className="text-gray-400 hover:text-cyan-400 transition-colors">历史记录</button>
          </nav>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8 flex gap-8">
        {/* 左侧配置面板 */}
        <div className="w-[320px] space-y-6">
          {/* 样本选择 */}
          <div className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-semibold">样本选择</h2>
            </div>
            <select
              value={currentSample}
              onChange={(e) => setCurrentSample(e.target.value)}
              className="w-full bg-[#0a0a1a] border border-cyan-500/30 rounded-lg px-4 py-3 text-white focus:border-cyan-400 focus:outline-none transition-colors"
            >
              {samples.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* 攻击配置 */}
          <div className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Sliders className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-semibold">攻击配置</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 mb-2 block">攻击方法</label>
                <select
                  value={attackMethod}
                  onChange={(e) => setAttackMethod(e.target.value as 'FGSM' | 'PGD')}
                  className="w-full bg-[#0a0a1a] border border-cyan-500/30 rounded-lg px-4 py-2 text-white focus:border-cyan-400 focus:outline-none"
                >
                  <option value="FGSM">FGSM</option>
                  <option value="PGD">PGD</option>
                </select>
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-2 block">扰动大小 (ε): {attackParams.epsilon}</label>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  value={attackParams.epsilon}
                  onChange={(e) => {
                    const newValue = parseFloat(e.target.value);
                    if (newValue !== attackParams.epsilon) {
                      setAttackParams({ ...attackParams, epsilon: newValue });
                    }
                  }}
                  className="w-full accent-cyan-400"
                />
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-2 block">扰动比例: {attackParams.perturb_ratio}</label>
                <input
                  type="range"
                  min="0.05"
                  max="0.5"
                  step="0.01"
                  value={attackParams.perturb_ratio}
                  onChange={(e) => {
                    const newValue = parseFloat(e.target.value);
                    if (newValue !== attackParams.perturb_ratio) {
                      setAttackParams({ ...attackParams, perturb_ratio: newValue });
                    }
                  }}
                  className="w-full accent-cyan-400"
                />
              </div>

              {attackMethod === 'PGD' && (
                <>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">迭代次数: {attackParams.steps}</label>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      step="1"
                      value={attackParams.steps}
                      onChange={(e) => {
                        const newValue = parseInt(e.target.value);
                        if (newValue !== attackParams.steps) {
                          setAttackParams({ ...attackParams, steps: newValue });
                        }
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">步长: {attackParams.step_size}</label>
                    <input
                      type="range"
                      min="0.001"
                      max="0.1"
                      step="0.001"
                      value={attackParams.step_size}
                      onChange={(e) => {
                        const newValue = parseFloat(e.target.value);
                        if (newValue !== attackParams.step_size) {
                          setAttackParams({ ...attackParams, step_size: newValue });
                        }
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          {/* 防御配置 */}
          <div className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-semibold">防御配置</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 mb-2 block">防御方法</label>
                <select
                  value={defenseMethod}
                  onChange={(e) => setDefenseMethod(e.target.value as 'SOR' | 'Gaussian')}
                  className="w-full bg-[#0a0a1a] border border-cyan-500/30 rounded-lg px-4 py-2 text-white focus:border-cyan-400 focus:outline-none"
                >
                  <option value="SOR">SOR滤波</option>
                  <option value="Gaussian">高斯模糊</option>
                </select>
              </div>

              {defenseMethod === 'SOR' && (
                <>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">近邻数 (k): {defenseParams.k}</label>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="1"
                      value={defenseParams.k}
                      onChange={(e) => {
                        const newValue = parseInt(e.target.value);
                        if (newValue !== defenseParams.k) {
                          setDefenseParams({ ...defenseParams, k: newValue });
                        }
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">标准差阈值: {defenseParams.std_ratio}</label>
                    <input
                      type="range"
                      min="0.5"
                      max="3"
                      step="0.1"
                      value={defenseParams.std_ratio}
                      onChange={(e) => {
                        const newValue = parseFloat(e.target.value);
                        if (newValue !== defenseParams.std_ratio) {
                          setDefenseParams({ ...defenseParams, std_ratio: newValue });
                        }
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                </>
              )}

              {defenseMethod === 'Gaussian' && (
                <>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">核大小: {defenseParams.kernel_size}</label>
                    <input
                      type="range"
                      min="3"
                      max="15"
                      step="2"
                      value={defenseParams.kernel_size}
                      onChange={(e) => {
                        const newValue = parseInt(e.target.value);
                        if (newValue !== defenseParams.kernel_size) {
                          setDefenseParams({ ...defenseParams, kernel_size: newValue });
                        }
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Sigma: {defenseParams.sigma}</label>
                    <input
                      type="range"
                      min="0.5"
                      max="5"
                      step="0.1"
                      value={defenseParams.sigma}
                      onChange={(e) => {
                        const newValue = parseFloat(e.target.value);
                        if (newValue !== defenseParams.sigma) {
                          setDefenseParams({ ...defenseParams, sigma: newValue });
                        }
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          {/* 运行按钮 */}
          <button
            onClick={handleRunExperiment}
            className="w-full bg-cyan-500 hover:bg-cyan-400 text-black font-semibold rounded-xl px-6 py-4 flex items-center justify-center gap-2 transition-all shadow-lg shadow-cyan-500/20 hover:shadow-cyan-400/30"
          >
            <Play className="w-5 h-5" />
            运行实验
          </button>
        </div>

        {/* 右侧预览面板 */}
        <div className="flex-1 space-y-6">
          {/* 样本信息 */}
          {samplePreview && (
            <div className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20">
              <h2 className="text-lg font-semibold mb-4">样本信息</h2>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">样本序号:</span>
                  <span className="ml-2 text-cyan-400">{samplePreview.sample_id}</span>
                </div>
                <div>
                  <span className="text-gray-400">点数:</span>
                  <span className="ml-2 text-cyan-400">{samplePreview.point_count}</span>
                </div>
                <div>
                  <span className="text-gray-400">标注:</span>
                  <span className="ml-2 text-cyan-400">{samplePreview.labels?.join(', ') || '-'}</span>
                </div>
              </div>
            </div>
          )}

          {/* 图像和BEV预览（横向并排） */}
          <div className="grid grid-cols-2 gap-5">
            {/* 图像预览 */}
            <div className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-4">
                <Image className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-semibold">原始图像</h2>
              </div>
              <div className="aspect-[4/3] bg-[#0a0a1a] rounded-lg overflow-hidden flex items-center justify-center">
                {loading ? (
                  <div className="text-gray-400 animate-pulse">加载中...</div>
                ) : samplePreview?.image_path ? (
                  <img
                    src={`/files/${samplePreview.image_path}`}
                    alt="原始图像"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-gray-400">无图像</div>
                )}
              </div>
            </div>

            {/* BEV预览 */}
            <div className="bg-[#1a1a2e] rounded-xl p-5 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-semibold">点云BEV视图</h2>
              </div>
              <div className="aspect-square bg-[#0a0a1a] rounded-lg overflow-hidden flex items-center justify-center">
                {loading ? (
                  <div className="text-gray-400 animate-pulse">加载中...</div>
                ) : samplePreview?.bev_path ? (
                  <img
                    src={`/files/${samplePreview.bev_path}`}
                    alt="BEV视图"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-gray-400">无BEV数据</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}