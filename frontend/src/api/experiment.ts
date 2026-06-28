const API_BASE = '/api';

export interface SamplePreview {
  sample_id: string;
  image_path: string;
  bev_path: string;
  labels: string[];
  point_count: number;
}

export interface ExperimentResult {
  id: string;
  sample_id: string;
  timestamp: string;
  attack_method: string;
  defense_method: string;
  attack_params: {
    epsilon: number;
    perturb_ratio: number;
    steps: number;
    step_size: number;
  };
  defense_params: {
    k: number;
    std_ratio: number;
    kernel_size: number;
    sigma: number;
  };
  predictions: {
    original: { class: string; confidence: number };
    attacked: { class: string; confidence: number };
    defended: { class: string; confidence: number };
  };
  attack_success: boolean;
  defense_success: boolean;
  images: {
    original: string;
    attacked: string;
    defended: string;
    comparison: string;
  };
  bevs: {
    original: string;
    attacked: string;
    defended: string;
  };
}

// 获取样本列表
export async function getSamples(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/samples`);
  const data = await response.json();
  return data.samples;
}

// 获取样本预览
export async function getSamplePreview(sampleId: string): Promise<SamplePreview> {
  const response = await fetch(`${API_BASE}/samples/${sampleId}/preview`);
  const data = await response.json();
  return data;
}

// 运行实验
export async function runExperiment(params: {
  sample_id: string;
  attack_method: 'FGSM' | 'PGD';
  attack_params: {
    epsilon: number;
    perturb_ratio: number;
    steps: number;
    step_size: number;
  };
  defense_method: 'SOR' | 'Gaussian';
  defense_params: {
    k: number;
    std_ratio: number;
    kernel_size: number;
    sigma: number;
  };
}): Promise<{ experiment_id: string; script_path: string; output_dir: string }> {
  const response = await fetch(`${API_BASE}/experiment/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = await response.json();
  return data;
}

// 获取实验进度
export async function getExperimentProgress(experimentId: string): Promise<{
  status: 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
}> {
  const response = await fetch(`${API_BASE}/experiment/${experimentId}/progress`);
  const text = await response.text();
  if (!text) {
    return { status: 'running', progress: 0, message: '等待启动' };
  }
  try {
    return JSON.parse(text);
  } catch {
    return { status: 'running', progress: 0, message: '等待启动' };
  }
}

// 获取实验结果
export async function getExperimentResult(experimentId: string): Promise<ExperimentResult> {
  const response = await fetch(`${API_BASE}/experiment/${experimentId}/result`);
  const data = await response.json();
  return data;
}

// 获取历史实验列表
export async function getHistory(): Promise<ExperimentResult[]> {
  const response = await fetch(`${API_BASE}/history`);
  const data = await response.json();
  return data.experiments;
}

// 获取历史实验详情
export async function getHistoryDetail(experimentId: string): Promise<ExperimentResult> {
  const response = await fetch(`${API_BASE}/history/${experimentId}`);
  const data = await response.json();
  return data;
}