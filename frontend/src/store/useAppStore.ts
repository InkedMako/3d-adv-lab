import { create } from 'zustand';

export interface AttackParams {
  epsilon: number;
  perturb_ratio: number;
  steps: number;
  step_size: number;
}

export interface DefenseParams {
  k: number;
  std_ratio: number;
  kernel_size: number;
  sigma: number;
}

export interface ExperimentResult {
  id: string;
  sample_id: string;
  timestamp: string;
  attack_method: string;
  defense_method: string;
  attack_params: AttackParams;
  defense_params: DefenseParams;
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

export interface SamplePreview {
  sample_id: string;
  image_path: string;
  bev_path: string;
  labels: string[];
  point_count: number;
}

interface AppState {
  // 当前选择的样本
  currentSample: string;
  samplePreview: SamplePreview | null;
  
  // 攻击配置
  attackMethod: 'FGSM' | 'PGD';
  attackParams: AttackParams;
  
  // 防御配置
  defenseMethod: 'SOR' | 'Gaussian';
  defenseParams: DefenseParams;
  
  // 实验状态
  experimentRunning: boolean;
  experimentProgress: number;
  experimentResult: ExperimentResult | null;
  
  // 历史实验
  history: ExperimentResult[];
  
  // 操作方法
  setCurrentSample: (sample: string) => void;
  setSamplePreview: (preview: SamplePreview | null) => void;
  setAttackMethod: (method: 'FGSM' | 'PGD') => void;
  setAttackParams: (params: AttackParams) => void;
  setDefenseMethod: (method: 'SOR' | 'Gaussian') => void;
  setDefenseParams: (params: DefenseParams) => void;
  setExperimentRunning: (running: boolean) => void;
  setExperimentProgress: (progress: number) => void;
  setExperimentResult: (result: ExperimentResult | null) => void;
  setHistory: (history: ExperimentResult[]) => void;
  addToHistory: (result: ExperimentResult) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // 初始状态
  currentSample: '000000',
  samplePreview: null,
  
  attackMethod: 'FGSM',
  attackParams: {
    epsilon: 0.3,
    perturb_ratio: 0.22,
    steps: 10,
    step_size: 0.01,
  },
  
  defenseMethod: 'SOR',
  defenseParams: {
    k: 20,
    std_ratio: 1.0,
    kernel_size: 5,
    sigma: 1.0,
  },
  
  experimentRunning: false,
  experimentProgress: 0,
  experimentResult: null,
  
  history: [],
  
  // 操作方法
  setCurrentSample: (sample) => set({ currentSample: sample }),
  setSamplePreview: (preview) => set({ samplePreview: preview }),
  setAttackMethod: (method) => set({ attackMethod: method }),
  setAttackParams: (params) => set({ attackParams: params }),
  setDefenseMethod: (method) => set({ defenseMethod: method }),
  setDefenseParams: (params) => set({ defenseParams: params }),
  setExperimentRunning: (running) => set({ experimentRunning: running }),
  setExperimentProgress: (progress) => set({ experimentProgress: progress }),
  setExperimentResult: (result) => set({ experimentResult: result }),
  setHistory: (history) => set({ history }),
  addToHistory: (result) => set((state) => ({ history: [result, ...state.history] })),
}));