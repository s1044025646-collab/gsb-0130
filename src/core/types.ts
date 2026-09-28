export interface TaggedSentence {
  tokens: string[];
  tags: string[];
}

export interface CorpusInput {
  name: string;
  sentences: TaggedSentence[];
}

export type UnknownStrategy = 'uniform' | 'closed';

export interface ViterbiStep {
  position: number;
  token: string | null;
  chosen_tag: string;
  prev_tag: string | null;
  step_log_prob: number;
  cumulative_log_prob: number;
}

export interface ViterbiResult {
  tags: string[];
  score: number;
  steps: ViterbiStep[];
  unknownPositions: number[];
}

export interface TagMetric {
  tag: string;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ReportSummary {
  tokenTotal: number;
  tokenCorrect: number;
  sentenceTotal: number;
  sentenceCorrect: number;
  accuracy: number;
  logLikelihood: number;
}

export class AppError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'AppError';
  }
}
