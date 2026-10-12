export type StructuredValue =
  | null
  | boolean
  | number
  | bigint
  | string
  | StructuredValue[]
  | { [key: string]: StructuredValue };

export interface StructuredQuestion {
  type: string;
  instructions: StructuredValue;
  criteria?: StructuredValue;
}

export interface StructuredRequest {
  state: StructuredValue;
  questions: { [id: string]: StructuredQuestion };
  temperature?: number;
}

export interface FreeFormRankRequest {
  state: StructuredValue;
  instructions: StructuredValue;
  candidates: { [key: string]: StructuredValue };
  temperature?: number;
}

export interface ModelAnswer {
  id: string;
  type: string;
  noul: number | null;
  choice: string | null;
  score: number | null;
  confidence: number | null;
  probabilities: { [key: string]: number };
  legend: { [key: string]: string };
}

export interface ModelResult {
  model: string;
  answers: ModelAnswer[];
}

export interface RankedItem {
  rank: number;
  key: string;
  value: StructuredValue;
  probability: number;
}

export interface RankingResult {
  model: string;
  items: RankedItem[];
}

export interface CacheStats {
  hits: bigint;
  misses: bigint;
  evictions: bigint;
  entries: number | bigint;
  bytes: number | bigint;
  entryCapacity: number | bigint;
  byteCapacity: number | bigint;
}

export interface PrefixReuseStats {
  prefixRuns: bigint;
  branchRuns: bigint;
  fallbackRuns: bigint;
}

export class DirectoryTokenizer {
  constructor(packagePath: string);
  encode(text: string): Int32Array;
  readonly padTokenId: number;
  close(): void;
}

export class RankingSession {
  constructor(packagePath: string, providers?: string[]);
  run(request: StructuredRequest): ModelResult;
  rank(request: FreeFormRankRequest): RankingResult;
  setCacheCapacity(entries: number | bigint, bytes: number | bigint): void;
  getCacheStats(): CacheStats;
  readonly cacheStats: CacheStats;
  clearCache(): void;
  invalidateCache(): void;
  close(): void;
}

export class DecisionSession {
  constructor(packagePath: string, providers?: string[]);
  run(request: StructuredRequest): ModelResult;
  decide(request: StructuredRequest): ModelResult;
  setCacheCapacity(entries: number | bigint, bytes: number | bigint): void;
  getCacheStats(): CacheStats;
  readonly cacheStats: CacheStats;
  clearCache(): void;
  invalidateCache(): void;
  setPrefixCacheCapacity(entries: number | bigint, bytes: number | bigint): void;
  prefixReuseEnabled: boolean;
  readonly prefixReuseStatus: string;
  readonly prefixCacheStats: CacheStats;
  readonly prefixReuseStats: PrefixReuseStats;
  close(): void;
}
