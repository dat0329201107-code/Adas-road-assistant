/**
 * TypeScript Type Definitions for VisionDrive AI Frontend.
 */

export type RoadType =
  | 'cao_toc'
  | 'duong_doi_do_thi'
  | 'pho_hai_chieu'
  | 'ngoai_khu_dan_cu'
  | 'hem_ngo';

export type AlertLevel = 'idle' | 'ok' | 'warn' | 'bad';

export type VehicleType = 'car' | 'motorcycle' | 'truck' | 'bus';

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  confidence: number;
}

export interface PredictResponse {
  road_type: RoadType;
  road_name_vi: string;
  road_class_id: number;
  confidence: number;
  confidence_percent: number;
  probabilities: Record<RoadType, number>;
  speed_limit: number | null;
  current_speed: number | null;
  diff_speed: number | null;
  alert_level: AlertLevel;
  alert_label: string;
  explanation: string;
  rule_description: string;
  legal_citation: string;
  boxes: BoundingBox[];
  latency_ms: number;
  smoothed: boolean;
}

export interface DemoScenario {
  chip: string;
  cls: number;
  road_type: RoadType;
  name_vi: string;
  limit: number;
  max_limit: number;
  main: number;
  probs: number[];
  rule: string;
  speeds: number[];
  boxes: Array<{ x: number; y: number; w: number; h: number; label: string }>;
}

export interface SpeedRuleItem {
  class_id: number;
  class_key: RoadType;
  name_vi: string;
  name_en: string;
  legal_max_speed: number;
  max_allowed_speed: number;
  min_allowed_speed: number | null;
  default_caution_margin: number;
  description: string;
  rule_detail: string;
  vehicle_limits: Record<VehicleType, number>;
}

export interface SpeedRulesResponse {
  legal_framework: string;
  version: string;
  road_classes: Record<RoadType, SpeedRuleItem>;
  alert_levels: Record<AlertLevel, { code: string; label: string; color_hex: string; description: string }>;
}

export interface FeedbackItem {
  id?: number;
  created_at?: string;
  predicted_class: string;
  correct_class: string;
  current_speed?: number | null;
  timestamp_str?: string | null;
  notes?: string | null;
  image_base64?: string | null;
}
