/**
 * API Client for VisionDrive AI Backend.
 */
import {
  DemoScenario,
  FeedbackItem,
  PredictResponse,
  RoadType,
  SpeedRulesResponse,
  VehicleType,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export async function predictFrame(
  imageBase64: string,
  speed?: number | null,
  vehicleType: VehicleType = 'car',
  cautionMargin: number = 5.0,
  applySmoothing: boolean = true
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      current_speed: speed ?? null,
      vehicle_type: vehicleType,
      caution_margin: cautionMargin,
      apply_smoothing: applySmoothing,
    }),
  });
  if (!res.ok) {
    throw new Error(`Predict API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function predictImage(
  file: File,
  speed?: number | null,
  vehicleType: VehicleType = 'car',
  cautionMargin: number = 5.0
): Promise<PredictResponse> {
  const form = new FormData();
  form.append('file', file);
  if (speed !== undefined && speed !== null) {
    form.append('speed', String(speed));
  }
  form.append('vehicle_type', vehicleType);
  form.append('caution_margin', String(cautionMargin));

  const res = await fetch(`${API_BASE}/predict/image`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    throw new Error(`Image Predict API error: ${res.status}`);
  }
  return res.json();
}

export async function getDemoScenarios(): Promise<DemoScenario[]> {
  const res = await fetch(`${API_BASE}/predict/demo-scenarios`);
  if (!res.ok) {
    throw new Error(`Demo Scenarios API error: ${res.status}`);
  }
  return res.json();
}

export async function getSpeedRules(): Promise<SpeedRulesResponse> {
  const res = await fetch(`${API_BASE}/rules`);
  if (!res.ok) {
    throw new Error(`Rules API error: ${res.status}`);
  }
  return res.json();
}

export async function updateSpeedRule(
  roadType: RoadType,
  legalMaxSpeed: number,
  cautionMargin: number = 5.0
): Promise<any> {
  const res = await fetch(`${API_BASE}/rules/${roadType}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      legal_max_speed: legalMaxSpeed,
      default_caution_margin: cautionMargin,
    }),
  });
  if (!res.ok) {
    throw new Error(`Update Rule error: ${res.status}`);
  }
  return res.json();
}

export async function submitFeedback(item: FeedbackItem): Promise<FeedbackItem> {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(item),
  });
  if (!res.ok) {
    throw new Error(`Feedback API error: ${res.status}`);
  }
  return res.json();
}

export async function getFeedbackList(): Promise<{ total: number; items: FeedbackItem[] }> {
  const res = await fetch(`${API_BASE}/feedback`);
  if (!res.ok) {
    throw new Error(`Feedback List error: ${res.status}`);
  }
  return res.json();
}

export async function resetStreamSmoother(): Promise<void> {
  await fetch(`${API_BASE}/predict/reset-stream`, { method: 'POST' });
}
