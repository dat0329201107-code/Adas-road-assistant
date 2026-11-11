"""Default labels and configuration for the prototype."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field

ROAD_PROMPTS = {
    "a photo of a busy urban city street with buildings, traffic and sidewalks": "Đường đô thị",
    "a photo of a residential neighborhood street with houses": "Đường khu dân cư",
    "a photo of a rural road in the countryside": "Đường nông thôn",
    "a photo of a highway or expressway with multiple lanes": "Đường cao tốc / đường cấp cao",
    "a photo of a road intersection with cross traffic": "Giao lộ / nút giao",
    "a photo of a road passing through a tunnel": "Đường hầm",
    "a photo of an industrial area road with warehouses": "Đường khu công nghiệp",
    "a photo of a parking lot or parking area": "Bãi đỗ xe",
    "a photo of a narrow alley or small local road": "Ngõ / đường nhỏ",
}

ROAD_LABELS = list(ROAD_PROMPTS.values())


@dataclass
class AppConfig:
    """Application configuration loaded from config.yaml or defaults."""

    # Camera
    camera_source: str | int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # Road classification
    road_ai_enabled: bool = True
    road_model_name: str = "openai/clip-vit-base-patch32"
    road_every: int = 20

    # Sign detection
    sign_enabled: bool = True
    sign_every: int = 5
    enable_ocr: bool = True
    tesseract_cmd: str | None = None
    use_yolo: bool = False
    yolo_model_path: str = "models/traffic_signs_yolo26n.pt"
    yolo_confidence: float = 0.4

    # Lane detection
    lane_enabled: bool = False
    lane_roi_top: float = 0.55

    # Object detection
    object_detection_enabled: bool = False
    object_model_path: str | None = None
    object_confidence: float = 0.35
    object_target_classes: list[str] = field(
        default_factory=lambda: ['car', 'truck', 'bus', 'motorcycle', 'person']
    )

    # Speed
    speed_source: str = "simulated"
    initial_demo_speed: float | None = None
    manual_limit: int | None = None
    overspeed_buffer: float = 0.0
    gps_port: str = "COM3"
    gps_baudrate: int = 9600
    obd_port: str | None = None

    # Alerts
    enable_sound: bool = True
    sound_repeat_seconds: float = 3.0
    confirm_seconds: float = 0.5
    sign_ttl_seconds: float = 45.0
    caution_margin: float = 5.0
    min_confidence: float = 0.60
    min_warmup_frames: int = 5

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "outputs/logs"
    log_frame_stats_every: int = 100

    # Execution controls
    max_frames: int | None = None
    headless: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'AppConfig':
        """Load config from YAML file, falling back to defaults for missing keys."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # Flatten nested yaml into flat dataclass fields
        kwargs: dict = {}

        cam = data.get('camera', {})
        if 'source' in cam:
            kwargs['camera_source'] = cam['source']
        if 'width' in cam:
            kwargs['camera_width'] = cam['width']
        if 'height' in cam:
            kwargs['camera_height'] = cam['height']

        road = data.get('road_classification', {})
        if 'enabled' in road:
            kwargs['road_ai_enabled'] = road['enabled']
        if 'model_name' in road:
            kwargs['road_model_name'] = road['model_name']
        if 'every_n_frames' in road:
            kwargs['road_every'] = road['every_n_frames']

        sign = data.get('sign_detection', {})
        if 'enabled' in sign:
            kwargs['sign_enabled'] = sign['enabled']
        if 'every_n_frames' in sign:
            kwargs['sign_every'] = sign['every_n_frames']
        if 'enable_ocr' in sign:
            kwargs['enable_ocr'] = sign['enable_ocr']
        if 'tesseract_cmd' in sign:
            kwargs['tesseract_cmd'] = sign['tesseract_cmd']
        if 'use_yolo' in sign:
            kwargs['use_yolo'] = sign['use_yolo']
        if 'yolo_model_path' in sign:
            kwargs['yolo_model_path'] = sign['yolo_model_path']
        if 'yolo_confidence' in sign:
            kwargs['yolo_confidence'] = sign['yolo_confidence']

        lane = data.get('lane_detection', {})
        if 'enabled' in lane:
            kwargs['lane_enabled'] = lane['enabled']
        if 'roi_top_ratio' in lane:
            kwargs['lane_roi_top'] = lane['roi_top_ratio']

        obj = data.get('object_detection', {})
        if 'enabled' in obj:
            kwargs['object_detection_enabled'] = obj['enabled']
        if 'model_path' in obj:
            kwargs['object_model_path'] = obj['model_path']
        if 'confidence' in obj:
            kwargs['object_confidence'] = obj['confidence']
        if 'target_classes' in obj:
            kwargs['object_target_classes'] = obj['target_classes']

        spd = data.get('speed', {})
        if 'source' in spd:
            kwargs['speed_source'] = spd['source']
        if 'initial_demo_speed' in spd:
            kwargs['initial_demo_speed'] = spd['initial_demo_speed']
        if 'manual_limit' in spd:
            kwargs['manual_limit'] = spd['manual_limit']
        if 'overspeed_buffer' in spd:
            kwargs['overspeed_buffer'] = spd['overspeed_buffer']
        if 'gps_port' in spd:
            kwargs['gps_port'] = spd['gps_port']
        if 'gps_baudrate' in spd:
            kwargs['gps_baudrate'] = spd['gps_baudrate']
        if 'obd_port' in spd:
            kwargs['obd_port'] = spd['obd_port']

        alt = data.get('alerts', {})
        if 'enable_sound' in alt:
            kwargs['enable_sound'] = alt['enable_sound']
        if 'sound_repeat_seconds' in alt:
            kwargs['sound_repeat_seconds'] = alt['sound_repeat_seconds']
        if 'confirm_seconds' in alt:
            kwargs['confirm_seconds'] = alt['confirm_seconds']
        if 'sign_ttl_seconds' in alt:
            kwargs['sign_ttl_seconds'] = alt['sign_ttl_seconds']
        if 'caution_margin' in alt:
            kwargs['caution_margin'] = alt['caution_margin']
        if 'min_confidence' in alt:
            kwargs['min_confidence'] = alt['min_confidence']
        if 'min_warmup_frames' in alt:
            kwargs['min_warmup_frames'] = alt['min_warmup_frames']

        log = data.get('logging', {})
        if 'level' in log:
            kwargs['log_level'] = log['level']
        if 'log_to_file' in log:
            kwargs['log_to_file'] = log['log_to_file']
        if 'log_dir' in log:
            kwargs['log_dir'] = log['log_dir']
        if 'log_frame_stats_every' in log:
            kwargs['log_frame_stats_every'] = log['log_frame_stats_every']

        return cls(**kwargs)

    @classmethod
    def from_args_and_yaml(
        cls, args, yaml_path: str | Path = 'config.yaml'
    ) -> 'AppConfig':
        """Load from YAML first, then override with CLI args (CLI wins)."""
        cfg = cls.from_yaml(yaml_path)

        # CLI overrides (only if explicitly provided by the user)
        if hasattr(args, 'source') and args.source != '0':
            cfg.camera_source = args.source
        if hasattr(args, 'width'):
            cfg.camera_width = args.width
        if hasattr(args, 'height'):
            cfg.camera_height = args.height
        if hasattr(args, 'no_road_ai') and args.no_road_ai:
            cfg.road_ai_enabled = False
        if hasattr(args, 'model_name') and args.model_name != 'openai/clip-vit-base-patch32':
            cfg.road_model_name = args.model_name
        if hasattr(args, 'road_every'):
            cfg.road_every = args.road_every
        if hasattr(args, 'sign_every'):
            cfg.sign_every = args.sign_every
        if hasattr(args, 'no_ocr') and args.no_ocr:
            cfg.enable_ocr = False
        if hasattr(args, 'tesseract_cmd') and args.tesseract_cmd:
            cfg.tesseract_cmd = args.tesseract_cmd
        if hasattr(args, 'manual_limit') and args.manual_limit is not None:
            cfg.manual_limit = args.manual_limit
        if hasattr(args, 'demo_speed') and args.demo_speed is not None:
            cfg.initial_demo_speed = args.demo_speed
        if hasattr(args, 'overspeed_buffer'):
            cfg.overspeed_buffer = args.overspeed_buffer
        if hasattr(args, 'limit_ttl'):
            cfg.sign_ttl_seconds = args.limit_ttl
        if hasattr(args, 'no_sound') and args.no_sound:
            cfg.enable_sound = False
        if hasattr(args, 'log_level'):
            cfg.log_level = args.log_level
        if hasattr(args, 'max_frames') and args.max_frames is not None:
            cfg.max_frames = args.max_frames
        if hasattr(args, 'headless') and args.headless:
            cfg.headless = True

        return cfg
