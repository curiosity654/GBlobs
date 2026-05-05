import json
from pathlib import Path

SCHEMA_VERSION = 'unieval.v1'
PAYLOAD_RELPATH = 'predictions.feather'
KITTI_REQUIRED_COLUMNS = (
    'frame_id',
    'center_x',
    'center_y',
    'center_z',
    'length',
    'width',
    'height',
    'yaw',
    'score',
    'category',
)

KITTI_UNIFIED_3CLASS_MAP = {
    'car': 'VEHICLE',
    'vehicle': 'VEHICLE',
    'van': 'VEHICLE',
    'truck': 'VEHICLE',
    'bus': 'VEHICLE',
    'trailer': 'VEHICLE',
    'cyclist': 'BICYCLE',
    'bicycle': 'BICYCLE',
    'motorcycle': 'BICYCLE',
    'pedestrian': 'PEDESTRIAN',
    'person_sitting': 'PEDESTRIAN',
}


def _build_manifest(export_cfg):
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'dataset': export_cfg.DATASET,
        'task': export_cfg.TASK,
        'split': export_cfg.SPLIT,
        'source_codebase': export_cfg.SOURCE_CODEBASE,
        'label_space': export_cfg.LABEL_SPACE,
        'coord_system': export_cfg.COORD_SYSTEM,
        'box_origin': export_cfg.BOX_ORIGIN,
        'payload_relpath': PAYLOAD_RELPATH,
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest):
    required = {
        'schema_version',
        'dataset',
        'task',
        'split',
        'source_codebase',
        'label_space',
        'coord_system',
        'box_origin',
        'payload_relpath',
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f'manifest.json is missing required fields: {missing}')

    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'Unsupported schema_version: {manifest["schema_version"]!r}')
    if manifest['dataset'] != 'kitti':
        raise ValueError(f'Unsupported UniEval export dataset: {manifest["dataset"]!r}')
    if manifest['payload_relpath'] != PAYLOAD_RELPATH:
        raise ValueError(f'payload_relpath must be {PAYLOAD_RELPATH!r}')
    if manifest['box_origin'] not in {'bottom_center', 'gravity_center'}:
        raise ValueError(f'Unsupported box_origin: {manifest["box_origin"]!r}')


def _normalize_category(name):
    if name is None:
        return None
    return KITTI_UNIFIED_3CLASS_MAP.get(str(name).strip().lower())


def _empty_payload_dict():
    return {column: [] for column in KITTI_REQUIRED_COLUMNS}


def _build_payload_dict(det_annos, export_cfg):
    if export_cfg.LABEL_SPACE != 'unified:3class':
        raise ValueError(
            f'Only LABEL_SPACE="unified:3class" is currently supported, got {export_cfg.LABEL_SPACE!r}'
        )

    payload = _empty_payload_dict()
    valid_frames = set()
    skipped_boxes = 0

    for anno in det_annos:
        frame_id = anno.get('frame_id')
        if frame_id is None:
            continue

        frame_id = str(frame_id)
        valid_frames.add(frame_id)
        names = anno.get('name', [])
        scores = anno.get('score', [])
        boxes_lidar = anno.get('boxes_lidar', [])

        for idx, raw_name in enumerate(names):
            category = _normalize_category(raw_name)
            if category is None:
                skipped_boxes += 1
                continue

            box = boxes_lidar[idx]
            payload['frame_id'].append(frame_id)
            payload['center_x'].append(float(box[0]))
            payload['center_y'].append(float(box[1]))
            payload['center_z'].append(float(box[2]))
            payload['length'].append(float(box[3]))
            payload['width'].append(float(box[4]))
            payload['height'].append(float(box[5]))
            payload['yaw'].append(float(box[6]))
            payload['score'].append(float(scores[idx]))
            payload['category'].append(category)

    return payload, valid_frames, skipped_boxes


def _validate_payload_columns(payload_dict):
    payload_columns = tuple(payload_dict.keys())
    if payload_columns != KITTI_REQUIRED_COLUMNS:
        raise ValueError(
            f'Unexpected payload columns: {payload_columns!r}, expected {KITTI_REQUIRED_COLUMNS!r}'
        )


def _write_manifest(output_dir, manifest):
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest_path


def _write_payload(output_dir, payload_dict):
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError(
            'UniEval feather export requires pyarrow to be installed.'
        ) from exc

    payload_path = output_dir / PAYLOAD_RELPATH
    table = pa.table(payload_dict)
    feather.write_feather(table, payload_path)
    return payload_path


def export_kitti_prediction_package(det_annos, export_cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(export_cfg)
    payload_dict, valid_frames, skipped_boxes = _build_payload_dict(det_annos, export_cfg)
    _validate_payload_columns(payload_dict)

    manifest_path = _write_manifest(output_dir, manifest)
    payload_path = _write_payload(output_dir, payload_dict)

    return {
        'manifest_path': str(manifest_path),
        'payload_path': str(payload_path),
        'package_dir': str(output_dir),
        'num_boxes': len(payload_dict['frame_id']),
        'num_frames': len(valid_frames),
        'skipped_boxes': skipped_boxes,
    }
