"""CPU-only serialization of generated pseudo-label index snapshots."""
from collections import Counter
import csv
import io
import json


def index_contents(records):
    fields = ['image_file','nir_file','date','split','sequence','width','height',
              'detection_file','road_file','road_probability_file','annotation_file','preview_file',
              'label_source','review_status','detection_count','road_fraction','uncertain_fraction']
    manifest = io.StringIO(newline='')
    writer = csv.DictWriter(manifest,fieldnames=fields,extrasaction='ignore')
    writer.writeheader(); writer.writerows(records)
    queue = io.StringIO(newline='')
    writer = csv.writer(queue)
    writer.writerow(['annotation_file','preview_file','review_status','review_reasons','uncertain_fraction'])
    for record in sorted(records,key=lambda r:(-len(r.get('review_reasons',[])),-r.get('uncertain_fraction',0),r['image_file'])):
        writer.writerow([record['annotation_file'],record.get('preview_file',''),record.get('review_status','pending'),
                         ';'.join(record.get('review_reasons',[])),record.get('uncertain_fraction',0)])
    detections = [r.get('filtered_detections',r.get('detections',[])) for r in records]
    summary = {
        'frames':len(records), 'splits':dict(Counter(r['split'] for r in records)),
        'review_status':dict(Counter(r.get('review_status','pending') for r in records)),
        'detections':dict(Counter(b['class_name'] for boxes in detections for b in boxes)),
        'empty_detection_frames':sum(not boxes for boxes in detections),
        'segmentation':'binary Road, 0=non-road, 1=road, 255=uncertain/ignore',
        'probability_encoding':'uint16 PNG / 65535 = P(road)',
    }
    return {'manifest.csv':manifest.getvalue().encode(), 'review_queue.csv':queue.getvalue().encode(),
            'summary.json':(json.dumps(summary,indent=2,ensure_ascii=False)+'\n').encode()}, summary


def rebuild_index(root):
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(root.glob('*/*/*/annotations/*.json'))]
    contents, summary = index_contents(records)
    for name, data in contents.items():
        path = root/name
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_bytes(data); temporary.replace(path)
    return summary
