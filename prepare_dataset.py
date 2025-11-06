
import json
import csv

def prepare_dataset(json_path, output_csv_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    drum_categories = {
        'clefUnpitchedPercussion', 'noteheadBlackOnLine', 'noteheadBlackInSpace',
        'noteheadHalfOnLine', 'noteheadHalfInSpace', 'noteheadWholeOnLine',
        'noteheadWholeInSpace', 'noteheadDoubleWholeOnLine', 'noteheadDoubleWholeInSpace',
        'augmentationDot', 'stem', 'tremolo1', 'tremolo2', 'tremolo3', 'tremolo4',
        'tremolo5', 'flag8thUp', 'flag16thUp', 'flag32ndUp', 'flag64thUp', 'flag128thUp',
        'flag8thDown', 'flag16thDown', 'flag32ndDown', 'flag64thDown', 'flag128thDown',
        'articAccentAbove', 'articAccentBelow', 'articStaccatoAbove', 'articStaccatoBelow',
        'articTenutoAbove', 'articTenutoBelow', 'articStaccatissimoAbove',
        'articStaccatissimoBelow', 'articMarcatoAbove', 'articMarcatoBelow',
        'fermataAbove', 'fermataBelow', 'restWhole', 'restHalf', 'restQuarter',
        'rest8th', 'rest16th', 'rest32nd', 'rest64th', 'rest128th', 'dynamicP',
        'dynamicM', 'dynamicF', 'dynamicS', 'dynamicZ', 'dynamicR',
        'graceNoteAcciaccaturaStemUp', 'graceNoteAppoggiaturaStemUp',
        'graceNoteAcciaccaturaStemDown', 'graceNoteAppoggiaturaStemDown', 'beam',
        'tie', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin'
    }

    prepared_data = []
    for image in images:
        for ann_id in image['ann_ids']:
            ann = annotations.get(str(ann_id))
            if ann:
                for cat_id in ann['cat_id']:
                    if cat_id:
                        category_name = categories[cat_id]['name']
                    if category_name in drum_categories:
                        prepared_data.append({
                            'filename': image['filename'],
                            'bbox': ann['a_bbox'],
                            'category': category_name
                        })

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'bbox', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prepared_data)

    return prepared_data

if __name__ == '__main__':
    prepared_data = prepare_dataset('data/ds2_dense/deepscores_train.json', 'data/prepared_data.csv')
    print(f"Prepared {len(prepared_data)} data points and saved to data/prepared_data.csv")
