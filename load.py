import os
import json
def load_annotation_file():
    folder = '/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation'
    subfolders = os.listdir(folder)
    subfolders.sort()
    # filter out files starts with 4 and has 5 digits
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    annotation_json = 'link_anno_gapartnet.json'
    annotation = {}
    for subfolder in subfolders:
        annotation_path = os.path.join(folder, subfolder, annotation_json)
        with open(annotation_path, 'r') as f:
            annotation[subfolder] = json.load(f)
    return annotation

def load_usd_paths():
    folder = '/home/nikepupu/Desktop/Orbit/NewUSD'
    subfolders = os.listdir(folder)
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    usds = [os.path.join(folder, f, 'mobility_relabel_gapartnet.usd') for f in subfolders]
    return usds

# usds = load_usd_paths()
# print(usds)
annotation = load_annotation_file()
tmp = annotation['40147']

print(tmp)
print()
# for item in tmp:
#     print(item['category'])
#     print(item['link_name'])
#     print('\n')


def load_joint_file():
    folder = '/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation'
    subfolders = os.listdir(folder)
    subfolders.sort()
    # filter out files starts with 4 and has 5 digits
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    annotation_json = 'mobility_v2.json'
    annotation = {}
    for subfolder in subfolders:
        annotation_path = os.path.join(folder, subfolder, annotation_json)
        with open(annotation_path, 'r') as f:
            annotation[int(subfolder)] = json.load(f)
    return annotation

anno = load_joint_file()
print(anno[40147])