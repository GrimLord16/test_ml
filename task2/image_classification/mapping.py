import json
from torchvision import datasets
ds = datasets.ImageFolder(root='./../mammals45/mammals')
idx_to_class = {i: name for i, name in enumerate(ds.classes)}
with open('./animal_classifie/class_mapping.json', 'w') as f:
    json.dump(idx_to_class, f, indent=2)
print('Saved', len(idx_to_class), 'classes')