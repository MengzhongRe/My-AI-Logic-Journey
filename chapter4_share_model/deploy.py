from huggingface_hub import HfApi,create_repo,ModelCard,ModelCardData

#1配置
USER = 'YiMeng-SYSU'
REPO = 'vit-base-patch16-224-in21k-finetuned-cifar100'
REPO_ID = f'{USER}/{REPO}'
LOCAL_DIR = '/home/msn/projects/DL/image-classification/day6_transfer'
BEST_ACCURACY = None
IGNORE_PATTERNS = [
    '.vscode/',
    'data/',
    '__pycache__/',
    'wandb/',
    '*.pyc',
    'deploy.py',
    '.git/',
]
#定义元数据给机器看
card_data = ModelCardData(
    language='en',
    license='apache-2.0',
    tags=['vision', 'image-classification', 'transfer-learning', 'vit', 'pytorch'],
    library_name='timm',
    metrics={'accuracy': BEST_ACCURACY} if BEST_ACCURACY else {},
    datasets=['cifar100'],
    eval_results=[{
        'task_type': 'image-classification',
        'dataset': 'cifar100',
        'metric_type':'accuracy',
        'metric_value':BEST_ACCURACY,
        }]
)

#定义正文内容（给人看）
content = f"""
# Vision Transformer (ViT) Base Model Fine-tuned on CIFAR-100

This model is a fine-tuned version of **`vit_base_patch16_224`** on the **CIFAR-100** dataset.
It achieves an accuracy of **{BEST_ACCURACY:.2%}** on the validation set.

## Performance
| Metric | Value |
|:---:|:---:|
| **Accuracy** | **{BEST_ACCURACY:.4f}** |
| **Epochs** | 20 |
| **Batch Size** | 128 |

## Usage

Here is how to use this model to classify the images:

```python
import timm
import torch
from PIL import Image
from urllib.request import urlopen

#1.Load Model
model = timm.create_model("hf_hub:{REPO_ID}",pretrained=True)
model.eval()

#2.Prepare Image
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cifar100-test.jpg'
image = Image.open(urlopen(url))

#3.Predict
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))
print(f"Predicted Class ID: {{output.argmax().item()}}")
```
"""
#创建Model Card对象
print('Updating Model Card...')
card = ModelCard.from_template(card_data,content=content)
card.push_to_hub(REPO_ID)
print('Done!')

#2准备API
api = HfApi()

#3创建远程仓库
print(f'🚀 Creating repository: {REPO_ID}')
create_repo(repo_id=REPO_ID,repo_type='model',private=False)

#4上传本地文件夹
print(f'🚀 Uploading files from {LOCAL_DIR} to {REPO_ID}')
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type='model',
    ignore_patterns=IGNORE_PATTERNS,
    commit_message='Initial commit of transfer learning project files',
)


