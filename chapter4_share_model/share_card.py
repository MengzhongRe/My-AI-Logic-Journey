from curses.ascii import US
from huggingface_hub import HfApi,create_repo,ModelCard,ModelCardData,EvalResult

USER_NAME = 'YiMeng-SYSU'
REPO_NAME = 'vit-base-patch16-224-in21k-finetuned-cifar100'
REPO_ID = f'{USER_NAME}/{REPO_NAME}'

BEST_ACC = 0.8358

#1.定义元数据给机器看
card_data = ModelCardData(
    language='en',
    license='apache-2.0',
    tags=['vision', 'image-classification', 'transfer-learning', 'vit', 'pytorch'],
    library_name='timm',
    model_name='vit_base_patch16_224',
    metrics=['accuracy'],
    datasets=['cifar100'],
    eval_results=[EvalResult(
        task_type='image-classification',
        dataset_type='cifar100',
        dataset_name='CIFAR_100',
        metric_type='accuracy',
        metric_value=BEST_ACC,
    )]
)
#2.定义正文内容给看人
content = f"""
# Vision Transformer (ViT) Base Model Fine-tuned on CIFAR-100

This model is a fine-tuned version of **`vit_base_patch16_224`** on the **CIFAR-100** dataset.
It achieves an accuracy of **{BEST_ACC:.2%}** on the validation set.

## Model Details
- **Architecture**: Vision Transformer (ViT)
- **Base Model**: ImageNet-21k pre-trained
- **Framework**: PyTorch + Timm
- **Hardware**: Trained on NVIDIA RTX 5070 Ti + AMD 9800X3D

## Performance
| Metric | Value |
|:---:|:---:|
| **Accuracy** | **{BEST_ACC:.4f}** |
| **Epochs** | 20 |
| **Batch Size** | 128 |

## Usage (Inference)

Here is how to use this model to classify an image:

```python
import timm
import torch
from PIL import Image
from urllib.request import urlopen

# 1. Load Model
model = timm.create_model("hf_hub:{REPO_ID}", pretrained=True)
model.eval()

# 2. Prepare Image
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cifar100-test.jpg'
img = Image.open(urlopen(url))

# 3. Predict
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))
print(f"Predicted Class ID: {{output.argmax().item()}}")
```
"""
#3.创建Model Card对象
print('Updating Model Card...')
card = ModelCard(content)
card.data = card_data
card.push_to_hub(REPO_ID)
print('Done!')