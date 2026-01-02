---
下面的示例是直接用**`huggiongface_hub python SDK`** 将你本地的模型文件上传到**huggingface_hub**上。
```python
from huggingface_hub import HfApi,create_repo,ModelCard,ModelCardData

#
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
```
---


编写高质量的 **Model Card (模型卡片)** 是让你的模型从“个人练手项目”变成“开源社区资产”的关键一步。

一个优秀的 Model Card 需要满足两点：
1.  **机器可读 (YAML)**：让 Hugging Face 的搜索引擎能找到你（通过 Task, Dataset, Metrics）。
2.  **人类可读 (Markdown)**：让别人哪怕没看代码，复制粘贴一段话就能跑起来。

我为你总结了 **“快速”** 且 **“准确”** 搞定它的两种最佳方案。

---

### 方案一：自动化生成（最推荐 🔥）
**适合场景**：集成在训练/上传脚本中，自动填入准确率（Accuracy）等变量，**绝对不会写错数据**。

利用 `huggingface_hub` 的 `ModelCard` 模块，你可以像写代码一样写文档。

请在你的 `share_script.py` 或 `deploy.py` 中加入以下逻辑：

```python
from huggingface_hub import HfApi, ModelCard, ModelCardData

# 假设这些是你训练脚本里的变量
USER = "Mengzhongren"
REPO_NAME = "vit-base-cifar100-v1"
BEST_ACC = 0.8358  # 你的最佳验证集准确率
REPO_ID = f"{USER}/{REPO_NAME}"

# ==========================================
# 1. 定义元数据 (给机器看)
# ==========================================
card_data = ModelCardData(
    language="en",
    license="apache-2.0",
    library_name="timm",
    tags=["image-classification", "vit", "vision", "pytorch"],
    datasets=["cifar100"],
    metrics=["accuracy"],
    # 关键：自动把变量里的准确率写入元数据
    eval_results=[{
        "task_type": "image-classification",
        "dataset_type": "cifar100",
        "metric_type": "accuracy",
        "metric_value": BEST_ACC
    }]
)

# ==========================================
# 2. 定义正文内容 (给人看)
# 使用 f-string 自动填充信息
# ==========================================
content = f"""
# ViT-Base Fine-tuned on CIFAR-100

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

# ==========================================
# 3. 合并并上传
# ==========================================
print("📝 Updating Model Card...")
card = ModelCard.from_template(card_data, content=content)
card.push_to_hub(REPO_ID)
print("✅ Done!")
```

---

### 方案二：手动套用“万能模板” (最快上手)
**适合场景**：已经上传了文件，只想在网页上快速补全说明书。

直接在本地新建一个 `README.md`，复制下面的内容，修改 `{}` 里的字，然后上传即可。

```markdown
---
language:
- en
license: apache-2.0
library_name: timm
tags:
- image-classification
- vision
- pytorch
datasets:
- cifar100
metrics:
- accuracy
---

# {你的模型名字，如 ResNet50-CIFAR10}

This model is a fine-tuned version of **{基础模型名}** on the **{数据集名}** dataset.

## Model Description
- **Model type:** Image Classification
- **Backbone:** {ResNet50 / ViT-Base}
- **Pretrained Dataset:** {ImageNet-1k / ImageNet-21k}
- **Fine-tuned Dataset:** {CIFAR-10 / CIFAR-100}

## Results
| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **{97.21%}** |
| Training Loss | {0.05} |

## How to Get Started with the Model

Use the code below to get started with the model:

```python
import timm
import torch

# Load the model
model = timm.create_model("hf_hub:{你的用户名/仓库名}", pretrained=True)
model.eval()

# Check configuration
config = model.default_cfg
print(f"Input image size: {config['input_size']}")
print(f"Mean: {config['mean']}")
print(f"Std: {config['std']}")
```

## Training Details
- **Hardware**: NVIDIA RTX 5070 Ti
- **Optimizer**: AdamW
- **Learning Rate Strategy**: Differential Learning Rates (Backbone vs Head)
- **Epochs**: {20}
```

---

### 💡 核心技巧：怎么写才显得“专业”？

1.  **Usage 代码块必不可少**：
    这是别人（包括未来的你）能不能用这个模型的关键。一定要放一段 **Copy-Paste 就能跑** 的 Python 代码。

2.  **YAML 头部 (Frontmatter) 很重要**：
    文件最上方 `---` 包裹的内容决定了你的模型能不能被搜到。
    *   **`pipeline_tag: image-classification`**：加上这个，你的模型页面右边就会出现一个“上传图片试试看”的测试窗口（Inference Widget）。这对展示效果非常有用！

3.  **引用基础库**：
    在 `tags` 或 `library_name` 里写上 `timm`，这样 timm 的官方文档或者社区就能关联到你的模型。

**建议**：
先用 **方案一** 写个脚本跑一遍。以后每次训练出新 SOTA，运行一下脚本，README 里的准确率自动更新，既准确又省心！