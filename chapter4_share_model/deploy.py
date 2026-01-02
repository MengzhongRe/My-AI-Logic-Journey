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


