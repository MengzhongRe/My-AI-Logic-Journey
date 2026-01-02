from huggingface_hub import HfApi,create_repo
import os

USER_NAME = 'YiMeng-SYSU'
REPO_NAME = 'vit-base-patch16-224-in21k-finetuned-cifar100'

LOCAL_FOLDER = '/home/msn/projects/DL/image-classification/day8_vit_base'

IGNORE_PATTERNS = [
    '.vscode/',
    '__pycache__',
    '*.pyc',
    '.git/',
    '.gitignore',
    'data/',
    'wandb/',
    'share_vit_base.py']

def main():
    api = HfApi()
    repo_id = f'{USER_NAME}/{REPO_NAME}'
    print(f'Creating repo {repo_id}...')

    try:
        create_repo(repo_id=repo_id, repo_type='model', private=False)
        print("✅ 仓库创建成功 (或已存在)")
    except Exception as e:
        print(f'仓库提示: {e}')

    print("☁️ 开始上传文件... (大文件可能需要几分钟)")

    api.upload_folder(
        folder_path=LOCAL_FOLDER,
        repo_id=repo_id,
        repo_type='model',
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f'Uploading files from {LOCAL_FOLDER}'
    )
    
    print('上传完成！')
    print(f'访问链接: https://huggingface.co/{repo_id}')

if __name__ == '__main__':
    main()
