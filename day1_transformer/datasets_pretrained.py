from datasets import load_dataset

print('正在下载数据集')
dataset = load_dataset("seamew/ChnSentiCorp")

print('数据集下载完成')
print(dataset)

print(dataset['train'][0])