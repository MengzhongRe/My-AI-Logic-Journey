from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
import torch

def main():
    checkpoint = 'gpt2'
    print(f'正在加载 {checkpoint} 模型和分词器...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    text = 'Deep learning is '
    inputs = tokenizer(text,return_tensors='pt').to(device)
    print(f'给定开头： {text}')
    print('AI 生成的后续内容为：')
    outputs = model.generate(
        max_length=100,
        **inputs,
        temperature=0.7,
        do_sample=True,
    )
    generated_text = tokenizer.decode(outputs[0],skip_special_tokens=True)

    print('-' * 40)
    print(generated_text)
    print('-' * 40)

if __name__ == '__main__':
    main()


