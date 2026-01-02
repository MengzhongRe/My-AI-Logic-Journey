from tabnanny import check
from transformers import AutoTokenizer,AutoModelForMaskedLM,set_seed
import torch

def main():
    checkpoint = 'bert-base-chinese'
    print(f'正在加载 {checkpoint} 模型和分词器...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    text = '.'
    inputs = tokenizer(text,return_tensors='pt').to(device)
    print(f'给定句子： {text}')
    print(f'input_ids: {inputs['input_ids']}')
    with torch.no_grad():
        logits = model(**inputs).logits
    
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # 8. 获取预测结果
    # 取出 [MASK] 位置的预测概率最大的那个词的 ID
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    
    # 解码回单词
    predicted_word = tokenizer.decode(predicted_token_id)
    
    print(f"🤖 模型预测结果: {predicted_word}")
    print(f"完整句子: {text.replace('[MASK]', predicted_word)}")

if __name__ == "__main__":
    main()
