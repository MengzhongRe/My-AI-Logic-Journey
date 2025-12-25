from transformers import pipeline

classifer = pipeline("sentiment-analysis")
result = classifer("我爱中国！")
print(result)

result = classifer("I hate waiting for my code to compile.")
print(result)