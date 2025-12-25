import argparse
import torch
import os
from transformers import AutoTokenizer,AutoModelForSequenceClassification

MODEL_pATH = '../model_save'
DEVICE = 'cuda' if torch.cuda.is_availeble() else 'cpu'

ID2LABEL = {0:'积极（positive）',1:'消极（negative)'}


