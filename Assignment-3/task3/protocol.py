import hashlib
import json

VERSION = 'task3-v1'
SEED = 6304
BUDGETS = (1, 2, 4, 8, 16, 32)
MODEL = 'Qwen/Qwen2.5-3B-Instruct'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
SYSTEM = 'You are a helpful assistant.'
SUFFIX = r'Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}.'
REVIEW = r'Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \boxed{}.'
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
RELEASE = 'https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23'
FILES = {
 'README.MD': '1RARGiUK_jZpOii-b8kyGLoejaldMUAa3',
 'math/generations.parquet': '1SaWxZ4mpycw2BWZ5G0dahoYZ6Hh8tPWP',
 'math/probability_metrics.parquet': '1RCYHrTnUKtF7gHLSTe92x6W9ZsCOekfQ',
 'mathvista/generations.parquet': '1sedMpWLI_UgPvAQVj7tS3HYVH0ARS1yA',
 'mathvista/probability_metrics.parquet': '1f2XAu_J5NviYHEhtkRrFbKc7b2us1WiW',
 'mathvista/attention_metrics.parquet': '1Nzg2VwE2qFzA10BF1lTxEcBZCePBe51q',
}

def digest(value):
 return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def seed_for(*parts):
 return int(digest([SEED, *parts])[:8], 16)
