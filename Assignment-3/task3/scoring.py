"""Gold-free equivalence grouping. The same scorer grades only after selection."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from functools import lru_cache
import re
import math
NO_ANSWER = '__NO_ANSWER__'

def boxed(text):
 start = text.rfind(r'\boxed{')
 if start < 0: return None
 start += len(r'\boxed{'); depth = 1
 for pos in range(start, len(text)):
  if text[pos] == '{': depth += 1
  elif text[pos] == '}': depth -= 1
  if depth == 0: return text[start:pos].strip() or None
 return None

@lru_cache(maxsize=32768)
def parse_math(text):
 from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig
 return parse(r'\boxed{' + text + '}', extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])

class Scorer:
 def __init__(self, dataset, metadata=None):
  self.dataset = dataset; self.metadata = metadata or {}
 def normalize(self, answer):
  if answer is None or not str(answer).strip() or answer == NO_ANSWER: return NO_ANSWER
  text = str(answer).strip()
  if self.dataset == 'math': return text
  m = self.metadata
  if m.get('question_type') == 'multi_choice':
   choices = list(map(str, m['choices'])); letters = [chr(65+i) for i in range(len(choices))]
   match = re.findall(r'\(([a-zA-Z])\)', text)
   if match: text = match[0].upper()
   if text in letters: return text
   def distance(a,b):
    row=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
     new=[i]
     for j,cb in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(ca!=cb)))
     row=new
    return row[-1]
   return letters[min(range(len(choices)),key=lambda i:distance(text,choices[i]))]
  try:
   if m.get('answer_type') == 'integer': return str(int(float(text)))
   if m.get('answer_type') == 'float': return str(round(float(text), int(m['precision']))) if math.isfinite(float(text)) else NO_ANSWER
  except (ValueError, OverflowError): return NO_ANSWER
  return text
 def equivalent(self, a, b):
  a, b = self.normalize(a), self.normalize(b)
  if a == NO_ANSWER or b == NO_ANSWER: return a == b
  if a == b: return True
  if self.dataset != 'math': return False
  from math_verify import verify
  pa, pb = parse_math(a), parse_math(b)
  return bool(pa and pb and verify(pa, pb))
 def correct(self, answer, gold):
  return self.normalize(answer) != NO_ANSWER and self.equivalent(gold, answer)
 def groups(self, answers):
  representatives=[]; ids=[]
  for answer in answers:
   value=self.normalize(answer)
   for i, rep in enumerate(representatives):
    if self.equivalent(rep, value): break
   else:
    i=len(representatives); representatives.append(value)
   ids.append(i)
  return ids, representatives
