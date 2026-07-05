import json
from sentence_transformers import SentenceTransformer, util

corpus = json.load(open("code_corpus.json", encoding="utf-8"))[100:]
question = [i for i in json.load(open("eval_questions.json", encoding="utf-8")) if int(i["correct_chunk_id"][5]) == 0][:15]
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embedding_c = model.encode([i["description"] for i in corpus])
embedding_q = model.encode([i["query"] for i in question])
precision = ''
for i in range(len(question)):
    q = util.cos_sim(embedding_q[i], embedding_c)[0]
    top = sorted(q, reverse=True)[:3]
    f = q[int(question[i]["correct_chunk_id"][5:]) - 1]
    print(f, top)
    if f in top:
        precision += '1'
    else:
        precision += '0'
print(precision, sum(int(i) for i in list(precision)))
