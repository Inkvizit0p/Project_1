import json
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mc

corpus = json.load(open("code_corpus.json", encoding="utf-8"))[100:]
question = [i for i in json.load(open("eval_questions.json", encoding="utf-8")) if int(i["correct_chunk_id"][5]) == 0][:15]
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedding_c = model.encode([i["description"] for i in corpus])
embedding_q = model.encode([i["query"] for i in question])
precision = 0
for i in range(len(question)):
    cos = util.cos_sim(embedding_q[i], embedding_c)[0]
    if cos[int(question[i]["correct_chunk_id"][5:]) - 1] in sorted(cos, reverse=True)[:3]:
        precision += 1
print(precision)
#Создание графика кластеров
coord = TSNE(n_components=2).fit_transform(embedding_c)
clusters = []
for i in range(len(coord)):
    if corpus[i]["category"] == "auth":
        color = mc.to_rgb("#E74C3C")
    elif corpus[i]["category"] == "database":
        color = mc.to_rgb("#3498DB")
    elif corpus[i]["category"] == "http":
        color = mc.to_rgb("#2ECC71")
    elif corpus[i]["category"] == "validation":
        color = mc.to_rgb("#F39C12")
    else:
        color = mc.to_rgb("#9B59B6")
    clusters.append([color, coord[i]])
fig, ax = plt.subplots()
for color, cluster in clusters:
    ax.scatter(cluster[0], cluster[1], color=color)
print(clusters)
plt.show()
