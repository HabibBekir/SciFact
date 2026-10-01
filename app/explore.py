from statistics import mean

from app.data import doc_to_text, load_corpus, load_qrels, load_queries

corpus, queries, qrels = load_corpus(), load_queries(), load_qrels("data/scifact", "test")
print(len(corpus), "documents |", len(queries), "requêtes au total |", len(qrels), "requêtes de test")

lengths = [len(doc_to_text(d).split()) for d in corpus.values()]
print("mots par document : moyenne", round(mean(lengths)), "| max", max(lengths))
print("docs pertinents par requête :", round(mean(len(v) for v in qrels.values()), 2))

qid = next(iter(qrels))
print("\nREQUÊTE :", queries[qid])
for did in qrels[qid]:
    print("PERTINENT :", corpus[did]["title"])
    print(corpus[did]["text"][:300], "...")
