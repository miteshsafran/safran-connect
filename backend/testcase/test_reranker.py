from sentence_transformers import CrossEncoder


model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

question = "How many casual leaves can an employee take?"

documents = [
    "Employees are entitled to 12 casual leave days per year.",
    "Employees can claim travel expenses after business trips.",
    "The office working hours are from 9 AM to 6 PM."
]

pairs = [
    [question, document]
    for document in documents
]

scores = model.predict(pairs)

for document, score in zip(documents, scores):
    print(f"{score:.4f} -> {document}")