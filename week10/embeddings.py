import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import pickle

n_cores = cpu_count()
print(f"Number of Logical CPU cores: {n_cores}")


# Se declara una función para usar como inicializer con el mpdelo y el tokenizer
def init_pool(model_name):
    global tokenizer, model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertModel.from_pretrained(model_name)


def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Usar la representación del token [CLS]


# Función para generar embeddings en procesamiento paralelo y guardarlos en un array de numpy
def generate_embeddings_parallel(texts, model_name):
    print("Generando embeddings")
    num_cores = cpu_count()

    # Usamos 2 núcleos menos que los disponibles
    num_cores = max(1, num_cores - 2)
    print(f"Usando {num_cores} cores")
    pool = Pool(processes=num_cores, initializer=init_pool, initargs=(model_name,))

    resultados = []
    with tqdm(total=len(texts)) as pbar:
        for resultado in pool.imap(generate_bert_embeddings, texts):
            resultados.append(resultado)
            pbar.update()

    pool.close()
    pool.join()
    print("Embeddings generados exitosamente")
    return np.array(resultados).transpose(0, 2, 1)


if __name__ == "__main__":
    # Cargar el corpus
    wine_df = pd.read_csv("winemag-data_first150k.csv")
    corpus = wine_df["description"]
    print("Datos cargados exitosamente")

    # Load pre-trained BERT model and tokenizer
    embeddings = generate_embeddings_parallel(corpus, "bert-base-uncased")

    with open("bert_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    # Guardar la variable en un archivo npy
    np.save("embeddings.npy", embeddings)

    print("Bert Shape:", embeddings.shape)
