import pandas as pd
import numpy as np
from importlib import import_module
import MeCab

def clustering(config):
    dataset = config['output_dir']
    path = f"outputs/{dataset}/clusters.csv"
    arguments_df = pd.read_csv(f"outputs/{dataset}/args.csv")
    arguments_array = arguments_df["argument"].values

    embeddings_df = pd.read_pickle(f"outputs/{dataset}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    clusters = config['clustering']['clusters']

    result = cluster_embeddings(
        docs=arguments_array,
        embeddings=embeddings_array,
        metadatas={
            "arg-id": arguments_df["arg-id"].values,
            "comment-id": arguments_df["comment-id"].values,
        },
        n_topics=clusters,
    )
    result.to_csv(path, index=False)


def cluster_embeddings(
    docs,
    embeddings,
    metadatas,
    min_cluster_size=2,
    n_components=2,
    n_topics=6,
):
    # 必要なモジュールのインポート
    SpectralClustering = import_module('sklearn.cluster').SpectralClustering
    HDBSCAN = import_module('hdbscan').HDBSCAN
    UMAP = import_module('umap').UMAP
    CountVectorizer = import_module('sklearn.feature_extraction.text').CountVectorizer
    BERTopic = import_module('bertopic').BERTopic

    # NEologdの辞書パスを指定
    neologd_path = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd'
    mecab = MeCab.Tagger(f"-d {neologd_path} -Ochasen")

    # 品詞フィルタリングを行うトークナイザー
    def tokenizer_mecab(text):
        node = mecab.parseToNode(text)
        words = []
        while node:
            # 名詞、動詞、形容詞だけを抽出する
            if node.feature.startswith('名詞') or node.feature.startswith('動詞') or node.feature.startswith('形容詞'):
                words.append(node.surface)
            node = node.next
        return words

    umap_model = UMAP(
        random_state=42,
        n_components=n_components,
    )
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size)

    # 日本語のストップワードリストを作成（必要に応じて拡張可能）
    japanese_stopwords = ["これ", "それ", "あれ", "こと", "もの", "ため", "よう", "さん", "する", "なる", "ある", "いる"]

    # 日本語トークナイザーとストップワードを使用する
    vectorizer_model = CountVectorizer(tokenizer=tokenizer_mecab, stop_words=japanese_stopwords)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    # トピックモデルのフィッティング
    _, __ = topic_model.fit_transform(docs, embeddings=embeddings)

    n_samples = len(embeddings)
    n_neighbors = min(n_samples - 1, 10)
    spectral_model = SpectralClustering(
        n_clusters=n_topics,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=42
    )
    umap_embeds = umap_model.fit_transform(embeddings)
    cluster_labels = spectral_model.fit_predict(umap_embeds)

    result = topic_model.get_document_info(
        docs=docs,
        metadata={
            **metadatas,
            "x": umap_embeds[:, 0],
            "y": umap_embeds[:, 1],
        },
    )

    result.columns = [c.lower() for c in result.columns]
    result = result[['arg-id', 'x', 'y', 'probability']]
    result['cluster-id'] = cluster_labels

    return result