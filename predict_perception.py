import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# === Load NLP Models ===
toxicity_pipe = pipeline("text-classification", model="unitary/toxic-bert", truncation=True)
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# === GCN Model ===
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.3, train=self.training)
        x = self.conv2(x, edge_index)
        return self.lin(x).squeeze()

# === Utility Functions ===
def safe_int(x):
    try: return int(x)
    except: return 0

emotion_map = {
    "joy": 1.0, "surprise": 0.5, "neutral": 0.0, "unknown": 0.0,
    "sadness": -0.5, "anger": -0.7, "disgust": -0.8, "fear": -1.0
}

def extract_features(posts):
    features = []

    for post in tqdm(posts, desc="Extracting features"):
        likes = safe_int(post.get("likes"))
        shares = safe_int(post.get("shares"))
        views = safe_int(post.get("views"))
        followers = safe_int(post.get("followers"))
        is_blue = 1 if post.get("isBlue", False) else 0
        mentions = post.get("user_mentions") or []
        hashtags = post.get("hashtags") or []
        comments = post.get("comments") or []

        comment_texts = [c.get("content", "").strip() for c in comments if c.get("content")]
        comment_followers = [safe_int(c.get("followers")) for c in comments]
        blue_commenters = [1 if c.get("is_blueTick") else 0 for c in comments]

        aggregated_comments = " ".join(comment_texts) if comment_texts else "neutral"

        # ✅ Use overall_probabilities to calculate avg_sentiment_score
        score_map = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0,
            "sarcastic": -0.5
        }
        try:
            probs = post.get("overall_probabilities", {})
            avg_sentiment = round(sum(score_map[label] * probs.get(label, 0.0) for label in score_map), 3)
        except:
            avg_sentiment = 0.0

        avg_followers = round(np.mean(comment_followers), 2) if comment_followers else 0.0
        blue_ratio = round(np.mean(blue_commenters), 2) if blue_commenters else 0.0

        try:
            emotion_label = emotion_pipe(aggregated_comments[:512])[0][0]['label']
        except:
            emotion_label = "unknown"
        try:
            toxicity_score = toxicity_pipe(aggregated_comments[:512])[0]['score']
        except:
            toxicity_score = 0.0

        row = {
            "likes": likes,
            "shares": shares,
            "views": views,
            "followers": followers,
            "is_blue": is_blue,
            "comment_count": len(comments),
            "engagement_score": likes + shares + views + len(comments),
            "avg_comment_followers": avg_followers,
            "blue_tick_comment_ratio": blue_ratio,
            "toxicity_score": round(toxicity_score, 3),
            "emotion_label": emotion_map.get(emotion_label, 0.0),
            "avg_mention_count": len(mentions),
            "avg_hashtag_count": len(hashtags),
            "avg_sentiment_score": avg_sentiment
        }

        bert = bert_model.encode(aggregated_comments)
        for i, val in enumerate(bert):
            row[f"bert_embedding_{i+1}"] = val

        features.append(row)

    return pd.DataFrame(features)

def build_graph(df):
    features_to_normalize = [
        "likes", "shares", "views", "comment_count", "followers",
        "is_blue", "engagement_score", "avg_comment_followers",
        "blue_tick_comment_ratio", "avg_mention_count", "avg_hashtag_count"
    ]
    df[features_to_normalize] = df[features_to_normalize].clip(
        df[features_to_normalize].quantile(0.01),
        df[features_to_normalize].quantile(0.99),
        axis=1
    )
    df[features_to_normalize] = MinMaxScaler().fit_transform(df[features_to_normalize])
    df[features_to_normalize] = df[features_to_normalize].round(3)

    final_features = features_to_normalize + ["toxicity_score", "emotion_label", "avg_sentiment_score"]
    df_final = df[final_features].reset_index(drop=True)

    x = torch.tensor(df_final.values, dtype=torch.float)

    k = min(4, len(df_final) - 1)
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(df_final.values)
    neighbors = knn.kneighbors(return_distance=False)

    edge_index = []
    for src_idx, nbrs in enumerate(neighbors):
        for dst_idx in nbrs:
            if src_idx != dst_idx:
                edge_index.append([src_idx, dst_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def predict(json_path, model_path="gcn_model.pt"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_posts = []
    post_index_map = []

    for brand, posts in data.items():
        if not isinstance(posts, list):
            continue
        for i, post in enumerate(posts):
            all_posts.append(post)
            post_index_map.append((brand, i))

    df = extract_features(all_posts)
    graph = build_graph(df)

    input_dim = graph.x.shape[1]
    model = GCN(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(graph).numpy()

    df["Predicted_Perception"] = preds
    overall_score = float(np.mean(preds))

    print(df[["Predicted_Perception"]])
    print(f"\n🌟 Overall Brand Perception Score: {overall_score:.4f}")

    for idx, (brand, post_idx) in enumerate(post_index_map):
        data[brand][post_idx]["post_perception"] = round(float(preds[idx]), 5)

    for brand in list(data.keys()):
        if isinstance(data[brand], list):
            data[brand + "_summary"] = {
                "perception": round(overall_score, 5)
            }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\n Post and overall perception scores added to the JSON successfully.\n")

    print(df[[
        "likes", "shares", "views", "followers", "engagement_score",
        "avg_comment_followers", "avg_sentiment_score", "emotion_label", "toxicity_score"
    ]])

# === Run ===
if __name__ == "__main__":
    predict("testing_new_updated_overall_1.json")
