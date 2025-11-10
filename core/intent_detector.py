import os, re, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL = os.path.join("data","intent_model.pkl")

SEED = [
    ("Thời tiết Đà Lạt", "weather"),
    ("Hà Nội hôm nay mưa không", "weather"),
    ("Gợi ý địa điểm tham quan ở Huế", "lookup"),
    ("Tìm quán cà phê yên tĩnh", "lookup"),
    ("Lên lịch trình 3 ngày ở Đà Nẵng", "plan"),
    ("Tạo route tham quan 1 ngày ở Sài Gòn", "plan"),
    ("Đề xuất khách sạn trung tâm", "lookup"),
]

def _train():
    X, y = zip(*SEED)
    vec = TfidfVectorizer()
    Xv = vec.fit_transform(X)
    clf = MultinomialNB().fit(Xv, y)
    with open(MODEL, "wb") as f: pickle.dump((vec, clf), f)

def _local(text: str) -> str:
    if not os.path.exists(MODEL): _train()
    vec, clf = pickle.load(open(MODEL, "rb"))
    return clf.predict(vec.transform([text]))[0]

def _rule(t: str):
    t = t.lower()
    if re.search(r"\b(thời tiết|mưa|nắng|nhiệt độ|gió)\b", t): return "weather"
    if re.search(r"\b(lịch trình|kế hoạch|route|tuyến|đi.*ngày|itinerary|plan)\b", t): return "plan"
    if re.search(r"\b(địa điểm|tham quan|quán|cà phê|nhà hàng|khách sạn|đi đâu|gợi ý)\b", t): return "lookup"
    return None

def detect_intent(text: str) -> str:
    rule = _rule(text)
    if rule: return rule
    try:
        return _local(text)
    except Exception:
        return "lookup"
