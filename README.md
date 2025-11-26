# ChatbotForTravel — Developer & Migration Guide

This README documents the architecture, flow, API contract, and a step-by-step migration plan to move the existing Streamlit UI to a Flutter client while keeping the current backend logic. It is written so that an automated code assistant (e.g., GitHub Copilot) or a developer can read it and implement the migration with minimal ambiguity.

---

## Table of contents
- Project overview
- Runtime & how it runs locally
- Core components and responsibilities
- API contract (endpoints, request/response examples)
- How images and maps are delivered
- Migration goals and constraints
- Step-by-step Flutter migration plan (scaffold → polish)
- Flutter implementation details & code snippets
- Optional backend improvements (image proxy, auth, caching)
- Deployment, CI, and testing
- FAQs and troubleshooting

---

## Project overview

ChatbotForTravel is a travel assistant prototype. The main capabilities are:
- Recommend POIs (points of interest) from offline CSV datasets.
- Build day-by-day itineraries and route ordering using OSM routing.
- Provide weather integration and optional OpenAI-powered LLM responses.
- Offer an interactive Streamlit UI for exploration and quick demos.

The core business logic lives in the `core/` package. The Streamlit UI (`app.py` + `core/ui_plan_renderer.py`) is a thin presentation layer over that logic. A FastAPI server (`API.py`) has been added to expose the same core functions as REST endpoints so any UI (including Flutter) can call them.

Files of interest (paths are relative to repo root):
- `app.py` — Streamlit entry point (UI wiring)
- `core/ui_plan_renderer.py` — Streamlit rendering helpers and CSS
- `core/recommender.py` — POI loading and ranking logic
- `core/planner.py` — high-level plan generator
- `core/itinerary.py` — per-day itinerary builder using routing
- `core/weather.py` — weather helper (OpenWeather fallback)
- `core/llm_orchestrator.py` — OpenAI integration wrapper
- `API.py` — FastAPI REST wrapper exposing core logic
- `data/` — CSV data and cached OSM files

---

## How it runs locally

Prerequisites:
- Python 3.10+ in the provided `venv`
- Optional environment variables in `.env`: `OPENAI_API_KEY`, `OPENWEATHER_API_KEY`

Run the Streamlit UI (development):
```powershell
& D:/ChatbotBackend/ChatbotForTravel/venv/Scripts/Activate.ps1
streamlit run app.py
```

Run the API server (if you want to use the REST endpoints):
```powershell
& D:/ChatbotBackend/ChatbotForTravel/venv/Scripts/Activate.ps1
uvicorn API:app --host 0.0.0.0 --port 8000
```

Notes:
- Streamlit serves the UI at `http://localhost:8501` by default.
- FastAPI exposes OpenAPI docs at `http://localhost:8000/docs`.

---

## Core components and responsibilities

- `recommender.py`: loads category CSV files from `data/`, ranks using TF-IDF and heuristics (budget, weather). Returns a list of POI dicts.
- `planner.py`: orchestrates recommendations and routing to produce a set of day plans.
- `itinerary.py`: takes recommended POIs, splits them across days and finds a reasonable visiting order using graph-based utilities.
- `route_optimizer.py` & `geo_graph.py`: build/operate on a road graph (OSM) to compute distances and route information.
- `weather.py`: obtains weather information (OpenWeather if configured, otherwise simulated/fallback).
- `llm_orchestrator.py`: wraps OpenAI Chat completions; returns friendly fallback if `OPENAI_API_KEY` is absent.
- `intent_detector.py`: simple rule-based + Naive Bayes intent classification.
- `API.py`: REST API exposing `detect_intent`, `ask`, `recommend`, `plan`, `itinerary` and `health` endpoints.

---

## API contract (summary)

All endpoints expect JSON and return JSON. The FastAPI server is implemented in `API.py`. Below are the main endpoints with example payloads.

- `GET /health`
  - Response: `{ "status": "ok" }`

- `POST /detect_intent`
  - Request JSON: `{ "text": "Lên lịch trình 3 ngày ở Đà Nẵng" }`
  - Response: `{ "intent": "plan" }`

- `POST /ask`
  - Request JSON: `{ "prompt": "Hãy giới thiệu 3 địa điểm ở HCM", "system_prompt": "..." }`
  - Response: `{ "answer": "...LLM text..." }`

- `POST /recommend`
  - Request JSON example:
    ```json
    {
      "city": "Hồ Chí Minh",
      "category": "food",
      "user_query": "cà phê yên tĩnh",
      "taste_tags": ["Vietnamese"],
      "activity_tags": [],
      "budget_per_day": 300000,
      "walk_tolerance_km": 5.0
    }
    ```
  - Response: `{ "results": [ { "name": "...", "lat": ..., "lon": ..., "image_url1": "..." }, ... ] }`

- `POST /plan`
  - Request: plan parameters (city, budget etc.)
  - Response: `{ "weather": {...}, "routes": [...] }` (see `planner.generate_travel_plans` return)

- `POST /itinerary`
  - Request example:
    ```json
    { "city": "Hồ Chí Minh", "days": 2, "budget_vnd": 1500000 }
    ```
  - Response: `{ "weather": {...}, "itinerary": [ {"title":..., "pois": [...], "distance": ...}, ... ] }`

Full schema is in `API.py` (Pydantic models). Use `http://localhost:8000/docs` for interactive testing.

---

## How images & maps are delivered

- Image URLs in CSVs often point to Google-hosted resources like `lh3.googleusercontent.com`.
- The project contains `fix_google_img(url)` (located in `core/ui_plan_renderer.py` and `test.py` script logic) which rewrites Google `lh3` links to use the Weserv proxy: `https://images.weserv.nl/?url={host-and-path}`. This helps browsers fetch images that might otherwise be blocked or subject to CORS restrictions.
- The Streamlit UI embeds images with `st.image(url, width="stretch")`. Browsers fetch images directly from the URL.
- Maps: Folium maps are generated server-side and embedded via `st_folium(...)` which returns an interactive map in the Streamlit page.

If you migrate to Flutter, prefer using `CachedNetworkImage` (or similar) to cache images locally; ensure the images are publicly accessible or use the server-side image proxy described later.

---

## Migration goals & constraints

Goals:
- Preserve all business logic on the server (recommendation, itinerary, routing, LLM calls).
- Build a Flutter client that reproduces UI behaviors (cards, expanders, images, maps, buttons), but with custom backgrounds, improved navigation, and mobile-friendly UX.
- Keep secrets (OpenAI key, OpenWeather key) only on the server.

Constraints / decisions:
- Keep the backend Python code intact and expose it via the REST API (`API.py`). The Flutter client should be a pure client with no secret keys.
- Do not attempt to port Python NLP or ML code to Dart. Call the server instead.
- UI parity is important but the Flutter UI will use native widgets and map components; accept that exact DOM/CSS fidelity is not required.

---

## Step-by-step migration plan (high-level)

1. Prep and API hardening
   - Confirm `API.py` exposes the endpoints you need. Add any missing data endpoints (e.g., search, favorites, image proxy).
   - Optionally add API key auth for the REST API (FastAPI dependency) or JWT if you want per-user access.

2. Scaffold Flutter app
   - Create a Flutter project (e.g., `flutter create travel_flutter`).
   - Add required packages (examples below).

3. Implement core screens (minimal viable set)
   - Home / Search screen → calls `POST /recommend` and displays list.
   - POI detail screen → shows image(s), description, ratings, and a button to add to itinerary.
   - Itinerary screen → calls `POST /itinerary` and renders day cards.
   - Map screen → shows markers and route polylines.

4. Wire navigation and state management
   - Use Riverpod or Provider. Keep a single API client service that performs HTTP calls.

5. Polish UI and behaviors
   - Implement custom backgrounds, animations, image carousels, and expansion panels.

6. Add optional mobile-only features
   - Offline caching (Hive/SQLite), push notifications, and deep-linking into local map/navigation apps.

7. QA, performance, and deployment
   - Test on Android/iOS, measure image load times, consider server-side image resizing or CDN.

---

## Flutter implementation details & code snippets

Recommended Flutter packages
- `dio` — HTTP client
- `riverpod` — state management
- `cached_network_image` — image caching
- `flutter_map` + `latlong2` — OpenStreetMap rendering (lighter than Google Maps and easier to set up for web)
- `google_maps_flutter` — if you prefer Google Maps (requires API key and native config)

Example: Minimal API client (Dart + Dio)

```dart
// lib/services/api_service.dart
import 'package:dio/dio.dart';

class ApiService {
  final Dio _dio;
  ApiService(String baseUrl)
      : _dio = Dio(BaseOptions(baseUrl: baseUrl, connectTimeout: 5000));

  Future<List<dynamic>> recommend(Map<String, dynamic> body) async {
    final res = await _dio.post('/recommend', data: body);
    return res.data['results'] as List<dynamic>;
  }

  Future<Map<String, dynamic>> itinerary(Map<String, dynamic> body) async {
    final res = await _dio.post('/itinerary', data: body);
    return res.data as Map<String, dynamic>;
  }
}
```

Example: Displaying a POI card with cached image

```dart
// lib/widgets/poi_card.dart
import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';

class PoiCard extends StatelessWidget {
  final Map<String, dynamic> poi;
  final VoidCallback onTap;

  PoiCard({required this.poi, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final imageUrl = poi['image_url1'] ?? poi['image_url2'] ?? 'https://via.placeholder.com/300x200?text=No+Image';
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 4,
      child: InkWell(
        onTap: onTap,
        child: Row(
          children: [
            Flexible(
              flex: 1,
              child: ClipRRect(
                borderRadius: BorderRadius.horizontal(left: Radius.circular(12)),
                child: CachedNetworkImage(
                  imageUrl: imageUrl,
                  fit: BoxFit.cover,
                  height: 120,
                  placeholder: (c, u) => Center(child: CircularProgressIndicator()),
                  errorWidget: (c, u, e) => Icon(Icons.broken_image),
                ),
              ),
            ),
            Flexible(
              flex: 2,
              child: Padding(
                padding: EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(poi['name'] ?? 'Unknown', style: Theme.of(context).textTheme.titleMedium),
                    SizedBox(height: 8),
                    Text(poi['address'] ?? '', maxLines: 2, overflow: TextOverflow.ellipsis),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

Example: calling recommend from a screen

```dart
// inside a Riverpod or StatefulWidget async loader
final api = ApiService('http://10.0.2.2:8000'); // Android emulator -> host machine
final results = await api.recommend({ 'city': 'Hồ Chí Minh', 'category': 'food' });
// parse and show in ListView
```

Notes on host addresses during development
- Android emulator: use `10.0.2.2` to reach host machine `localhost`.
- iOS simulator: use `localhost` directly.

Mapping Streamlit constructs to Flutter widgets
- `st.columns([1,2])` → `Row` with `Flexible` and `flex` values.
- `st.expander` → `ExpansionTile` or `showModalBottomSheet`.
- `st.image` → `CachedNetworkImage` with `BoxFit.cover`.
- `st_folium` map → `flutter_map` with markers and polylines.

---

## Optional backend improvements

1. Image proxy endpoint

   Implement an endpoint `GET /image_proxy?url={encoded_url}` that fetches the remote image server-side and streams it to the client. This handles private or restricted images and normalizes caching headers. Example in FastAPI:

   ```python
   from fastapi.responses import StreamingResponse
   import requests

   @app.get('/image_proxy')
   def image_proxy(url: str):
       r = requests.get(url, stream=True, timeout=10)
       return StreamingResponse(r.raw, media_type=r.headers.get('content-type','image/jpeg'))
   ```

   Caveat: proxying adds bandwidth and privacy concerns; consider caching or signed URLs.

2. Auth / API keys

   - Add a simple API key header check or JWT-based user auth in `API.py` if you plan to expose the API publicly.

3. Rate limiting

   - Use `slowapi` or a reverse-proxy (Cloudflare, Nginx) to rate-limit LLM endpoints.

4. Pagination & streaming

   - Large recommendation results can use pagination; itinerary building can stay server-side and return only the requested page of days.

---

## Deployment & CI

Containerize the backend for production. Example `Dockerfile` (simple):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
```

CI suggestions:
- Run unit tests (if added) and linting in CI.
- Build Docker image and run basic health-checks.

---

## Testing & validation

Manual smoke tests:
- Start backend and call `/health`.
- Post a `recommend` request and ensure images and coordinates are present.
- Run `streamlit run app.py` to visually validate the cards, maps, and images.

Automated tests:
- Add unit tests for `recommender._cosine_rank`, `itinerary.build_itinerary`, and `weather.get_weather`.
- Add integration tests for the `API.py` endpoints using `pytest` + `httpx` test client.

---

## FAQs & troubleshooting

- Q: My Flutter app shows broken images.
  - A: Confirm the image URLs are reachable from the mobile device (emulator/device). If needed, use `/image_proxy` to serve images from the server.

- Q: OpenAI responses are missing.
  - A: Ensure `OPENAI_API_KEY` is set in the server environment (and not in the Flutter client). Check `core/llm_orchestrator.py` for fallback behavior.

- Q: Why keep logic on server?
  - A: Reusing the Python ML/NLP and routing logic avoids porting complex code to Dart, keeps secrets safe, and centralizes data access.

---

## Next actions (recommended, in order)
1. If you want, I can scaffold the Flutter starter with the `ApiService` and a sample `HomeScreen` that calls `/recommend` and shows POI cards.
2. Or I can add the `image_proxy` endpoint to `API.py` and a small example showing how Flutter should request proxied images.
3. Or add unit tests and CI config for the Python backend.

Choose one and I will proceed to scaffold or implement it.

---

Author: Copilot-style assistant (implementation companion)
