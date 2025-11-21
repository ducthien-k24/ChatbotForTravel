# 🌍 TravelGPT+ API Documentation

## What Makes This API Special? 🚀

**This API does things ChatGPT CANNOT do:**

### 1. ✅ Real Street-Level Routing
- Uses **OpenStreetMap (OSM)** real road network data
- **Dijkstra's algorithm** for shortest path calculations
- Actual distances on **real streets**, not straight-line estimates
- Turn-by-turn navigation with street names

### 2. ✅ Advanced Graph Optimization
- **Minimum Spanning Tree (MST)** for route optimization
- **Traveling Salesman Problem (TSP)** heuristics
- Multi-modal transport optimization (walk, bike, drive)
- Cost-aware path selection

### 3. ✅ Live Data Integration
- **Real-time weather** from OpenWeatherMap
- **Live POI data** from OpenStreetMap
- **High-quality images** from Unsplash + Google Places
- Actual coordinates and opening hours

### 4. ✅ Professional Document Generation
- Travel agency-style **HTML packages**
- Embedded images and maps
- Turn-by-turn navigation instructions
- Cost breakdowns and timings

### 5. ✅ ML-Powered Personalization
- **TF-IDF + Cosine Similarity** for POI matching
- Weather-aware scoring adjustments
- Budget optimization algorithms
- Activity preference learning

---

## 📦 Installation

### 1. Install Dependencies
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys
notepad .env
```

**Required API Keys (at least one image source):**
- `OPENAI_API_KEY` - For natural language processing
- `UNSPLASH_ACCESS_KEY` - For high-quality POI images (recommended)
- `GOOGLE_PLACES_API_KEY` - Alternative image source
- `OPENWEATHER_API_KEY` - For real-time weather (optional)

### 3. Run the API
```bash
# Development mode with auto-reload
python api.py

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000
```

API will be available at: **http://localhost:8000**

---

## 🔥 API Endpoints

### 1. Generate Complete Itinerary
**POST** `/api/itinerary/generate`

The flagship endpoint that creates a comprehensive travel plan.

**Request Body:**
```json
{
  "city": "Hồ Chí Minh",
  "days": 3,
  "budget_vnd": 2000000,
  "taste_tags": ["Vietnamese", "Cafe", "Seafood"],
  "activity_tags": ["attraction", "food", "shopping"],
  "walk_tolerance_km": 6.0,
  "transport": "xe máy/ô tô",
  "include_images": true,
  "include_navigation": true,
  "optimize_cost": false,
  "weather_aware": true
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2025-11-22T10:30:00",
  "itinerary": {
    "weather": "nắng nhẹ",
    "days": [
      {
        "order": [
          {
            "name": "Bến Thành Market",
            "category": "shopping",
            "lat": 10.772461,
            "lon": 106.698055,
            "avg_cost": 200000,
            "description": "Famous market...",
            "image_url": "https://images.unsplash.com/...",
            "final": 0.92
          }
        ],
        "distance_km": 12.5,
        "total_cost_vnd": 850000,
        "navigation": {
          "total_distance_km": 12.5,
          "total_duration_min": 25,
          "segments": [
            {
              "from": "Bến Thành Market",
              "to": "War Museum",
              "distance_km": 2.3,
              "duration_min": 5,
              "directions": [
                {
                  "step": 1,
                  "instruction": "Đi theo Lê Lợi",
                  "distance_m": 450,
                  "duration_min": 1
                }
              ],
              "route_coordinates": [
                {"lat": 10.772, "lon": 106.698}
              ]
            }
          ]
        }
      }
    ]
  },
  "metadata": {
    "total_days": 3,
    "total_pois": 15,
    "total_distance_km": 38.2,
    "weather": {"temp": 32, "humidity": 75}
  }
}
```

---

### 2. Export as Professional HTML Document
**POST** `/api/itinerary/export/html`

Generates a travel agency-style package document.

**Request:** Same as `/api/itinerary/generate`

**Response:** Beautiful HTML page with:
- Professional layout and styling
- Embedded images for each POI
- Interactive navigation steps
- Cost breakdowns
- Weather advisories
- Print-ready format

---

### 3. Get POI Recommendations
**POST** `/api/pois/recommend`

Get personalized point-of-interest suggestions.

**Request:**
```json
{
  "city": "Đà Lạt",
  "query": "romantic cafes with mountain view",
  "taste_tags": ["Cafe", "French"],
  "activity_tags": ["attraction", "food"],
  "budget_per_day": 1500000,
  "walk_tolerance_km": 5.0,
  "limit": 10
}
```

**Response:**
```json
{
  "success": true,
  "count": 10,
  "pois": [
    {
      "name": "Mê Linh Coffee Garden",
      "category": "cafe",
      "lat": 11.9404,
      "lon": 108.4583,
      "avg_cost": 80000,
      "description": "Scenic mountain cafe...",
      "image_url": "https://images.unsplash.com/...",
      "final": 0.88
    }
  ],
  "weather": {"temp": 18, "description": "mây rải rác"}
}
```

---

### 4. Calculate Optimal Route
**POST** `/api/route/calculate`

Get actual street-level routing between points.

**Request:**
```json
{
  "city": "Hà Nội",
  "waypoints": [
    {"lat": 21.0285, "lon": 105.8542},
    {"lat": 21.0349, "lon": 105.8489},
    {"lat": 21.0278, "lon": 105.8342}
  ],
  "transport_mode": "drive"
}
```

**Response:**
```json
{
  "success": true,
  "route": {
    "total_distance_km": 5.3,
    "total_duration_min": 11,
    "segments": [
      {
        "from": "Waypoint 1",
        "to": "Waypoint 2",
        "distance_km": 2.1,
        "duration_min": 4,
        "directions": [
          {
            "step": 1,
            "instruction": "Đi theo Hàng Bài",
            "distance_m": 320,
            "duration_min": 1
          }
        ],
        "route_coordinates": [...]
      }
    ]
  }
}
```

---

### 5. Intelligent Chat Interface
**POST** `/api/chat`

Natural language interface with automatic intent detection.

**Request:**
```json
{
  "message": "Tôi muốn lên lịch trình 3 ngày ở Đà Nẵng với hoạt động biển",
  "city": "Đà Nẵng",
  "conversation_id": "user123_session456"
}
```

**Response:**
```json
{
  "intent": "plan",
  "response": "Đây là lịch trình 3 ngày ở Đà Nẵng...",
  "data": {
    "itinerary": {...}
  }
}
```

**Supported Intents:**
- `weather` - Weather queries
- `lookup` - POI search
- `plan` - Itinerary creation
- `general` - Other queries

---

### 6. Get Supported Cities
**GET** `/api/cities`

List all cities with coverage.

**Response:**
```json
{
  "cities": [
    {"name": "Hồ Chí Minh", "bbox": [10.85, 10.70, 106.83, 106.63]},
    {"name": "Đà Lạt", "bbox": [11.97, 11.90, 108.47, 108.40]},
    {"name": "Hà Nội", "bbox": [21.08, 20.95, 105.90, 105.75]},
    {"name": "Đà Nẵng", "bbox": [16.10, 15.90, 108.30, 108.10]},
    {"name": "Huế", "bbox": [16.50, 16.42, 107.63, 107.52]},
    {"name": "Nha Trang", "bbox": [12.28, 12.18, 109.22, 109.12]}
  ]
}
```

---

### 7. Health Check
**GET** `/health`

Check API status and enabled features.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-22T10:30:00",
  "version": "2.0.0",
  "features": {
    "osm_routing": true,
    "image_fetching": true,
    "turn_by_turn": true,
    "cost_optimization": true,
    "weather_aware": true,
    "pdf_export": true
  }
}
```

---

## 🎯 Usage Examples

### cURL Examples

```bash
# Generate itinerary
curl -X POST http://localhost:8000/api/itinerary/generate \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Hồ Chí Minh",
    "days": 2,
    "budget_vnd": 1500000,
    "include_images": true,
    "include_navigation": true
  }'

# Get POI recommendations
curl -X POST http://localhost:8000/api/pois/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Đà Lạt",
    "query": "romantic cafes",
    "limit": 5
  }'

# Chat interface
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Thời tiết Hà Nội hôm nay thế nào?",
    "city": "Hà Nội"
  }'
```

### Python Example

```python
import requests

# Generate itinerary
response = requests.post(
    "http://localhost:8000/api/itinerary/generate",
    json={
        "city": "Hồ Chí Minh",
        "days": 3,
        "budget_vnd": 2000000,
        "taste_tags": ["Vietnamese", "Cafe"],
        "activity_tags": ["attraction", "food"],
        "include_images": True,
        "include_navigation": True
    }
)

itinerary = response.json()
print(f"Total POIs: {itinerary['metadata']['total_pois']}")
print(f"Total Distance: {itinerary['metadata']['total_distance_km']} km")
```

### JavaScript Example

```javascript
// Generate itinerary
const response = await fetch('http://localhost:8000/api/itinerary/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    city: 'Hồ Chí Minh',
    days: 2,
    budget_vnd: 1500000,
    include_images: true,
    include_navigation: true
  })
});

const data = await response.json();
console.log('Itinerary:', data.itinerary);
```

---

## 🔧 Advanced Features

### Cost Optimization
Enable `optimize_cost: true` to automatically fit itinerary within budget:
```json
{
  "optimize_cost": true,
  "budget_vnd": 1000000
}
```

### Weather-Aware Planning
Enable `weather_aware: true` to adjust for weather:
- Rain → Indoor venues prioritized
- Hot weather → Cafes and shaded areas
- Cool weather → Outdoor activities

### Multi-Modal Routing
Specify transport mode in route calculations:
```json
{
  "transport_mode": "drive"  // or "walk", "bike"
}
```

### Image Quality Control
Control image fetching:
```json
{
  "include_images": true  // Fetch from Unsplash/Google
}
```

---

## 🚀 Performance

### First Run (No Cache)
- **Download OSM data:** 3-5 minutes
- **Build graph:** Automatic caching
- **Generate itinerary:** 10-30 seconds

### Subsequent Runs (With Cache)
- **Load from cache:** 5-15 seconds
- **Generate itinerary:** 5-30 seconds
- **Add images:** 2-5 seconds (parallel)
- **Export HTML:** 1-2 seconds

### Optimization Tips
1. **Cache is permanent** - delete only when needed
2. **Parallel image fetching** - uses async/await
3. **Graph algorithms optimized** - MST + Dijkstra
4. **Limit POIs** - fewer points = faster routing

---

## 🆚 Comparison: TravelGPT+ vs ChatGPT

| Feature | TravelGPT+ API | ChatGPT |
|---------|----------------|---------|
| Real street routing | ✅ Yes (OSM + Dijkstra) | ❌ No (estimates only) |
| Actual distances | ✅ Yes (graph-based) | ❌ No (straight-line) |
| Turn-by-turn nav | ✅ Yes (street names) | ❌ No |
| Live POI data | ✅ Yes (OSM real-time) | ❌ No (training cutoff) |
| Image fetching | ✅ Yes (API integration) | ❌ No (text only) |
| Route optimization | ✅ Yes (MST/TSP algorithms) | ❌ No (no algorithms) |
| Cost optimization | ✅ Yes (budget algorithms) | ❌ Limited |
| Weather integration | ✅ Yes (live API) | ❌ No (cannot access) |
| HTML/PDF export | ✅ Yes (professional docs) | ❌ No |
| Graph algorithms | ✅ Yes (NetworkX) | ❌ No |
| Coordinates | ✅ Yes (lat/lon precise) | ❌ Approximate |
| Opening hours | ✅ Yes (OSM data) | ❌ Outdated |

---

## 📊 API Rate Limits

No rate limits currently enforced. For production:
- Implement Redis-based rate limiting
- Add authentication with API keys
- Monitor usage per endpoint

---

## 🐛 Troubleshooting

### "No cache found" - Long initial load
**Normal behavior** on first run. Cache files will be created.

### "Cannot find path between nodes"
POIs might be on disconnected road network parts. The API will fallback gracefully.

### Images not loading
Check API keys in `.env`:
- `UNSPLASH_ACCESS_KEY`
- `GOOGLE_PLACES_API_KEY`

### Slow routing
Large cities (Ho Chi Minh) have 2M+ nodes. Consider:
- Reducing number of POIs
- Using smaller bbox
- Upgrading server RAM

---

## 🔐 Security Notes

- Never commit `.env` file
- Use environment variables in production
- Implement API authentication
- Rate limit external endpoints
- Sanitize user inputs

---

## 📞 Support

Issues? Feature requests?
- Check documentation
- Review code comments
- Test with smaller cities first

---

**Built with ❤️ using FastAPI, OSMnx, NetworkX, and AI**
