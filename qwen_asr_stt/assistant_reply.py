#!/usr/bin/env python3
import json
import re
import urllib.parse
import urllib.request


WEATHER_CODE_TEXT = {
    0: "晴",
    1: "大致晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "有霜雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "较强阵雨",
    82: "暴雨阵雨",
    95: "雷暴",
}


def is_weather_query(text: str) -> bool:
    lowered = text.strip().lower()
    keywords = ["天气", "气温", "温度", "下雨", "下雪", "冷不冷", "热不热", "weather", "temperature"]
    return any(keyword in lowered for keyword in keywords)


def post_chat_reply(
    user_text: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 160,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个简短中文语音助手。"
                    "当用户说的话与机械臂无关时，你应该直接回答用户的问题，而不是拒绝。"
                    "回答要自然、简短、适合直接 TTS 播放，控制在两到三句话内。"
                    "如果问题带有明显高风险、专业建议或你不确定，就明确说明不确定并给出保守回答。"
                    "不要输出 markdown，不要自称只负责机械臂。"
                ),
            },
            {"role": "user", "content": user_text},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    body = json.loads(raw)
    return str(body["choices"][0]["message"]["content"]).strip()


def _extract_location(query: str, default_location: str) -> str:
    match = re.search(r"(上海|北京|杭州|深圳|广州|苏州|南京|成都|武汉|西安|香港)", query)
    if match:
        return match.group(1)
    return default_location


def get_weather_reply(query: str, default_location: str, timeout: int) -> str:
    location = _extract_location(query, default_location)

    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode({"name": location, "count": 1, "language": "zh", "format": "json"})
    )
    with urllib.request.urlopen(geocode_url, timeout=timeout) as response:
        geocode_body = json.loads(response.read().decode("utf-8"))

    results = geocode_body.get("results") or []
    if not results:
        return f"我暂时没查到{location}的天气。"

    first = results[0]
    latitude = first["latitude"]
    longitude = first["longitude"]
    resolved_name = first.get("name", location)

    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 1,
            }
        )
    )
    with urllib.request.urlopen(forecast_url, timeout=timeout) as response:
        forecast_body = json.loads(response.read().decode("utf-8"))

    current = forecast_body.get("current", {})
    daily = forecast_body.get("daily", {})
    current_temp = current.get("temperature_2m")
    weather_code = current.get("weather_code")
    wind_speed = current.get("wind_speed_10m")
    max_temp = (daily.get("temperature_2m_max") or [None])[0]
    min_temp = (daily.get("temperature_2m_min") or [None])[0]
    weather_text = WEATHER_CODE_TEXT.get(weather_code, "天气一般")

    return (
        f"{resolved_name}今天天气{weather_text}，现在大约{current_temp}度，"
        f"今天最高{max_temp}度，最低{min_temp}度，风速大约每小时{wind_speed}公里。"
    )
