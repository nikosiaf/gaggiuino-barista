"""Shots shaped like the Gaggiuino API's replies. Values are tenths, as the machine sends them."""


def shot(shot_id=7, seconds=30, flow=2.0, profile="Stock - 9 Bar"):
    n = seconds * 5
    t = [i * 2 for i in range(n)]
    ramp = [min(1.0, i / 30) for i in range(n)]
    return {
        "id": shot_id,
        "duration": seconds * 10,
        "profile": {
            "name": profile,
            "waterTemperature": 93,
            "globalStopConditions": {"weight": 36},
            "phases": [{"name": "Main", "type": "PRESSURE", "target": {"start": 9, "end": 9, "curve": "INSTANT", "time": 0},
                        "stopConditions": {"time": seconds * 1000}, "skip": False}],
        },
        "datapoints": {
            "timeInShot": t,
            "pressure": [round(90 * r) for r in ramp],
            "targetPressure": [90] * n,
            "pumpFlow": [round(flow * 10)] * n,
            "targetPumpFlow": [round(flow * 10)] * n,
            "temperature": [930] * n,
            "targetTemperature": [930] * n,
            "shotWeight": [round(flow * 10 * i * 0.2) for i in range(n)],
            "weightFlow": [round(flow * 10)] * n,
        },
    }
