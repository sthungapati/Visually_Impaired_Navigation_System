# AI Navigation Assistant Prototype Guide

This prototype is designed for a final project presentation:
- polished visual overlays
- assistant status panel
- rule-based navigation guidance
- spoken assistance with anti-spam cooldown

## Modules

- `navigation/detector.py`  
  Loads YOLO and returns detections for each frame.

- `navigation/distance.py`  
  Converts bounding box size into relative distance buckets and approximate meters/feet.

- `navigation/assistant_logic.py`  
  Core decision layer:
  - class filtering for outdoor obstacles
  - region mapping (`left/center/right`)
  - danger scoring (0-100)
  - guidance text generation

- `navigation/presentation_ui.py`  
  Draws the center walking zone, per-object labels, and the right assistant panel.

- `presentation_demo.py`  
  End-to-end video runner with output writing and TTS cooldown control.

## Danger Scoring (Simple, Explainable)

Each detection receives a score from `0` to `100` based on:
- estimated distance safety bucket (`clear/watch/caution/stop`)
- region importance (`center` more dangerous than sides)
- model confidence

Danger level mapping:
- `LOW`: `< 40`
- `MEDIUM`: `40-64`
- `HIGH`: `65-84`
- `CRITICAL`: `>= 85`

## Threshold Tuning Recommendations

For safer behavior (more warnings):
- lower `--conf` to `0.20-0.25`
- increase center zone width with `--center-zone-ratio` to `0.38-0.45`
- reduce `--tts-cooldown` to `2.5-3.0`

For calmer behavior (fewer false alarms):
- raise `--conf` to `0.30-0.40`
- shrink center zone to `0.28-0.33`
- increase `--tts-cooldown` to `4.0-6.0`

For low-power demos:
- set `--frame-stride 2` or `3`
- optionally reduce input video resolution beforehand

## Outdoor Obstacle Class Suggestions

Recommended classes for outdoor walking:
- person
- bicycle
- motorcycle
- car
- bus
- truck
- traffic light
- stop sign
- bench
- dog
- fire hydrant
- pole
- tree
- trash can

## Future OpenAI Integration (Natural Assistant Voice)

To make the narration more human-like, add a language layer:
1. Keep current rule engine as the safety ground-truth.
2. Build a structured context object per frame-window:
   - top obstacle
   - region
   - danger level
   - recent guidance history
3. Send context to an OpenAI model to rewrite guidance in natural language while preserving intent.
4. Add guardrails:
   - never contradict safety rule
   - keep sentence short (`<= 14 words`) when danger is high
   - enforce deterministic fallback to rule text if API fails.

Suggested prompt strategy:
- system prompt: safety-first mobility assistant for visually impaired users
- user payload: compact JSON context (no raw frame)
- output format: one sentence + urgency tag
