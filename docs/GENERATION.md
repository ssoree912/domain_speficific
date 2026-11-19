# 합성 데이터 생성 방식: Writer, Projection, anchor (2026-10-11 기준)

현재 GSM8K 1k 실험(G1 대 G3)에서 데이터를 실제로 어떻게 만들고 있는지 정리한다.
실험 설계와 결과는 `docs/PLAN.md`, 코드는 `scripts/pangea_gen.py`(생성 단계), `scripts/anchors.py`(anchor),
`scripts/dream_chat.py`(Dream 디코딩)에 있다.

## 1. 전체 흐름

```
plan        일반 예시 d_g 1,000개를 뽑아 seed 문제 d_s와 무작위로 짝짓기 (모든 조건 공유)
writer      d_g → 문제 재료 τ                         Qwen2.5-7B-Instruct (두 조건 공유)
projection  τ + seed → 새 문제 q̃                      G1: Qwen AR / G3: Dream + anchor
annotation  q̃ → 정답·풀이                             Qwen2.5-7B-Instruct (두 조건 공유)
export      (q̃, 풀이) → 학생 학습 데이터 train.jsonl
```

| 단계 | G1 | G3 (현재, `g3_qtau`) |
|---|---|---|
| writer | Qwen, PANGEA 프롬프트 | **G1의 출력을 그대로 사용** (`--writer-from g1`) |
| projection | Qwen, PANGEA 프롬프트, AR | Dream-v0-Instruct-7B, 같은 PANGEA 프롬프트, anchor 캔버스 |
| annotation | Qwen | Qwen |
| 출력 위치 | `data/synth/g1/gsm8k/n1000/` | `data/synth/g3_qtau/gsm8k/n1000/` |

두 조건은 같은 (d_g, seed) 짝, 같은 τ, 같은 라벨러를 쓴다. 다른 것은 Projection에서 문제 문장을
쓰는 모델과 방식뿐이다.

실행 명령:
```
bash scripts/run_gsm8k_1k.sh                                   # G1 (CONDS 기본값에 포함)
CONDS=g3 TAG=qtau STAGES=projection \
  GEN_ARGS="--writer-from g1 --projection-anchors rich --dream-sampling pangea" \
  bash scripts/run_gsm8k_1k.sh                                 # G3
```

## 2. plan: 일반 예시와 seed 짝짓기

- **일반 예시 d_g:** `data/general/cot_collection/filtered.parquet`(CoT-Collection에서 수학·의료·금융·코드를 뺀
  1,534,735개)에서 1,000개를 무작위로 뽑는다(`random_state=0`). 문제 본문(`source` 열)만 쓴다.
- **seed:** GSM8K train에서 미리 뽑은 100개(`data/seeds/gsm8k.jsonl`) 중 하나를 각 d_g에 무작위로 붙인다.
- **출력:** `data/synth/plan/gsm8k_n1000.jsonl`. 한 줄에 `{uid, dg_id, general, seed_id, seed}`가 들어간다.

## 3. Writer: 문제 재료 τ 만들기 (Qwen, 두 조건 공유)

**입력 프롬프트:** `pipeline/prompts/gsm8k_writer.txt`(PANGEA 부록 원문)의 `{general}` 자리에 d_g를 넣는다.
프롬프트는 일반 예시에서 일상적인 상황과 숫자를 뽑아 아래 형식으로만 출력하라고 지시한다.
```
Quantitative Building Blocks:
- Scenario - <one kid-friendly clause>
- Symbols - <A = value unit - label>, <B = ...>, <C = ...>[, <D = ...>[, <E = ...>]]
- Equations - (1) ... = ... (2) ... = ... (3) ... = ... (4) ... = ... [(5) ... = ...]
- Target - <T = description of numeric goal>
- Plan - (1) ... → (2) ... → (3) ... → (4) ... [→ (5) ...]
```

**생성 설정** (`pangea_gen.py writer --cond g1`)
- Qwen 채팅 템플릿에 사용자 메시지로 넣는다.
- 어시스턴트 답변 앞부분을 `Quantitative Building Blocks:`로 미리 채운다(assistant prefix, PANGEA 공식 코드와 같은 방식).
  모델은 그 뒤를 이어 쓴다.
- 샘플링: PANGEA 설정 그대로 T 0.9, top-p 0.7, top-k 10, 최대 768토큰. AR 생성이고 batch 16이다.
- 저장되는 τ는 prefix 뒤에 생성된 부분이다.

**실제 예시** (`gsm8k-000001`)
```
d_g: This goose breeds north of the timberline in Greenland, Canada, Alaska, and the northeastern tip of
     Siberia, and spends winters in warm parts of North America ... They fly as far south as Texas and
     Mexico during winter ...

τ:   - Scenario - Snow geese fly from their breeding grounds to wintering grounds.
     - Symbols - A = 3000 km - distance, B = 1000 km - part of journey, C = 2000 km - remaining distance
     - Equations - (1) A = B + C (2) C = A - B (3) B = 1000 km (4) C = 2000 km
     - Target - T = 2000 km - remaining distance to wintering grounds
     - Plan - (1) Total distance = part of journey + remaining distance → ...
```

## 4. Projection: 새 문제 만들기

**프롬프트(두 조건 공통):** `pipeline/prompts/gsm8k_projection.txt`(PANGEA 원문)의 `{block}`에 τ를, `{seed}`에
seed 문제를 넣는다. seed는 "어조와 길이만 참고하고 내용은 베끼지 말 것"이라고 명시돼 있다. 주요 규칙은 다음과 같다.
- 재료의 상황·숫자·관계를 자연스러운 이야기로 녹일 것, 2–4문장
- 재료의 숫자를 모두 한 번씩 쓸 것, 새 숫자를 지어내지 말 것
- 중간 계산이나 결과를 노출하지 말 것, 문제 문장만 출력할 것

assistant prefix는 `Transformed Domain Question:`이다.

### 4.1 G1: Qwen AR

`pangea_gen.py projection --cond g1`: 위 프롬프트 + prefix 뒤를 Qwen이 왼쪽부터 이어 쓴다.
T 0.9, top-p 0.7, top-k 10, 최대 1024토큰. anchor는 없다.

```
G1 결과: Snow geese fly from their breeding grounds to wintering grounds, covering a total distance of
         3000 km. If they have already flown 1000 km, how much farther do they need to travel to reach
         their wintering grounds?
```

### 4.2 G3: Dream + anchor 캔버스

**(1) τ에서 anchor 뽑기** (`anchors.projection_plan(..., style="rich")`). 순서대로 다음과 같다.

| 종류 | 규칙 | 예시 (gsm8k-000001) |
|---|---|---|
| 개체 | Scenario의 고유명사. 문장 첫 단어 하나만 대문자인 경우는 제외. 없으면 Scenario에서 일반 예시와 겹치는 가장 긴 단어(5글자 이상) | ` geese` |
| 수량 | Symbols의 각 `X = 값 단위 - 설명`에서 "값 단위"(숫자 + 단위 최대 두 단어), 중복 제거, 최대 6개. **τ가 계산해서 얻는 값은 뺀다** (아래) | ` 1000 km` (`3000 km`, `2000 km`는 계산값으로 판정돼 빠짐) |
| 목표 구절 | Target의 `T = ...`에서 설명 부분(2–10단어). 숫자로 시작하거나 기호(=, +, 단일 대문자 변수)가 있거나 형식 예시 복사면 뺀다 | ` remaining distance to wintering grounds` |
| 끝 | `?` | `?` |

anchor 텍스트 앞에 공백을 붙여 문장 중간에 자연스럽게 이어지게 한다.
1,000개 기준 추출 비율: 개체 818개, 수량 anchor가 하나 이상 837개, 목표 구절 429개.

**계산값 제외** (`anchors.derived`, 2026-10-11 추가). anchor로 고정한 값은 반드시 문제에 나오므로, 답이나
중간 결과를 고정하면 문제에 답이 드러난다. 다음을 계산값으로 보고 수량 anchor에서 뺀다.
- Target의 값(`T = 2000 km - ...`의 2000)과 Target이 가리키는 변수(`I = extra time ...`의 I)
- 식으로 정의되는 변수: `C = A - B`의 C, `A + B + C + D = E`의 E
- 식의 결과로만 나오는 숫자: `B - A = 24 yards`의 24

모든 변수가 서로를 정의해 하나도 안 남는 경우에만, Target 값과 같은 것만 빼고 나머지는 둔다.
1,000개 τ에서 수량 2,704개 중 929개가 빠졌고, Target 값이 anchor로 남은 τ는 0개다(이전 214개).
규칙이 거칠어서 주어진 값도 함께 빠질 수 있다. 예를 들어 기러기 예시의 `A = B + C`는 A(3000 km, 실제로는
주어진 값)를 계산값으로 판정한다. 빠진 값은 anchor가 아닐 뿐이고, 모델은 프롬프트의 τ를 보고 쓸 수 있다.

**(2) 캔버스 구성** (`anchors.canvas`). plan은 `(anchor, 그 앞에 둘 빈칸 수)`의 목록이다.
- 빈칸 수는 `max(8, 64 // (anchor 수 + 1))`로 정한다. 64는 seed 문제 길이 중앙값 60토큰에 맞춘 값이다.
- 첫 anchor가 개체면 그 앞 빈칸은 4로 줄인다. 이야기 첫머리에 등장하도록 하기 위해서다.
- 목표 구절이 있으면 마지막 `?` 앞 빈칸은 6이다.

예시의 실제 캔버스(Dream 토큰 위치, 길이 L = 77; 계산값 제외 규칙을 넣기 전의 anchor다.
지금은 ` 3000 km`, ` 2000 km`가 빠진 `geese / 1000 km / remaining distance … / ?` 캔버스가 된다):
```
위치  0– 3   [M] × 4    빈칸 4
위치  4– 5   " geese"   (2토큰)
위치  6–15   [M] × 10   빈칸 10
위치 16–21   " 3000 km" (6토큰: Qwen 계열 토크나이저는 숫자를 한 자리씩 나눔)
위치 22–31   [M] × 10
위치 32–37   " 1000 km"
위치 38–47   [M] × 10
위치 48–53   " 2000 km"
위치 54–63   [M] × 10
위치 64–69   " remaining distance to wintering grounds"
위치 70–75   [M] × 6
위치 76      "?"
```
`{4: ' geese', 16: ' 3000 km', 32: ' 1000 km', 48: ' 2000 km', 64: ' remaining distance to wintering grounds', 76: '?'}`
가 anchor 위치이고, 나머지 [M]은 Dream이 채울 마스크다.

**(3) Dream 디코딩** (`DreamChat.chat`, Dream의 `diffusion_generate` 사용)
- 입력은 채팅 템플릿을 적용한 프롬프트 + `Transformed Domain Question:` + 길이 L의 마스크 캔버스다.
- 첫 스텝 전에 hook(`generation_tokens_hook_func`, step=None)으로 anchor 토큰을 해당 위치에 써 넣는다.
  Dream은 마스크 위치만 바꾸므로 anchor는 끝까지 고정된다.
- 스텝 수 = L(한 스텝에 한 토큰), 채우는 순서는 Dream 기본값 `entropy`(확신이 높은 위치부터)다.
  모든 anchor를 처음부터 보면서 사이를 채운다.
- 샘플링은 G1과 같은 T 0.9, top-p 0.7, top-k 10이다.
- **"=" 금지:** logits hook(`generation_logits_hook_func`)으로 "="가 들어간 모든 토큰의 확률을 0으로 만든다.
  근거는 PANGEA 규칙 "중간 계산 노출 금지"다. 금지하지 않으면 τ의 `A = 100 people` 표기를 문제 절반이 그대로 옮겼다.
  PANGEA에는 없는 G3만의 디코딩 제약이다.
- 남는 빈칸은 EOS로 채워질 수 있고, 디코딩할 때 EOS·pad·mask 토큰은 지운다.
  마지막에 `??`처럼 겹친 물음표는 하나로 합친다(`anchors.clean`).
- **배치:** Dream은 한 배치 안에서 마스크 수가 같아야 하므로 캔버스 길이 L이 같은 행끼리 묶는다. batch 4다.
  배치를 더 키우면 top-p 정렬에서 메모리가 부족했다.

```
G3 결과: A group of snow geese fly from their breeding grounds to wintering grounds, 3000 km apart. If the
         first part of their journey is 1000 km, and the remaining distance to wintering grounds is 2000 km
         after they reach the halfway point, what is the remaining distance to wintering grounds after they
         reach the halfway point?
```

## 5. Annotation과 export (두 조건 공유)

- Qwen에 `pipeline/prompts/gsm8k_annotation.txt`(PANGEA 원문)의 `{question}`으로 q̃를 넣는다.
- assistant prefix는 `- **Answer**: `이고, T 0.9 / top-p 0.7 / top-k 10, 최대 1536토큰이다.
- export는 `{instruction: q̃, output: 라벨러 응답}`을 `train.jsonl`로 쓴다. q̃나 응답이 비면 뺀다.
- 학생은 이 데이터를 Alpaca 템플릿으로 학습한다(`scripts/train_student.py`).

## 6. 알려진 문제 (G3)

1. **τ 기호 표기 노출 (64개 중 약 18개):** "="를 막아도 `there are A, 100 people - pilots`,
   `A is 20 minutes - time to walk`처럼 Symbols의 변수 이름과 설명이 문장에 남는다.
   수량 anchor가 τ의 해당 문자열과 너무 비슷해서 그 주변까지 같이 복사하는 것으로 보인다. G1은 0개다.
2. **답이나 중간 결과의 노출 (해결, 위의 계산값 제외):** 이전에는 Symbols의 계산 결과(`C = 2000 km`)까지 anchor로
   고정해서, 위 G3 예시처럼 "남은 거리가 2000 km"라고 말한 뒤 남은 거리를 묻는 문제가 나왔다.
   제외 규칙 적용 후 64개 시험에서 답 값이 문제에 나온 경우는 7개다(G1 4개). 대부분 τ 자체가
   `T = 1`처럼 값이 작고 모호한 경우이고, anchor가 아니라 모델이 프롬프트의 τ에서 가져온 것이다.
5. **τ 원문 출력:** anchor가 개체와 `?`뿐일 때 Dream이 문제 대신 τ 형식(`- Scenario - ... - Symbols - ...`)을
   그대로 쓰는 경우가 64개 중 2개 있었다.
3. **2–4문장 준수가 낮음:** 64개 중 47개(G1 60개). anchor 사이 빈칸이 고정 길이라 문장을 늘리는 경향이 있다.
4. **개체 대체 단어가 약함:** 고유명사가 없을 때 `getting`, `standard` 같은 단어가 개체 anchor가 되는 경우가 있다.

## 7. 이전 방식 (지금은 쓰지 않음)

| 방식 | 내용 | 버린 이유 |
|---|---|---|
| G2 | Dream, anchor 없음 | Writer에서 형식 예시를 그대로 복사(84%), Projection은 64개 중 59개 빈 출력 |
| G1' | Qwen에 anchor를 구간별로 끼우며 순차 생성 | 구간 상한에서 문장이 잘려 문제가 깨짐 |
| G3 Writer (heads) | Dream 캔버스에 칸 이름만 anchor | 다섯 칸에 같은 내용 반복 41% |
| G3 Writer (rich) | `A = `, `(1)`–`(4)`, `T = `, `→ (k)`까지 anchor, 칸별 길이 상한 | 형식은 맞지만 일반 예시 반영이 낮음(21/64, Qwen 51/64). 학생 11.68 대 G1 15.39 |
| G3 Writer, Dream T 0.9 | 위와 같은 anchor에 온도만 올림 | 반영도 10/64, "10, 20, 30" 같은 숫자 나열 증가 |
| G3 Writer, LLaDA | 블록 32 + T 0.9, 또는 블록 없음 + T 0 | 반영도 25/64, 22/64. 과일 문제나 숫자 나열로 쏠림 |
| G3 Projection (numbers) | Dream Writer의 τ에서 숫자만 anchor + `?` | τ 자체가 일반 예시와 끊겨 문제도 과일·합계 유형으로 쏠림 |
