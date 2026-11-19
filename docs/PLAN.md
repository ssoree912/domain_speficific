# dLLM을 합성 데이터 생성기로: 실험 계획 (개정 2, 2026-10-09)

이전 계획(dLLM을 학생으로 학습)에서 바뀐 점: **bidirectional은 데이터를 만드는 쪽에서만 쓴다.**
생성기만 바꾸고 학생·라벨러·벤치·학습 설정은 PANGEA(NeurIPS 2025)와 같게 둔다.

개정 2: 생성 방식(AR / dLLM) × anchor 유무의 2×2 설계로 바꿨다(G1' 추가). 학생 기준선 S0/S1을 직접 잰다.
PANGEA 표 번호를 proceedings 본문 기준으로 고쳤다. Writer 구간 길이는 G2 시험 생성으로 정한다.

## 가설

- **H1 (핵심, 상호작용).** 같은 anchor를 줄 때 dLLM이 AR보다 더 큰 이득을 본다:
  (G3 − G2) − (G1' − G1) > 0. AR은 뒤쪽 anchor를 보지 못한 채 앞 구간을 채우고, dLLM은 처음부터 모든 anchor를
  보고 채운다는 차이만 남긴 비교다. 뒤쪽 구조가 중요한 경우(MedQA 질문문·보기 칸이 증례 뒤, FinQA 표가 서술 뒤)에
  차이가 커야 한다.
- **H2.** G3가 G2보다 높다(dLLM 안에서 anchor의 효과).
- **H3.** G3가 같은 초기값의 AR 생성기 Qwen2.5-7B-Instruct(G1, G1')보다 높다.
- **H4.** G3 데이터는 형식 준수율이 높고, seed 내 다양성(cosine)이 PANGEA 수준을 유지한다.

## 조건 (2×2 + 기준선)

| | anchor 없음 | anchor 있음 |
|---|---|---|
| AR (Qwen2.5-7B-Instruct) | **G1** | **G1'**: 구간별 순차 생성. anchor k를 붙이고 구간 길이 상한 또는 줄바꿈까지 생성한 뒤 다음 anchor를 붙인다 |
| dLLM (Dream-v0-Instruct-7B) | **G2** | **G3 (Ours)**: 모든 anchor를 캔버스에 고정하고 infilling |

- G1' − G1: AR에서 형식 강제만의 효과
- G3 − G2: dLLM에서 anchor의 효과
- (G3 − G2) − (G1' − G1): bidirectional 효과 (핵심 주장의 직접 근거)

G1'과 G3는 같은 anchor 명세(텍스트, 순서, 구간 길이 상한)를 공유한다. Dream은 Qwen2.5-7B에서 출발했으므로
두 생성기의 초기값은 같다.

학생 기준선 (우리 파이프라인 안에서 직접 측정, 파서·라벨러가 PANGEA와 달라 논문 수치를 출발점으로 쓰지 않는다):

| 기준선 | 내용 |
|---|---|
| S0 | Llama-3.2-1B zero-shot (학습 없음) |
| S1 | seed 100개만으로 SFT (`data/seeds/<domain>_sft.jsonl`, 같은 학습 설정) |
| 참고 | PANGEA 논문 수치 (Llama3.3-70B 생성기, 아래) |

S0 실측 (2026-10-10, LLM 파서 기준, GSM8K / MedQA / FinQA): **5.38 / 19.80 / 4.62** (논문 pre-trained 5.69 / 28.91 / 6.02).
MedQA가 찍기 수준(25%)보다 낮아 파서가 보기를 못 뽑은 응답 비율을 확인해야 한다.

## 공통 설정 (모든 조건이 공유)

| 항목 | 값 | 비고 |
|---|---|---|
| D_g | CoT-Collection 1.84M에서 수학·의료·금융·코드를 LLM 분류로 제거 → **1,534,735개 (83.5%)** | Qwen2.5-7B-Instruct 분류기 (`scripts/filter_general.py`). PANGEA는 70B/72B 사용 |
| D_s | 도메인별 train split 무작위 100개 | `data/seeds/` |
| (d_g, seed) 짝 | 도메인·규모별로 한 번 뽑아 모든 조건이 공유 | `data/synth/plan/` |
| 프롬프트 | PANGEA 원문 (Prompt Writer / Projection / Annotation) | `pipeline/prompts/` |
| 라벨러 | **Qwen2.5-7B-Instruct** | 서버 한계로 70B 대신 사용. 모든 조건 공유라 비교는 공정, 절대 수치는 PANGEA보다 낮을 수 있음 |
| 학생 | Llama-3.2-1B (base) | full FT, Alpaca, lr 1e-5, batch 64, seq 4096, cosine, warmup 0.03, 3 epoch (`scripts/train_student.py`) |
| 평가 | PANGEA test 파일·Alpaca 프롬프트·0-shot greedy, LLM 파서로 정답 추출 | lm-eval (`hf`) + `scripts/pangea_parse.py`, 파서는 Qwen2.5-7B-Instruct |
| 규모 | 개발 10k, 최종 30k (도메인별) | PANGEA는 10k/30k/120k 보고 |
| 생성 디코딩 | Qwen(G1, G1'): PANGEA T 0.9 / top-p 0.7 / top-k 10. Dream(G2, G3): **T 0.2 / top-p 0.95** (2026-10-10 결정) | 아래 시험 생성 참고 |

PANGEA 참고값 (proceedings 본문 **Table 2**, Llama3.2-1B, 0-shot, GSM8K / MedQA / FinQA):
pre-trained 5.69 / 28.91 / 6.02, PANGEA 10k 32.52 / 37.78 / 36.44, 30k 38.36 / 39.98 / 41.41.
(별도 보충자료 PDF에서는 같은 표가 Table 5로 번호가 다르다. 본문 Table 5는 한국 수능 실험이다.)

## anchor 설계 (G1'과 G3 공유)

Writer와 Projection 모두 PANGEA 프롬프트를 조건으로 주고, 응답에 anchor를 고정한다.
G2는 같은 프롬프트에 anchor 없는 캔버스(같은 길이 L)를 쓴다.

구간 길이 상한 ℓ:
- **Prompt Writer**: seed에는 τ가 없으므로 seed 통계를 쓸 수 없다. **G2 시험 생성(도메인당 ~200개) 출력의 필드별 길이 통계**로
  정한다. G1 출력을 쓰면 AR 분포에 맞춰지므로 G2를 쓴다.
- **Projection**: seed 질문의 해당 구간 토큰 수 통계로 정한다.

**Prompt Writer (τ)**: PANGEA 출력 형식의 머리말을 anchor로 고정.
- MedQA: `Clinical Building Blocks:\n• Demographics – `[M] `\n• Timeline – `[M] … `\n• Pathophysiologic principles – `[M]
- GSM8K: `Quantitative Building Blocks:\n- Scenario - `[M] `\n- Symbols - `[M] `\n- Equations - `[M] `\n- Target - `[M] `\n- Plan - `[M]
- FinQA: `Financial Scenario:\n`[M]

**Projection (q̃)**: seed 질문의 구조와 τ의 값을 anchor로 고정.
- MedQA (기본): `Question: `[증례 M] [seed 질문문의 머리, 예: `Which of the following is the most likely`] [M] `?\nChoose one of the following:\nA. `[M] `\nB. `[M] `\nC. `[M] `\nD. `[M]
- MedQA (변형 G3-opt): 질문문 머리를 anchor에서 빼고 보기 칸만 고정. 기본형은 seed 하나에서 나온 문제가 모두 같은
  질문 유형이 되어 다양성 지표가 낮게 나올 수 있으므로 함께 본다.
- GSM8K: τ의 Symbols에서 뽑은 숫자를 등장 순서대로 anchor로 두고 사이를 infilling: [M] `12` [M] `$3` [M] `5` [M] `?`
- FinQA: `Context:\n`[M] seed 표의 격자(행·열 수, `|` 구분자, 단위 행)를 anchor로 두고 칸은 [M], [후속 서술 M] `\nQuestion:\n`[M]

## 시험 생성 결과 (2026-10-09, Writer 단계, 도메인당 8–16개; 탐색 스크립트 scripts/probes/)

`data/synth/{g1,g2}/<domain>/n200/writer.jsonl`.

| 설정 | MedQA | GSM8K | FinQA |
|---|---|---|---|
| G1 Qwen, PANGEA 샘플링 | 채움. 다만 "Not applicable"가 잦음 | 채움 | 채움 |
| G2 Dream, PANGEA 샘플링 (T 0.9, top-p 0.7, top-k 10) | 프롬프트의 형식 예시(`– …`)를 그대로 복사 | 형식 예시(`<A = value unit - label>`)를 복사 | 빈 출력(전부 EOS) |
| G2 Dream, Dream 권장 디코딩 (T 0.2, top-p 0.95, steps = L) | 대부분 채움 (8개 중 1개 복사) | – | 대부분 채움 (8개 중 2개 빈 출력) |
| G3 Dream + anchor, PANGEA 샘플링 (구간 40토큰) | 복사 없이 채움. 반복·내용 없는 항목 나열이 있음 | – | – |

함의:
- **G2의 디코딩 설정을 정해야 한다.** PANGEA 샘플링을 그대로 쓰면 G2는 거의 무너지고, G3 > G2가
  "anchor가 나쁜 디코딩을 구해 준 것"으로 읽힌다. G2·G3 모두 같은 Dream 디코딩 설정을 쓰되,
  그 설정에서 G2가 제대로 동작해야 한다. 후보: Dream 권장값(T 0.2, top-p 0.95), 또는 온도 sweep(0.2/0.5/0.9)으로
  G2가 무너지지 않는 가장 높은 온도. 낮은 온도는 다양성을 줄이므로 H4와 함께 본다.
- G1'과 G1도 같은 원칙(Qwen은 PANGEA 샘플링)을 유지한다.

계산량 실측 (A100 40GB, Dream, Writer 프롬프트):
- L 512, steps 512, batch 8: 약 56초/개
- L 256, steps 64, batch 8: 약 5초/개 → 10k에 약 14시간 (단계·도메인당)
- batch 16 × L 256은 top-p 정렬에서 메모리 부족
- Qwen(G1) AR, batch 8: 약 2.4초/개

## 데이터 자체 지표

- 형식 준수율: 생성물이 seed 형식(보기 4개, 표 격자, 문장 수 등)을 지킨 비율
- 다양성: 같은 seed에서 나온 샘플끼리 평균 cosine 유사도 (all-mpnet-base-v2, PANGEA 본문 Fig. 6 / 부록 **Table 14**와 같은 방식)
- 라벨러가 풀 수 있는 비율: 라벨러 응답에서 정답이 파싱되는 비율, 두 번 샘플링했을 때 답이 일치하는 비율

## 리스크

- **계산량:** Dream은 KV 캐시 없이 매 스텝 전체 시퀀스를 forward한다. 위 실측대로 steps를 L/4로 줄여도
  10k × 2단계 × 3도메인 × 2조건(G2, G3)이면 약 170시간이다. steps와 규모(10k 우선)를 실측 품질과 함께 정한다.
- **G2 붕괴:** 위 시험 생성 참고. 디코딩 설정 결정 전에는 G2/G3 대량 생성을 하지 않는다.
- **7B 라벨러 노이즈:** 정답이 틀린 학습 데이터가 늘 수 있다. 모든 조건에 같은 영향이라 비교는 유지된다.
- **연결 문장 품질:** 구간 길이가 맞지 않으면 문장이 늘어지거나 반복된다(시험 생성에서 관찰). ℓ을 통계로 정하고 EOS 패딩을 허용한다.
- **MedQA 다양성:** 질문문 머리 anchor가 다양성을 낮출 수 있다 → G3-opt 변형.
- **선행 연구:** "dLLM을 합성 데이터 생성기로" 쓰는 연구가 있는지 확인이 필요하다.

## Writer anchor 시험 (2026-10-10, GSM8K, 같은 64개)

anchor 명세는 `scripts/probes/anchors_spec.py`(heads: 칸 이름만, 칸당 30토큰 / rich: PANGEA 형식의 고정 부분
`A = `, `(1)`–`(4)`, `T = `, `→ (k)`까지 anchor, 칸마다 길이 상한 다름). 지표는 `scripts/probes/writer_metrics*.py`.

| | G1 | G1' heads | G1' rich | G2 | G3 heads | G3 rich |
|---|---|---|---|---|---|---|
| 형식 예시 복사 | 0 | 0 | 0 | 54 | 0 | 0 |
| 칸끼리 같은 내용 | 1 | 2 | 0 | 0 | 26 | 0 |
| Target이 `T =` | 60 | 56 | (anchor) | – | 1 | (anchor) |
| Equations 안 반복 | 0 | – | 4 | – | – | 10 |
| Scenario에 계산식 | 0 | – | 0 | – | – | 10 |

- 칸 이름만 anchor로 주면 Dream은 칸끼리 내용을 복사한다. 형식의 고정 부분까지 anchor로 주면 사라진다.
- rich에서도 G3는 식 반복과 Scenario에 식을 쓰는 경우가 G1'보다 많다. 규칙 지표로는 여기까지만 보이고,
  실제 차이는 Projection 이후 문제와 학생 성능으로 봐야 한다.
- G1' 구현 메모: 줄 안에 이어지는 anchor는 모델이 다음 번호를 스스로 쓰기 시작하는 지점("→", "(2")에서 자른다.

## 결정 (2026-10-10): G1 대 G3만 진행

GSM8K 시험(같은 64개)에서 G2는 Projection 출력 59/64가 비었고, G1'은 span 상한에서 문장이 잘려 문제가 깨졌다.
사용자 결정으로 G1'·G2는 더 돌리지 않고 **G1(Qwen, PANGEA) 대 G3(Dream + anchor)** 로 비교한다.
이 비교는 모델(Qwen/Dream)과 생성 방식(AR/anchor infilling)이 함께 바뀐다는 점을 결과 해석에 적는다.
Dream 실측: G3 Writer(L 239) 약 18초/개, G3 Projection 약 6초/개.

## 결과: GSM8K 1k (2026-10-11, `scripts/run_gsm8k_1k.sh`)

학생 Llama-3.2-1B, PANGEA 평가(1,319문항, 0-shot, Qwen 파서), 학습 seed 1회.

| | GSM8K acc | 정규식 acc |
|---|---|---|
| S0 (학습 전) | 5.38 | 1.90 |
| G1 1k (Qwen) | **15.39** | 14.40 |
| G3 1k (Dream + anchor, T 0.2) | **11.68** | 10.99 |

학습 데이터 비교 (각 1,000개):

| | 질문 단어 수 중앙값 | 질문당 숫자 | 답변 단어 수 | 서로 다른 3-gram 비율 | 가장 흔한 시작 |
|---|---|---|---|---|---|
| G1 | 48 | 3 | 228 | 0.85 | "In a small" 26개 |
| G3 | 54 | 6 | 246 | **0.49** | **"A group of" 302개** |

- G3 데이터의 다양성이 크게 낮다(문제의 30%가 "A group of"로 시작, "총합" 묻는 문제가 354개 대 G1 224개).
  Dream을 T 0.2로 돌린 것이 주원인으로 추정된다. T 0.2는 anchor 없는 G2가 무너지지 않게 고른 값인데,
  anchor가 있으면 높은 온도에서도 형식이 무너지지 않았다(Writer 탐색, T 0.9).
- 다음: G3를 더 높은 온도(T 0.5 / 0.9)로 다시 생성해 다양성과 학생 성능을 본다.

## Writer 단계의 일반 예시 반영 (2026-10-11, GSM8K, 같은 64개)

`scripts/probes/grounding_metrics.py`. G3 T 0.9(Dream) 1k 재생성은 첫 64개에서 반영도가 더 낮아 중단했다.

| | Scenario가 일반 예시와 단어 공유 | 일반 예시에 없는 과일·쿠키 문제 | 식 반복 |
|---|---|---|---|
| G1 Qwen | 51 | 3 | 0 |
| G3 Dream T 0.2 | 21 | 18 | 10 |
| G3 Dream T 0.9 (다른 64개) | 10 | 14 | – |
| G3 LLaDA-8B-Instruct, block 32, T 0.9 / top-p 0.7 / top-k 10 | 25 | **41** | 17 |

dLLM 두 개 모두 Writer 단계(무관한 일반 예시를 수학 재료로 바꾸는 단계)에서 일반 예시를 대부분 무시하고
범용 수학 재료(과일 개수, 가격 합계)를 만든다. 다양성 저하와 G3 1k의 낮은 성능은 이 단계에서 온다.
LLaDA 생성기: `scripts/llada_chat.py` (semi-AR block, low-confidence remasking, anchor 지원), 약 30초/개.

## 설계 변경 (2026-10-11): Writer 공유, dLLM은 Projection에만

Writer는 모든 조건이 G1(Qwen)의 τ를 공유하고, Projection만 Qwen AR(G1) 대 Dream + anchor(G3)로 비교한다.
같은 τ, 문제를 쓰는 방식만 다르다. 실행: `CONDS=g3 TAG=qtau STAGES=projection GEN_ARGS="--writer-from g1
--projection-anchors rich --dream-sampling pangea" bash scripts/run_gsm8k_1k.sh` → `data/synth/g3_qtau/`.

Projection anchor "rich" (`scripts/anchors.py`): Scenario의 고유명사(없으면 일반 예시와 공유하는 단어),
Symbols의 값+단위("3000 km"), Target 설명구, "?". Dream T 0.9 / top-p 0.7 / top-k 10(G1과 같은 샘플링),
캔버스에서 "="가 든 토큰 금지(PANGEA 규칙 "중간 계산 노출 금지"; 없으면 τ의 `A = 100 people` 표기를 절반이 복사).

시험 64개 (같은 Qwen τ, `scripts/probes/question_metrics.py`):

| | 일반 예시와 단어 공유 | 과일·쿠키 | "A group of" 시작 | 3-gram 다양성 | 2–4문장 | τ 기호 표기 노출 |
|---|---|---|---|---|---|---|
| G1 Qwen | 47 | 2 | 0 | 0.97 | 60 | 0 |
| G3 이전 (Dream Writer + numbers anchor) | 23 | 18 | 15 | 0.80 | 51 | – |
| G3 qtau T 0.2 | 53 | 3 | 4 | 0.92 | 48 | 약 19 |
| G3 qtau T 0.9, "=" 금지 | 54 | 3 | 4 | 0.92 | 47 | 약 18 ("A, 100 people - pilots", "A is 20 minutes") |

남은 문제: τ의 기호 표기("A, 100 people - pilots") 노출이 약 28%. G1은 0.
