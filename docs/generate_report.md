# generate_report.py

시뮬레이션 결과 분석 및 보고서 생성 모듈

## 개요

시뮬레이션 결과를 분석하고 Markdown/HTML 보고서를 생성합니다. 선택적으로 Anthropic Claude API를 사용한 AI 기반 재해 위험 분석을 포함할 수 있습니다.

## 사용법

```bash
python generate_report.py <results_npz_path> [options]

# 예시
python generate_report.py ./guryoung_dem_10m/simulation_results.npz
python generate_report.py ./guryoung_dem_10m/simulation_results.npz --no-llm
```

## 명령줄 옵션

| 옵션 | 설명 |
|------|------|
| `--no-llm` | LLM 분석 비활성화 |

## 입력

### 필수 파일

| 파일 | 설명 |
|------|------|
| `simulation_results.npz` | 시뮬레이션 결과 |

### 선택적 설정 파일

| 파일 | 설명 |
|------|------|
| `simulation_config.yaml` | 시뮬레이션 설정 (프로젝트명 등) |
| `report_config.yaml` | 보고서/LLM 설정 |

### LLM 설정 (`report_config.yaml`)

```yaml
llm:
  api_key: "sk-ant-api03-..."  # Anthropic API 키
  model: "claude-sonnet-4-20250514"  # 모델 선택
```

## 출력

| 파일 | 설명 |
|------|------|
| `analysis_report.md` | Markdown 보고서 |
| `analysis_report.html` | HTML 보고서 (시각화 포함) |

## 분석 항목

### 기본 통계 (`SimulationStats`)

```python
@dataclass
class SimulationStats:
    # 기본 정보
    n_frames: int           # 프레임 수
    duration: float         # 시뮬레이션 시간
    n_particles_initial: int
    n_particles_final: int

    # 속도 통계
    max_speed: float        # 최대 속도 (m/s)
    max_speed_time: float   # 최대 속도 도달 시간
    mean_speed_final: float

    # Runout 통계
    runout_head_final: float     # 최종 도달 거리 (m)
    runout_centroid_final: float
    runout_head_max: float

    # 침식 통계
    entrainment_enabled: bool
    concentration_initial: float
    concentration_final: float
```

### 이동거리별 속도 분석

Head 입자가 특정 거리에 도달했을 때의 속도/농도 분석

```python
velocity_by_distance = {
    'thresholds': [100, 200, 300, 400, 500, 600],
    'velocity_at_distance': {
        100: {
            'time': 2.5,
            'head_speed': 5.2,
            'front_mean_speed': 4.8,
            'overall_mean_speed': 3.2,
            'front_concentration': 0.42
        },
        # ...
    }
}
```

## 시각화 (HTML 보고서)

### 자동 생성 플롯

1. **초기 조건 플롯**: 파티클 위치, 지형 평면도, 고도 프로파일
2. **속도 변화 플롯**: 시간-속도 그래프, 가속도 분석
3. **Runout 플롯**: 도달거리 변화
4. **침식 분석 플롯**: 속도/거리, 경사/침식량/농도 2패널
5. **경사도 분석 플롯**: Head 경로 DEM 경사 분석
6. **파티클 분포 플롯**: 멀티프레임 분포 변화

## AI 분석 (Claude)

LLM 분석 활성화 시 다음 내용 자동 생성:

1. **위험도 평가 요약**
   - 유속/농도 기반 종합 위험 평가
   - 충격압력 기준 피해 예상

2. **주거지역 피해 예상**
   - 거리별 속도/농도 데이터 기반
   - 건물 유형별 피해 분석

3. **대피 권고사항**
   - 위험 반경 설정
   - 대피 시간 추정

4. **재해 예방 권고사항**
   - 즉시 조치 사항
   - 중장기 대책

## 주요 함수

### `analyze_results(data) -> SimulationStats`

기본 통계 분석

### `analyze_velocity_by_distance(data, distance_thresholds) -> dict`

이동거리별 속도 분석

### `get_llm_analysis(stats, data, config_info, velocity_by_distance) -> Optional[str]`

Claude API 호출, AI 분석 생성

### `generate_markdown_report(...) -> str`

Markdown 보고서 생성

### `generate_html_report(...) -> str`

HTML 보고서 생성 (시각화 포함)

### 플롯 생성 함수

- `create_initial_condition_plot(data, config_info) -> str`
- `create_velocity_evolution_plot(data) -> str`
- `create_runout_plot(data) -> str`
- `create_slope_runout_analysis_plot(data) -> str`
- `create_entrainment_plot(data) -> Optional[str]`
- `create_particle_distribution_plot(data) -> str`

## 의존성

- `numpy`
- `matplotlib`
- `pyyaml`
- `anthropic` (선택적, LLM 분석용)

## 예시

```python
from generate_report import generate_report

# 보고서 생성 (LLM 분석 포함)
report_path = generate_report(
    results_path='./simulation_results.npz',
    use_llm=True
)

# LLM 없이 생성
report_path = generate_report(
    results_path='./simulation_results.npz',
    use_llm=False
)
```

## HTML 보고서 구조

```
├── 헤더 (프로젝트명, 날짜, 신뢰도)
├── 핵심 지표 카드 (최대속도, 도달거리, 농도)
├── AI 재해 위험 분석 (LLM 활성화 시)
├── 시뮬레이션 결과
│   ├── 발생 위치 정보
│   └── 초기 조건 플롯
├── 유속 및 농도 변화
│   ├── 속도 변화 플롯
│   └── 침식 분석 플롯
├── 지형 경사도 분석
│   └── 경사도 분석 플롯
└── 푸터
```
