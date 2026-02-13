> **개발: 한국지질자원연구원(KIGAM) 지질자원AI융합연구실 권지회**
>
> **저작권: © 2025 한국지질자원연구원(Korea Institute of Geoscience and Mineral Resources, KIGAM). All rights reserved.**
>
> **본 소프트웨어 및 관련 문서의 저작권은 한국지질자원연구원에 있습니다. 출처를 명시하지 않은 무단 복제, 배포, 수정 및 상업적 이용은 저작권법에 의해 금지되어 있으며, 위반 시 민·형사상 법적 책임을 질 수 있습니다.**
>
> **본 소프트웨어를 연구 또는 교육 목적으로 사용하고자 하는 경우, 반드시 아래와 같이 출처를 명시하여야 합니다:**
>
> **한국지질자원연구원 지질자원AI융합연구실, "Landslide SPH Simulation", 2025.**
>
> **문의: 한국지질자원연구원 지질자원AI융합연구실 권지회 (jihoek@kigam.re.kr)**

---

# Landslide SPH Simulation

Non-Newtonian rheology와 bed entrainment를 적용한 GPU 가속 depth-averaged SPH 산사태 시뮬레이션입니다.

## 개요

본 프로젝트는 서울특별시 강남구 일원동 대모산의 실제 DEM 데이터를 활용하여 산사태 및 토석류 동역학을 시뮬레이션합니다. 구현 내용:

- **Depth-averaged SPH** 정식화를 통한 계산 효율성 확보
- **Non-Newtonian rheology** (Voellmy + Bingham)
- **Takahashi (2007) entrainment model**을 통한 bed erosion/deposition

CuPy 기반 GPU 병렬 연산으로 수천 개 입자의 효율적 계산이 가능합니다.

---

## 시뮬레이션 결과 예시

### 3D 애니메이션

![Landslide 3D Animation](guryoung_dem_10m/landslide_animation.gif)

*구룡 지형에서의 60초 토석류 시뮬레이션. 색상은 입자 속도(m/s)를 나타냄.*

### 주요 지표

| 지표 | 값 |
|--------|-------|
| **최대 속도** | 9.87 m/s (35.5 km/h) |
| **Runout distance** | 362.8 m |
| **토사 농도** | 35% |
| **충격 압력** | 97.4 kPa |
| **시뮬레이션 시간** | 60초 |

### 위험도 분석 (AI 생성)

#### 1. 위험도 평가 요약

본 시뮬레이션 분석 결과, 구룡산 일대에서 발생 가능한 토석류는 **매우 높은 위험도**를 나타내는 것으로 평가됩니다. 토석류의 최대 유속은 9.87m/s(35.5km/h)로 고속 자동차 수준에 달하며, 이는 위험등급 기준상 '고위험' 단계에 해당합니다. 토사 농도는 35%로 강력한 파괴력을 가진 전형적인 토석류 특성을 보이며, 이로 인한 추정 충격압력은 97.4kPa(974톤/㎡)에 이릅니다.

이러한 충격압력은 철근콘크리트 구조물의 손상 기준(100kPa)에 근접한 수준으로, 일반 목조건물 붕괴 기준(20kPa)의 약 5배에 달합니다.

#### 2. 거리별 피해 예상

| 도달거리 | 도달시간 | Head 속도 | 충격압력 | 예상 피해 |
|---------|---------|----------|---------|----------|
| **100m** | 10.7초 | 5.9 m/s | 35 kPa | 목조건물 완전 붕괴, 차량 전복 |
| **200m** | 27.5초 | 5.9 m/s | 35 kPa | 목조건물 붕괴, RC건물 저층부 손상 |
| **300m** | 49.2초 | 6.1 m/s | 37 kPa | 목조건물 붕괴, 지하주차장 매몰 |
| **362.8m** | 60초 | 6.7 m/s | - | 최대 도달 지점, 토사 퇴적 |

#### 3. 대피 권고사항

- **1차 위험구역 (Red Zone)**: 발생지점에서 반경 400m 이내 - 즉시 대피 필요
- **2차 경계구역 (Orange Zone)**: 반경 400-500m - 예방적 대피 권고
- **골든타임**: 10분 이내 (100m 지점 도달 시간 기준)
- **대피 방향**: 토석류 흐름 방향과 수직인 좌우측 고지대

#### 4. 재해 예방 권고사항

**즉시 조치:**
1. 24시간 감시체계 구축 (CCTV, 현장 관측소)
2. 주민 대피 훈련 실시 (분기별 1회 이상)
3. 비상방송 시설 정비
4. 응급의료체계 준비
5. 비상용품 비축 (3일분)

**중장기 대책:**
1. 사방댐 건설
2. 배수로 정비
3. 건축물 내진·내충격 보강
4. 대피소 건설 (지상 3층 이상)
5. 비상 우회도로 확보

---

## 1. Depth-Averaged Model

### 1.1 Governing Equations

본 모델은 3D Navier-Stokes 방정식을 유동 깊이에 대해 적분한 depth-averaged (shallow water) 방정식을 사용합니다. 기본 가정:
- Hydrostatic pressure 분포
- Depth-averaged velocity profile
- 수평 길이 스케일 >> 수직 길이 스케일

**질량 보존 (Continuity)**:

$$\frac{\partial h}{\partial t} + \nabla \cdot (h \bar{v}) = E - D$$

여기서:
- $h$ = 유동 깊이 [m]
- $\bar{v}$ = depth-averaged velocity [m/s]
- $E$ = entrainment rate [m/s]
- $D$ = deposition rate [m/s]

**운동량 보존 (Momentum)**:

$$\frac{\partial (h\bar{v})}{\partial t} + \nabla \cdot (h\bar{v} \otimes \bar{v}) = -gh\nabla(h + z_b) - \frac{\tau_b}{\rho} + S$$

여기서:
- $z_b$ = bed elevation [m]
- $\tau_b$ = basal shear stress [Pa]
- $S$ = source terms (viscosity 등)

### 1.2 SPH Discretization

SPH는 Lagrangian 입자법이므로, 입자를 따라가며 material derivative를 직접 계산합니다. 따라서 Eulerian 형태의 **convective term** $\nabla \cdot (h\bar{v} \otimes \bar{v})$ 이 별도로 등장하지 않고, 입자 advection 자체로 처리됩니다.

#### 1.2.1 연속 방정식 (Continuity)

$$\frac{dh_i}{dt} = \sum_j \Omega_j \, (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla W_{ij}$$

- $\Omega_j = \frac{m_j}{\rho_0 \, h_j}$: 입자 $j$의 면적 (depth-averaged SPH에서 질량/면밀도)
- $W_{ij} = W(|\mathbf{r}_i - \mathbf{r}_j|, h_{sml})$: cubic spline 커널
- $h_{sml}$: SPH smoothing length (유동 깊이 $h$와 별개)

#### 1.2.2 운동량 방정식 (Momentum)

입자 $i$의 가속도는 다음 항들의 합으로 구성됩니다:

$$\frac{d\mathbf{v}_i}{dt} = \mathbf{a}_i^{grav} + \mathbf{a}_i^{press} + \mathbf{a}_i^{art.visc} + \mathbf{a}_i^{phys.visc} + \mathbf{a}_i^{friction} + \mathbf{a}_i^{yield}$$

**① 중력 (경사면 구동력)**

$$\mathbf{a}_i^{grav} = -g \, \nabla z_b \Big|_i$$

지형 경사 $\nabla z_b$를 DEM에서 유한 차분으로 계산합니다.

**② 압력 구배력 (정수압)**

정수압: $P = \frac{1}{2} \rho_0 g h^2$

$$\mathbf{a}_i^{press} = -g \sum_j \Omega_j \, \frac{h_i + h_j}{2} \, \nabla W_{ij}$$

대칭 평균 $(h_i + h_j)/2$를 사용하여 운동량 보존을 보장합니다.

**③ Artificial Viscosity (Monaghan 1992)**

수치적 충격파 처리 및 안정성을 위한 항입니다. 접근하는 입자 쌍($\mathbf{v}_{ij} \cdot \mathbf{r}_{ij} < 0$)에만 적용합니다:

$$\mu_{ij} = \frac{h_{sml} \, (\mathbf{v}_{ij} \cdot \mathbf{r}_{ij})}{|\mathbf{r}_{ij}|^2 + \eta^2}, \quad \eta^2 = 0.01 \, h_{sml}^2$$

$$\Pi_{ij} = \frac{-\alpha \, c_0 \, \mu_{ij} + \beta \, \mu_{ij}^2}{\bar{\rho}_{ij}}$$

$$\mathbf{a}_i^{art.visc} = -\sum_j m_j \, \Pi_{ij} \, \nabla W_{ij} \quad (\mathbf{v}_{ij} \cdot \mathbf{r}_{ij} < 0 \text{ 인 경우만})$$

- $\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j$, $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$
- $\bar{\rho}_{ij} = \frac{1}{2}(\rho_i + \rho_j)$, 여기서 $\rho = \rho_0 h$ (면밀도)
- $\alpha, \beta$: artificial viscosity 계수 (기본값 0.1)
- $c_0$: 수치적 음속

**④ Physical Viscosity (Morris et al. 1997)**

동적 점성 계수 $\mu$에 의한 내부 전단 저항입니다:

$$\mathbf{a}_i^{phys.visc} = \sum_j m_j \, \frac{2\mu \, |\mathbf{r}_{ij}|}{\rho_i \, \rho_j \, (|\mathbf{r}_{ij}|^2 + \eta^2)} \, \frac{\nabla W_{ij} \cdot \mathbf{r}_{ij}}{|\mathbf{r}_{ij}|} \, \mathbf{v}_{ij}$$

- 방사대칭 커널에서 $\frac{\nabla W_{ij} \cdot \mathbf{r}_{ij}}{|\mathbf{r}_{ij}|} = \frac{\partial W}{\partial r}$ 이므로, 실질적으로 커널 기울기의 크기에 비례
- Artificial viscosity와 달리, 모든 이웃 입자 쌍에 적용 (접근/이격 무관)
- 각 속도 성분($v_x, v_y$)이 독립적으로 확산

**⑤ Basal Friction (Voellmy)**

$$\mathbf{a}_i^{friction} = -g \left( \mu_b + \frac{|\mathbf{v}_i|^2}{\xi} \right) \frac{\mathbf{v}_i}{|\mathbf{v}_i|}$$

- Coulomb 항 ($\mu_b$): 저속에서 지배적, 바닥면과의 마찰
- Turbulent 항 ($v^2/\xi$): 고속에서 지배적, 내부 충돌·난류 소산

**⑥ Yield Stress (Bingham)**

저속 ($|\mathbf{v}| < 1$ m/s) 영역에서만 활성화되는 정지 조건:

$$\mathbf{a}_i^{yield} = -\frac{\tau_y}{\rho_0 \, h_i} \, \frac{\mathbf{v}_i}{|\mathbf{v}_i|} \quad (|\mathbf{v}_i| < 1 \text{ m/s 인 경우만})$$

- $\tau_y / (\rho_0 h)$: yield stress를 depth-averaged 단위 질량당 힘으로 환산
- 구동 응력이 $\tau_y$ 이하이면 유동이 정지

#### 1.2.3 시간 적분 (Leapfrog)

Half-step leapfrog 방식으로 적분합니다:

$$h_i^{n+1/2} = h_i^n + \frac{\Delta t}{2} \left.\frac{dh}{dt}\right|^n$$

$$\mathbf{v}_i^{n+1/2} = \mathbf{v}_i^n + \frac{\Delta t}{2} \, \mathbf{a}_i^n$$

$$\mathbf{r}_i^{n+1} = \mathbf{r}_i^n + \Delta t \, \mathbf{v}_i^{n+1/2}$$

$$h_i^{n+1} = h_i^{n+1/2} + \frac{\Delta t}{2} \left.\frac{dh}{dt}\right|^{n+1}$$

$$\mathbf{v}_i^{n+1} = \mathbf{v}_i^{n+1/2} + \frac{\Delta t}{2} \, \mathbf{a}_i^{n+1}$$

#### 1.2.4 Entrainment 갱신

시간 적분 후, Takahashi (2007) 모델에 의한 침식/퇴적이 $h$와 $C$를 갱신합니다 (Section 3 참조).

### 1.3 Continuum ↔ SPH Form 대응 관계

| Continuum (Eulerian) | SPH (Lagrangian) | 비고 |
|---|---|---|
| $\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{v})$ | $\frac{dh_i}{dt} = \sum_j \Omega_j (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla W_{ij}$ | convective term이 입자 이동으로 흡수 |
| $\nabla \cdot (h\mathbf{v} \otimes \mathbf{v})$ | (입자 advection으로 자동 처리) | Lagrangian 좌표계의 장점 |
| $-gh\nabla(h + z_b)$ | $\mathbf{a}^{grav} + \mathbf{a}^{press}$ | 경사 구동력 + 정수압 구배 |
| $-\frac{\tau_b}{\rho}$ | $\mathbf{a}^{friction} + \mathbf{a}^{yield}$ | Voellmy + Bingham |
| $\nu \nabla^2 \mathbf{v}$ | $\mathbf{a}^{phys.visc}$ | Morris (1997) 근사 |
| (수치 안정성) | $\mathbf{a}^{art.visc}$ | Monaghan (1992) |

### 1.4 참고문헌 - Depth-Averaged SPH

- **Savage & Hutter (1989)**. The motion of a finite mass of granular material down a rough incline. *J. Fluid Mech.*, 199, 177-215.
- **Pastor, M., et al. (2009)**. Application of a SPH depth-integrated model to landslide run-out analysis. *Géotechnique*, 59(1), 45-57.
- **McDougall, S. & Hungr, O. (2004)**. A model for the analysis of rapid landslide motion across three-dimensional terrain. *Can. Geotech. J.*, 41, 1084-1097.
- **Monaghan, J.J. (1992)**. Smoothed particle hydrodynamics. *Annu. Rev. Astron. Astrophys.*, 30, 543-574.
- **Morris, J.P., Fox, P.J. & Zhu, Y. (1997)**. Modeling low Reynolds number incompressible flows using SPH. *J. Comput. Phys.*, 136, 214-226.

---

## 2. Non-Newtonian Rheology

토석류는 non-Newtonian 거동을 보입니다 - 전단 응력과 변형률 사이의 관계가 비선형입니다. 본 시뮬레이션은 두 가지 rheology 모델을 결합하여 사용합니다.

### 2.1 Voellmy Friction Model

Voellmy 모델 (1955)은 Coulomb friction과 속도 의존 turbulent 저항을 결합합니다:

$$\tau_b = \mu_b \sigma_n + \frac{\rho g v^2}{\xi}$$

마찰 계수 형태로 표현하면:

$$f = \mu_b + \frac{v^2}{\xi}$$

| 매개변수 | 기호 | 값 | 설명 |
|-----------|--------|-------|-------------|
| Coulomb friction | μ_b | 0.1 | basal friction coefficient (~6°) |
| Turbulent coefficient | ξ | 200 m/s² | 속도 제곱 저항 |

**물리적 해석**:

1. **Coulomb 항 (μ_b·σ_n)**:
   - 저속에서 지배적
   - 유동체와 바닥면 사이의 마찰 접촉을 나타냄
   - 건조 입상체 마찰과 유사

2. **Turbulent 항 (ρg·v²/ξ)**:
   - 고속에서 지배적
   - 에너지 소산 요인:
     - 입자 간 내부 충돌
     - 난류 혼합 및 와류 생성
   - 개수로 수리학의 Chézy 공식에서 유도

**한계**: Voellmy 모델은 현상론적(경험적) 모델입니다. 계수 ξ는 관측 사례의 back-analysis를 통해 보정해야 합니다. 일반적인 값:
- 설사태 (snow avalanche): ξ = 1000-2000 m/s²
- 토석류 (debris flow): ξ = 100-500 m/s²
- 암석 사태 (rock avalanche): ξ = 500-1000 m/s²

#### μ_b vs 정적 마찰각

Basal friction coefficient $\mu_b$와 마찰각 $\varphi$의 관계:

$$\mu_b = \tan(\varphi)$$

| μ_b | 등가 마찰각 |
|-----|------------------|
| 0.1 | 5.7° |
| 0.2 | 11.3° |
| 0.3 | 16.7° |
| 0.5 | 26.6° |
| 0.7 | 35.0° |

**왜 μ_b = 0.1 (5.7°)을 사용하는가? (일반적인 정적 마찰각 30-40°와 비교)**

건조 재료의 정적 마찰각:
- 건조 모래: 30-35°
- 자갈: 35-45°
- 토사/토양: 25-40°

그러나 토석류 유동 중의 **유효 마찰**은 다음 요인들로 인해 훨씬 낮습니다:

| 요인 | 효과 |
|--------|--------|
| **간극수압 (pore water pressure)** | 물이 입자 간 접촉력을 감소시킴 → 마찰↓ |
| **유동화 (fluidization)** | 고속 유동으로 입자가 부유 상태가 됨 |
| **윤활 (lubrication)** | 물 + 세립자가 윤활막을 형성 |

**핵심 구분**:

| 매개변수 | 값 | 의미 | 사용처 |
|-----------|-------|---------|---------|
| $\varphi_{bed}$ | 35° | 정지 상태의 재료 물성 | Entrainment model |
| $\mu_b$ | 0.1 | 유동 중 유효 저항 | Voellmy model |

**μ_b 문헌값** (Hungr, 1995):

| 재료 | μ_b | 등가 마찰각 |
|----------|-----|------------------|
| 포화 토석류 (saturated debris flow) | 0.05-0.15 | 3-9° |
| 암석 사태 (rock avalanche) | 0.1-0.3 | 6-17° |
| 건조 암석 활동 (dry rockslide) | 0.4-0.6 | 22-31° |

### 2.2 Bingham Viscoplastic Model

Bingham 모델은 yield stress를 가진 유체를 기술합니다 - 가해진 응력이 임계값을 초과할 때만 유동이 발생합니다:

$$\tau = \tau_y + \mu \dot{\gamma} \quad \text{if } \tau > \tau_y$$

$$\dot{\gamma} = 0 \quad \text{if } \tau \leq \tau_y$$

| 매개변수 | 기호 | 값 | 설명 |
|-----------|--------|-------|-------------|
| Yield stress | τ_y | 500 Pa | 유동 개시 최소 응력 |
| Plastic viscosity | μ | 100 Pa·s | 점성 저항 |

**물리적 해석**:

- **Yield stress (τ_y)**: 토석류의 내부 구조(입자 접촉, 점토 매트릭스)를 나타냅니다. 중력 응력이 이 임계값을 초과할 때만 유동이 시작됩니다.

- **Viscosity (μ)**: 유동 개시 후 변형 속도를 지배합니다. 점성이 높을수록 → 느리고 균일한 유동이 됩니다.

**정지 조건**: 구동 응력이 yield stress 아래로 떨어지면 유동이 정지합니다:

$$\tau_{driving} = \rho g h \sin\theta$$

$$\tau_{driving} < \tau_y \Rightarrow \text{유동 정지}$$

**문헌 대표값**:
| 재료 | τ_y (Pa) | μ (Pa·s) |
|----------|----------|----------|
| 이류 (mudflow) | 10-100 | 1-10 |
| 토석류 (debris flow) | 100-1000 | 10-100 |
| 고농도류 (hyperconcentrated flow) | 1000-10000 | 100-1000 |

### 2.3 결합 구현

본 모델의 전체 basal resistance:

**Voellmy friction**:
$$a_{friction} = -g \left( \mu_b + \frac{v^2}{\xi} \right) \frac{\mathbf{v}}{|\mathbf{v}|}$$

**Bingham yield** (저속 $|v| < 1$ m/s 에서 적용):
$$a_{yield} = -\frac{\tau_y}{\rho_0 h} \frac{\mathbf{v}}{|\mathbf{v}|}$$

Bingham yield 항은 저속에서만 정지 조건으로 적용됩니다.

### 2.4 참고문헌 - Rheology

- **Voellmy, A. (1955)**. Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung*, 73, 159-162.
- **Bingham, E.C. (1922)**. *Fluidity and Plasticity*. McGraw-Hill, New York.
- **Hungr, O. (1995)**. A model for the runout analysis of rapid flow slides, debris flows, and avalanches. *Can. Geotech. J.*, 32, 610-623.
- **Iverson, R.M. (1997)**. The physics of debris flows. *Rev. Geophys.*, 35(3), 245-296.
- **Coussot, P. & Meunier, M. (1996)**. Recognition, classification and mechanical description of debris flows. *Earth-Sci. Rev.*, 40, 209-227.

---

## 3. Entrainment Model (Takahashi 2007)

토석류는 바닥 재료의 entrainment(침식)으로 크게 성장하거나, deposition(퇴적)으로 체적이 감소할 수 있습니다. 본 시뮬레이션은 Takahashi equilibrium concentration 모델을 구현합니다.

### 3.1 Equilibrium Concentration

Takahashi (1991, 2007)는 토석류가 수로 경사에 의존하는 평형 토사 농도를 향해 수렴한다고 제안했습니다:

$$C_{eq} = \frac{\rho_w \tan\theta}{(\rho_s - \rho_w)(\tan\varphi - \tan\theta)}$$

| 매개변수 | 기호 | 값 | 설명 |
|-----------|--------|-------|-------------|
| 물 밀도 | ρ_w | 1000 kg/m³ | 간극 유체 |
| 고체 입자 밀도 | ρ_s | 2650 kg/m³ | 토사 입자 |
| Bed friction angle | φ | 35° | 바닥 재료의 내부 마찰각 |
| Slope angle | θ | 가변 | 국부 바닥 경사 |
| Max packing | C_max | 0.65 | 최대 고체 농도 |

**물리적 의미**:
- 급경사 → 높은 $C_{eq}$ (더 많은 토사를 수송 가능)
- $\theta \to \varphi$일 때: $C_{eq} \to \infty$ (파괴 임계 경사)
- $\theta \to 0$일 때: $C_{eq} \to 0$ (평탄면, 수송 능력 없음)

### 3.2 Erosion Rate

유동 농도가 평형 이하일 때 ($C < C_{eq}$), 침식이 발생합니다:

$$i_e = \delta_e \cdot C_{eq} \cdot v$$

| 매개변수 | 기호 | 값 | 설명 |
|-----------|--------|-------|-------------|
| Erosion coefficient | δ_e | 0.0007 | 경험적 (0.0001-0.01) |

**물리적 해석**:
- 침식률은 equilibrium concentration(수송 능력)에 비례
- 침식률은 속도(바닥면 전단 응력)에 비례
- 바닥면이 $i_e$ 속도로 저하되며, 바닥 재료($C_{bed}$ 농도)가 유동체에 유입

### 3.3 Deposition Rate

유동 농도가 평형을 초과할 때 ($C > C_{eq}$), 퇴적이 발생합니다:

$$i_d = \delta_d \cdot C \cdot v \cdot \left(1 - \frac{\tan\theta}{\tan\varphi}\right)$$

| 매개변수 | 기호 | 값 | 설명 |
|-----------|--------|-------|-------------|
| Deposition coefficient | δ_d | 0.01 | 경험적 (0.01-0.05) |

$\left(1 - \frac{\tan\theta}{\tan\varphi}\right)$ **의 물리적 해석**:
- $\theta \to \varphi$ (급경사): factor → 0, **퇴적 없음** (경사가 너무 급해 토사가 안정적으로 유지 불가)
- $\theta \to 0$ (평탄): factor → 1, **최대 퇴적**
- 퇴적된 재료가 경사면에서 안정적이어야 한다는 물리적 현실을 반영

### 3.4 Mass Conservation

**유동 깊이 변화**:

$$\frac{dh}{dt} = i_e - i_d$$

(침식은 $h$를 증가, 퇴적은 $h$를 감소시킴)

**고체 질량 보존**:

$$\frac{d(hC)}{dt} = C_{bed} \cdot i_e - C_{bed} \cdot i_d$$

여기서 $C_{bed} = C_{max} = 0.65$ (바닥 충전 농도).

**농도 변화**:
- 침식: 바닥 재료 ($C_{bed} = 0.65$)가 유입 → 유동 농도 증가
- 퇴적: 재료가 $C_{bed}$ 농도로 침전 → 잔류 유동이 희석

### 3.5 Erosion vs Deposition: $C_{init}$의 역할

**핵심**: 침식 또는 퇴적의 발생 여부는 현재 농도 ($C$)와 국부 경사에서의 equilibrium concentration ($C_{eq}$) 간의 관계에 의해 결정됩니다.

#### 경사에 따른 $C_{eq}$

| 경사 ($\theta$) | $\tan\theta$ | $C_{eq}$ ($\varphi$=35°) | $C_{init}$=0.4일 때 거동 |
|-----------|--------|--------------|------------------------|
| 5° | 0.087 | 0.087 | $C > C_{eq}$ → **퇴적** |
| 10° | 0.176 | 0.204 | $C > C_{eq}$ → **퇴적** |
| 14° | 0.249 | 0.335 | $C > C_{eq}$ → **퇴적** |
| 20° | 0.364 | 0.656 | $C < C_{eq}$ → **침식** |
| 25° | 0.466 | 1.21 | $C < C_{eq}$ → **침식** |
| 30° | 0.577 | 2.85 | $C < C_{eq}$ → **침식** |

**주요 관찰**:
- $\varphi_{bed} = 35°$에서, $C_{eq} = 0.4$이 되는 **전이 경사**는 약 $\theta \approx 20°$
- 경사 < 20°: $C_{init}(0.4) > C_{eq}$ → 순 퇴적
- 경사 > 20°: $C_{init}(0.4) < C_{eq}$ → 순 침식

#### 실용적 함의

1. **지형이 대부분 완경사 (< 20°)인 경우**:
   - C_init = 0.4에서 **순 퇴적**이 나타남
   - 침식을 관찰하려면 C_init을 낮춰야 함 (예: 0.2-0.3)

2. **지형이 급경사 (> 20°)인 경우**:
   - C_init = 0.4에서 **순 침식**이 나타남
   - 유동체가 바닥 재료를 취입하며 성장

3. **매개변수 민감도**:
   - $\varphi_{bed}$ 증가 → $C_{eq}$ 증가 → 침식 증가
   - $C_{init}$ 감소 → $C < C_{eq}$인 경우가 많아짐 → 침식 증가
   - $\delta_e / \delta_d$ 비율 증가 → 퇴적 대비 침식 속도 증가

#### 권장 C_init 값

| 유동 유형 | C_init 범위 | 일반적 거동 |
|-----------|--------------|------------------|
| 희석 고농도류 (dilute hyperconcentrated) | 0.2-0.3 | 침식 지배 (bulking) |
| 일반 토석류 (typical debris flow) | 0.4-0.5 | 침식/퇴적 혼합 |
| 고밀도 입상류 (dense granular flow) | 0.5-0.6 | 퇴적 지배 |

#### 예시: 순 퇴적이 발생하는 이유

다음 조건의 시뮬레이션에서:
- 평균 지형 경사: 14°
- $C_{init}$: 0.4
- $\varphi_{bed}$: 35°

14° 경사에서:

$$C_{eq} = \frac{1000 \times \tan(14°)}{1650 \times (\tan(35°) - \tan(14°))} \approx 0.34$$

$C_{init}$ (0.4) > $C_{eq}$ (0.34)이므로, **유동체가 재료를 퇴적**시켜 농도를 평형 쪽으로 감소시킵니다.

### 3.6 구현 참고

```python
# Erosion (C < C_eq)
erosion_rate = delta_e * C_eq * speed
dh = +erosion_rate * dt
d(h*C) = +C_bed * erosion_rate * dt

# Deposition (C > C_eq)
slope_factor = max(1 - tan(theta)/tan(phi), 0)
deposition_rate = delta_d * C * speed * slope_factor
dh = -deposition_rate * dt
d(h*C) = -C_bed * deposition_rate * dt
```

### 3.7 참고문헌 - Entrainment

- **Takahashi, T. (1991)**. *Debris Flow*. IAHR Monograph, Balkema, Rotterdam.
- **Takahashi, T. (2007)**. *Debris Flow: Mechanics, Prediction and Countermeasures*. Taylor & Francis, London. (Primary reference)
- **Egashira, S., et al. (1997)**. Mechanism of debris flow deposition and characteristics of debris flow deposits. *J. Japan Soc. Eng. Geol.*, 38, 149-155.
- **Hungr, O., et al. (2005)**. A review of the classification of landslides of the flow type. *Environ. Eng. Geosci.*, 11(3), 167-194.

---

## 4. 모델 매개변수 요약

### 4.1 물리 매개변수

| 분류 | 매개변수 | 기호 | 값 | 참고문헌 |
|----------|-----------|--------|-------|-----------|
| **밀도** | 기준 밀도 | ρ₀ | 2000 kg/m³ | - |
| | 고체 입자 | ρ_s | 2650 kg/m³ | 일반 광물 |
| | 물 | ρ_w | 1000 kg/m³ | - |
| **Voellmy** | Basal friction | μ_b | 0.1 | Hungr (1995) |
| | Turbulent coeff. | ξ | 200 m/s² | 보정값 |
| **Bingham** | Yield stress | τ_y | 500 Pa | Coussot (1996) |
| | Viscosity | μ | 100 Pa·s | Coussot (1996) |
| **Entrainment** | Erosion coeff. | δ_e | 0.0007 | Takahashi (2007) |
| | Deposition coeff. | δ_d | 0.01 | Takahashi (2007) |
| | Bed friction angle | φ | 35° | 일반 사질/자갈 |
| | 초기 농도 | C_init | 0.4 | - |
| | Max packing | C_max | 0.65 | Random close packing |

### 4.2 수치 매개변수

| 매개변수 | 값 | 설명 |
|-----------|-------|-------------|
| Smoothing length (h) | 2.5 m | SPH 커널 크기 |
| Particle spacing | 2.5 m | 초기 입자 간격 |
| Time step (dt) | 0.005 s | 적분 시간 단계 |
| Cutoff distance | 5.0 m | 이웃 탐색 범위 (2h) |
| Velocity ceiling | 30 m/s | 수치 안정성용 속도 상한 |

---

## 5. 파일 구조

```
landslide/
├── landslide_sph_gpu.py      # GPU SPH 시뮬레이터 핵심 모듈
├── run_simulation.py         # 시뮬레이션 실행 스크립트
├── visualize_results.py      # 결과 시각화 및 애니메이션
│
├── irwon_terrain_hires.npy   # DEM 지형 배열
├── simulation_results.npz    # 시뮬레이션 출력 데이터
├── satellite_texture.png     # 시각화용 위성 텍스처
│
├── irwon_landslide_3d.gif    # 3D 애니메이션 출력
├── entrainment_2panel.png    # Entrainment 분석 플롯
└── runout_distance.png       # Runout distance 플롯
```

---

## 6. 사용법

### 시뮬레이션 실행
```bash
python run_simulation.py
```

### 결과 시각화
```bash
python visualize_results.py
```

---

## 7. 출력 데이터

시뮬레이션 결과는 `simulation_results.npz`에 저장됩니다:

| 키 | 형태 | 설명 |
|-----|-------|-------------|
| `times` | (N_frames,) | 시간 배열 [s] |
| `x`, `y` | (N_frames, N_particles) | 위치 [m] |
| `vx`, `vy` | (N_frames, N_particles) | 속도 [m/s] |
| `height` | (N_frames, N_particles) | 유동 깊이 [m] |
| `concentration` | (N_frames, N_particles) | 고체 농도 [-] |
| `density` | (N_frames, N_particles) | $\rho = \rho_0 h$ [kg/m²] |
| `pressure` | (N_frames, N_particles) | $P = \frac{1}{2}\rho_0 g h^2$ [Pa·m] |
| `terrain` | (ny, nx) | 바닥 표고 [m] |

---

## 8. 참고문헌 (전체)

### Depth-Averaged Flow
1. Savage, S.B. & Hutter, K. (1989). The motion of a finite mass of granular material down a rough incline. *J. Fluid Mech.*, 199, 177-215.
2. Pastor, M., et al. (2009). Application of a SPH depth-integrated model to landslide run-out analysis. *Géotechnique*, 59(1), 45-57.
3. McDougall, S. & Hungr, O. (2004). A model for the analysis of rapid landslide motion across three-dimensional terrain. *Can. Geotech. J.*, 41, 1084-1097.

### Rheology
4. Voellmy, A. (1955). Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung*, 73, 159-162.
5. Bingham, E.C. (1922). *Fluidity and Plasticity*. McGraw-Hill, New York.
6. Hungr, O. (1995). A model for the runout analysis of rapid flow slides, debris flows, and avalanches. *Can. Geotech. J.*, 32, 610-623.
7. Iverson, R.M. (1997). The physics of debris flows. *Rev. Geophys.*, 35(3), 245-296.
8. Coussot, P. & Meunier, M. (1996). Recognition, classification and mechanical description of debris flows. *Earth-Sci. Rev.*, 40, 209-227.

### Entrainment
9. Takahashi, T. (1991). *Debris Flow*. IAHR Monograph, Balkema, Rotterdam.
10. Takahashi, T. (2007). *Debris Flow: Mechanics, Prediction and Countermeasures*. Taylor & Francis, London.
11. Egashira, S., et al. (1997). Mechanism of debris flow deposition and characteristics of debris flow deposits. *J. Japan Soc. Eng. Geol.*, 38, 149-155.

### SPH Method
12. Bui, H.H., et al. (2008). Lagrangian meshfree particles method (SPH) for large deformation and failure flows of geomaterial. *Int. J. Numer. Anal. Meth. Geomech.*, 32, 1537-1570.
13. Monaghan, J.J. (1992). Smoothed particle hydrodynamics. *Annu. Rev. Astron. Astrophys.*, 30, 543-574.

---

## 9. 연구 지역

- **위치**: 대모산, 서울특별시 강남구 일원동
- **좌표계**: TM (EPSG:5186) - Korea 2000 / Central Belt
- **원점**: X=203461, Y=535418
- **셀 크기**: 30 m

---

## 10. 의존성

```bash
pip install numpy cupy-cuda12x matplotlib pillow scipy
```

- Python 3.8+
- NumPy
- CuPy (CUDA GPU 필요)
- Matplotlib
- Pillow
- SciPy
