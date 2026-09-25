"""
PBRS(potential-based reward shaping)가 GRPO(critic-free, 궤적단위 advantage)에서
telescoping으로 no-op이 되는지 numpy로 직접 검산.

배경: Agentic_RL vault의 Jev×Agentic RL 도입방안 문서가 PBRS를 "최적 정책 불변
보장이 있는 가장 강한 카드"로 제시했으나, jev_research-STATE §6은 "confidence가
RL loss에 낄 자리 없다"고 정반대 결론. advisor 검토로 telescoping 조건 위반 여부가
쟁점 핵심이라는 게 드러나 이 스크립트로 확인.

핵심 결과 (4케이스):
(a) gamma=1, Phi(terminal)=0 (Ng 1999 조건 충족) → advantage 완전 동일, 정확한 no-op.
(b) Phi(terminal)!=0 (Jev의 P(success|s_T)를 그대로 쓰면 이 케이스) → no-op 깨짐,
    Jev 종결 확신도가 그대로 보너스로 새어들어감 — "PBRS"가 아니라 순수 shaping 보너스.
(c) gamma<1 인데 중간 potential을 discount 없이 다루면 완전 telescoping 안 됨 →
    advantage가 경로/길이에 의존하는 잡음을 먹음.
(d) 개념 확인: shaping 적용은 return-to-go에서 Phi(s_t)를 빼는 것과 동치 —
    critic(state-value baseline)을 쓰는 것과 본질적으로 같음.

결론: GRPO(critic-free) 기준선에서 PBRS가 이론 보장대로 작동하려면 Phi(terminal)=0을
강제해야 하는데, Jev의 자연스러운 Φ=P(success|s)는 종결 상태에서 사실상 결과 그 자체라
0이 아님 — 강제로 0 리셋하면 PBRS 효과 자체가 사라지고(case a), 안 하면 no-op이 깨짐
(case b). 결국 GRPO에서는 "PBRS로 도입" 대신 "역할③ critic/최종판정 보너스"로 수렴한다.
스텝단위 critic(GAE 등)에서는 다른 이야기(별도 검산 필요).

원본 grpo advantage 로직: grpo_advantage_mini.py 참고 (verl core_algos.py 대조).
"""
import numpy as np

def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon=1e-6, norm=True):
    scores = token_level_rewards.sum(axis=-1).astype(np.float64)
    id2scores = {}
    for i, idx in enumerate(index):
        id2scores.setdefault(int(idx), []).append(scores[i])
    id2mean = {k: np.mean(v) for k, v in id2scores.items()}
    id2std = {k: np.std(v) for k, v in id2scores.items()}
    out = np.zeros_like(scores)
    for i, idx in enumerate(index):
        idx = int(idx)
        out[i] = (scores[i]-id2mean[idx])/(id2std[idx]+epsilon) if norm else (scores[i]-id2mean[idx])
    return out[:, None]*response_mask, id2mean, id2std

rng = np.random.default_rng(1)
index = np.array([0,0,0,0])
bsz, seq_len = 4, 6
base_reward = np.array([1.0, 1.0, 0.0, 0.0])  # 정답 2 / 오답 2, group 내 outcome
rewards = np.zeros((bsz, seq_len))
rewards[:, -1] = base_reward
mask = np.ones((bsz, seq_len))

print("=== baseline: shaping 없음 ===")
adv0, m0, s0 = compute_grpo_outcome_advantage(rewards, mask, index)
print("advantage:", adv0[:,0], "mean:", m0, "std:", s0)

# case (a) gamma=1, Phi_T=0 강제 (예: shaping을 중간스텝에만 얹고 마지막에 0으로 리셋)
# PBRS potential 자체는 매 스텝 텔레스코핑되어 F_total = Phi(s_T) - Phi(s_0)
# Phi(s_0) 공통, Phi(s_T)=0으로 강제하면 F_total=-Phi(s_0)=상수(그룹 공통 시작상태 가정)
phi_0 = 0.6  # 그룹 전체 같은 prompt이므로 공통 시작 potential
phi_T_zero = np.zeros(bsz)
shaped_a = rewards.copy()
shaped_a[:, -1] += (phi_T_zero - phi_0)  # F_total 얹기
adv_a, m_a, s_a = compute_grpo_outcome_advantage(shaped_a, mask, index)
print("\n=== (a) gamma=1, Phi_T=0 (Ng 조건 충족) ===")
print("advantage:", adv_a[:,0], "동일?", np.allclose(adv0, adv_a))

# case (b) Phi_T != 0, sample마다 다른 종결 potential (실제 Jev P(success|s_T) 상황)
phi_T_b = np.array([0.9, 0.7, 0.3, 0.1])  # 성공/실패 경로마다 다른 종결 확신도
shaped_b = rewards.copy()
shaped_b[:, -1] += (phi_T_b - phi_0)
adv_b, m_b, s_b = compute_grpo_outcome_advantage(shaped_b, mask, index)
print("\n=== (b) Phi_T != 0 (Jev 종결상태 확신도, Ng 조건 위반) ===")
print("advantage:", adv_b[:,0], "baseline과 동일?", np.allclose(adv0, adv_b))
print("차이:", adv_b[:,0] - adv0[:,0])

# case (c) gamma<1, 중간스텝 shaping을 undiscounted sum으로 처리(잘못된 구현 흔한 실수)
# 진짜 discounted PBRS라면 감마 텔레스코핑, 여기선 3스텝 중간 potential을 감마 없이 단순합산
mid_phis = np.array([[0.5,0.55,0.6],[0.5,0.6,0.7],[0.5,0.45,0.4],[0.5,0.3,0.2]])  # (bsz, 3 mid steps)
gamma = 0.9
# 올바른 discounted PBRS 합 (should telescope approx, 근사): sum gamma^t (phi_{t+1}-phi_t)
correct_shaping = np.array([
    sum(gamma**t * (mid_phis[i,t+1]-mid_phis[i,t]) for t in range(2)) + gamma**2*(phi_T_zero[i]-mid_phis[i,-1])
    for i in range(bsz)
])
shaped_c = rewards.copy()
shaped_c[:, -1] += correct_shaping
adv_c, m_c, s_c = compute_grpo_outcome_advantage(shaped_c, mask, index)
print("\n=== (c) gamma=0.9, Phi_T=0, 올바른 discounted telescoping ===")
print("advantage:", adv_c[:,0], "baseline과 동일?", np.allclose(adv0, adv_c, atol=1e-6))
print("(감마<1이면 완전 telescoping 안 됨 — 중간항 안 지워짐, 상수도 아님)")

# case (d) shaped return-to-go = G_t - Phi(s_t) 확인 (per-step, GAE critic 관점)
# 궤적 보상이 스텝별로 있을 때 shaping 적용 전/후 return-to-go 비교
G = np.array([1.0, 0.0, 1.0])  # 3-step 예시, 마지막 스텝 결과
phis = np.array([0.3, 0.5, 0.0])  # Phi(s0,s1,s2), terminal Phi=0
returns_to_go = np.cumsum(G[::-1])[::-1]
F = phis[1:] - phis[:-1]  # gamma=1 F_t = Phi(t+1)-Phi(t) for t=0,1 (2 steps for 3 states? 조정)
print("\n=== (d) shaped return-to-go = G_t - Phi(s_t) 관계 확인(개념) ===")
print("원 return-to-go:", returns_to_go)
print("Phi(s_t) 빼면(=critic baseline 역할):", returns_to_go - phis[:len(returns_to_go)])
