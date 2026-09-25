"""
GAE(critic 기반) PBRS telescoping 검산 - pbrs_telescoping_check.py(GRPO)의 후속.

배경: GRPO(critic-free)에서는 PBRS shaping이 Phi(terminal)=0을 강제해야 no-op이 된다는
결론을 이미 검산함(pbrs_telescoping_check.py). 이 스크립트는 "critic이 있으면 다르지
않을까"라는 질문을 검산한다.

핵심 결과: **GAE도 GRPO와 같은 결론(Phi(terminal)=0 필요)으로 수렴한다** - 단, 다른
경로를 통해서다.

- Wiewiora(2003)의 shaping-equivalence 정리: PBRS는 critic을 Phi만큼 오프셋해서
  초기화(또는 재구성)하는 것과 정확히 동일하다. critic이 V'=V-Phi로 완전히 수렴하면
  매 스텝 delta가 shaping 없을 때와 정확히 같아진다 - 이 자체는 학습 중 내내 성립하는
  항등식이지, "수렴 후에만 성립"하는 게 아니다(초안 v1/v2에서 이 부분을 "학습 중엔
  이론보장 깨짐"이라 잘못 적었다가 advisor 지적으로 정정).
- **단, 이 항등식은 critic이 실제로 모든 state의 V를 -Phi만큼 옮길 수 있을 때만 성립한다.**
  표준 GAE 구현(verl 포함, 02번 노트의 compute_gae_advantage_return과 동일 관행)은
  에피소드 종결 시 bootstrap value를 0으로 마스킹한다 - 즉 critic이 "terminal state의
  V"를 애초에 학습하지 않는다. Phi(terminal)!=0(Jev의 P(success|s)를 그대로 쓰면 이 경우)
  이면 critic이 non-terminal state는 전부 오프셋할 수 있어도 terminal bootstrap 자리는
  손대지 못해 마지막 스텝에서만 정확히 gamma*Phi(terminal) 크기의 잔차가 남는다.
- 즉 GRPO(그룹정규화가 Phi(s_0) 공통부분만 지우고 Phi(terminal)은 못 지움)와 GAE(critic이
  non-terminal V는 흡수해도 masked terminal bootstrap은 못 흡수) 둘 다 서로 다른
  메커니즘으로 "Phi(terminal)=0 필요"라는 같은 결론에 도달한다.
- PBRS는 곧 critic을 Jev의 P(success|s)로 사전학습/오프셋하는 것과 동치 - 도입방안
  문서의 역할③(critic)과 실질적으로 같은 실험이 된다는 결론(GRPO 검산과 일치).

참고: Wiewiora, "Potential-Based Shaping and Q-Value Initialization are Equivalent",
JAIR 2003.
"""
import numpy as np


def deltas_masked_bootstrap(rewards, values_nonterm, gamma):
    """values_nonterm: V(s0..s_{T-1})만. 표준 GAE 관행대로 terminal bootstrap=0 마스킹."""
    values = np.concatenate([values_nonterm, [0.0]])
    return rewards + gamma * values[1:] - values[:-1]


def make_shaped_reward(rewards, Phi_nonterm, Phi_term, gamma):
    Phi_full = np.concatenate([Phi_nonterm, [Phi_term]])
    F = gamma * Phi_full[1:] - Phi_full[:-1]
    return rewards + F


def main():
    gamma = 0.99
    T = 5
    rewards = np.zeros(T)
    rewards[-1] = 1.0
    true_V_nonterm = np.array([0.3, 0.4, 0.5, 0.7, 0.9])
    Phi_nonterm = np.array([0.3, 0.4, 0.5, 0.7, 0.9])

    print("=== baseline: shaping 없음 ===")
    d0 = deltas_masked_bootstrap(rewards, true_V_nonterm, gamma)
    print("delta:", d0)

    print("\n=== case A: Phi(terminal)=0, critic이 V-Phi로 완전 오프셋 ===")
    shaped_A = make_shaped_reward(rewards, Phi_nonterm, 0.0, gamma)
    V_A = true_V_nonterm - Phi_nonterm
    dA = deltas_masked_bootstrap(shaped_A, V_A, gamma)
    print("delta:", dA, "| baseline과 동일?", np.allclose(d0, dA))
    assert np.allclose(d0, dA), "case A는 정확한 no-op이어야 함(Ng 1999 조건 충족)"

    print("\n=== case B: Phi(terminal)=0.6(Jev 실사용 형태), critic은 non-terminal만 오프셋 가능 ===")
    Phi_term_B = 0.6
    shaped_B = make_shaped_reward(rewards, Phi_nonterm, Phi_term_B, gamma)
    V_B = true_V_nonterm - Phi_nonterm  # terminal bootstrap은 표준관행상 0 고정, 손 못 댐
    dB = deltas_masked_bootstrap(shaped_B, V_B, gamma)
    residual = dB - d0
    print("delta:", dB)
    print("baseline과 동일?", np.allclose(d0, dB))
    print("잔차:", residual)
    assert np.allclose(residual[:-1], 0), "마지막 스텝 이전은 완전히 상쇄되어야 함"
    assert not np.isclose(residual[-1], 0), "마지막 스텝엔 Phi(terminal) 잔차가 남아야 함"
    print(f"마지막 스텝 잔차 크기 = {residual[-1]:.4f} (Phi(terminal)={Phi_term_B} 관련)")
    print("-> GRPO와 같은 결론: Phi(terminal)=0이 아니면 no-op 깨지고 종결 확신도가 보너스로 샘")


if __name__ == "__main__":
    main()
