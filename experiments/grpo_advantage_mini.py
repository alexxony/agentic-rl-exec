"""
verl compute_grpo_outcome_advantage 재현 미니 실험 (numpy, torch 불필요).

원본: volcengine/verl `verl/trainer/ppo/core_algos.py` (main 브랜치, 2026-09-16 fetch).
대조 원문은 vault Agentic_RL-02-오픈소스프로젝트.md "verl 실제 GRPO 코드 대조" 절 참고.

목적: 그룹 정규화 advantage 계산 + Dr.GRPO(std 나눗셈 생략) 옵션 + 토큰 broadcast
      구조를 가짜 보상값으로 직접 돌려서 숫자로 확인.
"""
import numpy as np


def compute_grpo_outcome_advantage(
    token_level_rewards: np.ndarray,  # (bsz, seq_len)
    response_mask: np.ndarray,        # (bsz, seq_len), 1=유효토큰 0=패딩
    index: np.ndarray,                # (bsz,) 같은 prompt에서 나온 sample끼리 같은 id
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # False면 Dr.GRPO 스타일(평균만 빼고 std 나눗셈 생략)
):
    scores = token_level_rewards.sum(axis=-1).astype(np.float64)  # outcome reward: 토큰합 → 스칼라

    id2scores: dict[int, list[float]] = {}
    for i, idx in enumerate(index):
        id2scores.setdefault(int(idx), []).append(scores[i])

    id2mean = {k: np.mean(v) for k, v in id2scores.items()}
    id2std = {k: np.std(v) for k, v in id2scores.items()}  # verl은 그룹 내 표준편차(ddof=0)

    out = np.zeros_like(scores)
    for i, idx in enumerate(index):
        idx = int(idx)
        if norm_adv_by_std_in_grpo:
            out[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)
        else:
            out[i] = scores[i] - id2mean[idx]

    # 스칼라 advantage를 response 전체 토큰에 broadcast (GAE처럼 토큰별로 다른 값 아님)
    advantages = out[:, None] * response_mask
    return advantages, advantages, id2mean, id2std


def main():
    rng = np.random.default_rng(0)

    # 가짜 셋업: prompt 2개(index 0, 1), 각 prompt당 group size 4 (verl group_size 관례)
    # 같은 prompt group 안에서 outcome reward만 다르게 줌 — RLVR처럼 0/1 정답보상 + 약간의 변주
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bsz, seq_len = 8, 6

    # outcome reward는 마지막 토큰에만 얹는 전형적 RLVR 패턴(중간 토큰 0)
    rewards = np.zeros((bsz, seq_len))
    # group 0: 정답 2개(reward 1), 오답 2개(reward 0) — 학습 신호 섞인 정상 케이스
    rewards[0, -1] = 1.0
    rewards[1, -1] = 1.0
    rewards[2, -1] = 0.0
    rewards[3, -1] = 0.0
    # group 1: 전부 정답(reward 1) — "전부 맞음" 케이스, std=0이라 epsilon 안전장치가 실제로 작동해야 함
    rewards[4, -1] = 1.0
    rewards[5, -1] = 1.0
    rewards[6, -1] = 1.0
    rewards[7, -1] = 1.0

    # response_mask: 응답 길이가 sample마다 다른 상황 재현(패딩 포함)
    response_mask = np.ones((bsz, seq_len))
    response_mask[2, 4:] = 0  # sample 2는 짧은 응답(뒤 2토큰 패딩)
    response_mask[6, 5:] = 0  # sample 6은 더 짧은 응답(뒤 1토큰 패딩)

    print("=== 1) 표준 GRPO (norm_adv_by_std_in_grpo=True) ===")
    adv, _, id2mean, id2std = compute_grpo_outcome_advantage(rewards, response_mask, index)
    print("group별 mean:", id2mean)
    print("group별 std:", id2std)
    print("sample별 advantage(첫 토큰만, broadcast 확인용):")
    for i in range(bsz):
        print(f"  sample {i} (group {index[i]}): adv={adv[i, 0]:.4f}, mask_sum={response_mask[i].sum():.0f}")
    print()
    print("확인 포인트:")
    print("- group 0(정답 2/오답 2, mean=0.5, std=0.5): 정답 adv≈+1.0, 오답 adv≈-1.0 (표준정규화)")
    print("- group 1(전부 정답, mean=1.0, std=0.0): (score-mean)=0이라 advantage도 깔끔히 0.0")
    print("  → std=0이면 0/epsilon=0이라 발산 안 함. 하지만 '전부 정답'은 애초에 advantage가 전부 0이라")
    print("    학습 신호가 완전히 사라지는 케이스 — 이게 논문의 '너무 쉬운/어려운 prompt는 신호 없음' 근거")
    print()

    print("=== 2) Dr.GRPO 옵션 (norm_adv_by_std_in_grpo=False, std 나눗셈 생략) ===")
    adv_dr, _, id2mean_dr, id2std_dr = compute_grpo_outcome_advantage(
        rewards, response_mask, index, norm_adv_by_std_in_grpo=False
    )
    for i in range(bsz):
        print(f"  sample {i} (group {index[i]}): adv={adv_dr[i, 0]:.4f}")
    print()
    print("확인 포인트: group 1(전부 정답)도 advantage가 전부 0.0으로 깔끔 — std 나눗셈이 없으니")
    print("epsilon 안전장치의 왜곡(작은 std로 나눠서 advantage가 과도하게 커지는 현상)이 애초에 없음.")
    print("Dr.GRPO 논문 주장(std 나눗셈이 길이 편향 만든다)과 별개로, 여기선 '쉬운 그룹 신호 소실'")
    print("문제도 std 나눗셈 생략 쪽이 더 안정적으로 보임 — 단, 이건 이 가짜 데이터 한정 관찰.")
    print()

    print("=== 3) broadcast 구조 확인 (sample 2: 패딩 2토큰) ===")
    print("sample 2 advantage 전체:", adv[2])
    print("→ 패딩 위치(mask=0)는 advantage도 0 — 토큰별로 다른 advantage가 아니라")
    print("  스칼라 하나를 유효 토큰 전체에 그대로 복사(mask로 패딩만 0 처리)하는 구조 실측 확인.")


if __name__ == "__main__":
    main()
