# agentic_rl

Agentic RL(GRPO/RLVR) 실행 경험 축적 트랙. vault `/mnt/c/ObsidianVault/Agentic_RL_Exec/`가
기획·상태 관리(항상 그 폴더의 STATE.md부터 확인), 이 리포는 실행 코드·로그.

## 목표

verl(verl-project/verl) 기반 GRPO를 실제로 돌려서 실행 경험 확보. 처음부터 agentic
(tool-use, 멀티턴)이 아니라 RLVR 단일턴 파이프라인(GSM8K)부터 — 계획은 vault
[[Agentic_RL_Exec-STATE]] 참고.

## 구조

- `verl/` — verl-project/verl clone(git 관리 대상 아님, `.gitignore`). 필요시
  `git clone --depth 1 https://github.com/verl-project/verl.git verl`로 재생성.
- `experiments/gsm8k_grpo_0.5b/` — GSM8K + Qwen2.5-0.5B-Instruct 단일GPU GRPO 실험.
- `ledger/*.jsonl` — 실행 기록(commit, cmd, GPU/sm_xx, driver, lib버전, seed, result).
  러너가 자동 append하는 정본, 자동 로드 없음(필요할 때 grep).
- `JOURNAL.md` — 실패 진단 수기 기록. 결론 한 줄 + ledger/커밋 링크.

## 환경

Colab 세션(`colab-cli`)에서 실행. Python 3.13 환경엔 verl 자체가 `<3.13` 요구라
설치 불가 — `uv run --frozen --extra vllm --extra fsdp`(committed `uv.lock`)로
맞는 Python까지 함께 해결. 자세한 함정은 `JOURNAL.md`.
