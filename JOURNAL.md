# JOURNAL

실패 진단 수기. 항목 = 결론 한 줄 + ledger/커밋 링크.

## 2026-09-27 GSM8K 전처리: verl 패키지 전체 설치 불가(Python 3.13 비호환)

Colab CPU세션 기본 Python 3.13.15, verl `pyproject.toml`은
`requires-python = ">=3.10,<3.13"` — `pip install -e verl` 자체가 즉시 거부됨
(단순 종속성 문제 아니라 버전 게이트).

`examples/data_preprocess/gsm8k.py`가 import하는 `verl.utils.hdfs_io`는
`--hdfs_dir` 안 쓰면 실사용 안 됨(파일 자체는 os/shutil만 쓰는 가벼운 모듈).
하지만 `verl/__init__.py`가 패키지 최상위에서 `from .protocol import DataProto`를
끌어와 `ray`까지 로드 — 서브모듈 하나만 쓰려 해도 무거운 종속 전체가 걸림.

**해법(전처리 단계 한정)**: `importlib.util.spec_from_file_location`으로
`hdfs_io.py`를 패키지 `__init__.py` 우회해서 직접 로드. `sys.modules`에 `verl`,
`verl.utils` stub 등록 후 `gsm8k.py` 본문을 `exec()`. verl 전체 설치도 `ray` 설치도
불필요 — `experiments/gsm8k_grpo_0.5b/preprocess_bypass_ray.py`.

**주의**: 이 우회는 전처리(단순 파일 I/O)에만 유효. 실제 GRPO 학습은 verl 전체
스택(ray, vllm, torch, flash-attn)이 필요해 우회 불가 — GPU 세션에서는 `uv run
--frozen --extra vllm --extra fsdp`로 `uv.lock`이 고정한 Python 버전까지 함께
해결(committed lockfile이 3.10~3.12 인터프리터를 가져옴).

## 2026-09-27 K8s 요구범위 재확인 (XiaomiMiMo/verl 조사, 착수 안 함)

지난 조사에서 "code/cyber arm은 Ray만 써서 K8s 불필요"라 추정했으나 env.example
원문 재확인 결과 틀림 — code(`scripts/code/env.example`의 "task pods" 섹션,
`KUBECONFIG`+`MAX_CONCURRENT_SESSIONS`가 "cluster quota"로 제한)와
cyber(`scripts/arvo/arvo.env.example`, `KUBECONFIG` "pod-create permission")
둘 다 K8s 필요. 최소 3/5 도메인이 단일세션(Colab/RunPod) 구조적으로 불가 —
이 리포는 지금 착수 대상 아님, vault [[xiaomi-mimo-verl-oss-2026-09-26]] 참고.
