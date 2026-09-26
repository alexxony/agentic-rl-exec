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

## 2026-09-27 colab exec가 background 전환되면 원격 실행 미보장 (원인 정정)

**정정(같은 날 advisor 지적)**: 최초 기록에서 "`colab exec --timeout N`이 로컬
wrapper N초 넘으면 background 전환"이라 썼는데, 실제 원인은 **Claude Code Bash
도구 자체의 기본 120초 timeout**이었음 — `colab exec`에 넘긴 `--timeout 180`과는
무관, Bash 도구 호출에 `timeout` 파라미터를 명시 안 하면 120초에서 끊겨 background로
넘어감. `colab exec`의 `--timeout`은 원격 커널 실행 시간 제한이라 별개 축.

증상 자체(원격에서 스크립트가 실제로 끝까지 실행됐는지 별도 확인 필요)는 유효 —
`git clone`(verl) + `uv sync` 백그라운드 launch가 포함된 스크립트가 Bash 도구
timeout에 걸렸을 때, 재확인 결과 `/content/verl`도 `/tmp/gpu_setup.sh`도 존재하지
않아 원격 실행 자체가 시작도 안 된 상태였음(세션은 uptime 23분으로 정상 생존).
원문 로그(`bsh91r9ta.output`)를 그때 안 읽고 추정만 했던 것도 절차 결함.

**교훈**: (1) Bash 도구로 `colab exec`를 호출할 땐 도구의 `timeout` 파라미터를
`colab exec --timeout`보다 넉넉히 준다(예: 150000ms). (2) background 전환 알림
받으면 "이제 돈다"로 넘기지 말고 output 파일을 먼저 읽고, 다음 exec에서
`ls`/`ps`/로그 파일로 재확인. (3) 무거운 명령(clone, uv sync)은 ① 짧은 동기 호출로
먼저 끝내고 ② 진짜 오래 걸리는 것만 `nohup ... & disown`으로 별도 백그라운드
launch ③ 같은 exec 안에서 `ps aux`+로그 tail로 launch 성공 직접 확인.

**일반화된 CLI 도구 함정**은 [[Colab_Ops-MOC]]로 이관(로컬 timeout/background
오판, drivemount stdin 문제 등) — 이 프로젝트엔 이 사례만 남김.

## 2026-09-27 Drive 마운트 시도 실패, checkpoint 저장은 우회

장기 보관(checkpoint 등)용으로 `colab drivemount -s agentic-rl-gpu` 시도 →
브라우저 OAuth 동의 필요 URL 발급, 사용자가 동의했으나 CLI 쪽은 `Press Enter after
you have granted access...`에서 stdin 대기 중 background 전환되어 확인을 못 받음
→ 마운트 최종 실패, `/content/drive` 미생성. 상세 원인·해법은 [[Colab_Ops-MOC]]
(§drive.mount 헤드리스 무한대기)로 이관.

**이 트랙에서의 결론**: `SAVE_FREQ=-1`(체크포인트 미저장)로 첫 실행 진행, 결과는
로그만 `colab download`로 회수. Drive는 나중에 필요해지면 사용자가 직접
`! colab drivemount -s <session>`으로 실행(Enter 직접 입력 가능한 포그라운드).

## 2026-09-27 K8s 요구범위 재확인 (XiaomiMiMo/verl 조사, 착수 안 함)

지난 조사에서 "code/cyber arm은 Ray만 써서 K8s 불필요"라 추정했으나 env.example
원문 재확인 결과 틀림 — code(`scripts/code/env.example`의 "task pods" 섹션,
`KUBECONFIG`+`MAX_CONCURRENT_SESSIONS`가 "cluster quota"로 제한)와
cyber(`scripts/arvo/arvo.env.example`, `KUBECONFIG` "pod-create permission")
둘 다 K8s 필요. 최소 3/5 도메인이 단일세션(Colab/RunPod) 구조적으로 불가 —
이 리포는 지금 착수 대상 아님, vault [[xiaomi-mimo-verl-oss-2026-09-26]] 참고.
