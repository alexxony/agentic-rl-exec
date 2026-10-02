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

**일반화된 CLI 도구 함정**은 [[Cloud_GPU_Ops-MOC]]로 이관(로컬 timeout/background
오판, drivemount stdin 문제 등) — 이 프로젝트엔 이 사례만 남김.

## 2026-09-27 Drive 마운트 시도 실패, checkpoint 저장은 우회

장기 보관(checkpoint 등)용으로 `colab drivemount -s agentic-rl-gpu` 시도 →
브라우저 OAuth 동의 필요 URL 발급, 사용자가 동의했으나 CLI 쪽은 `Press Enter after
you have granted access...`에서 stdin 대기 중 background 전환되어 확인을 못 받음
→ 마운트 최종 실패, `/content/drive` 미생성. 상세 원인·해법은 [[Cloud_GPU_Ops-MOC]]
(§drive.mount 헤드리스 무한대기)로 이관.

**이 트랙에서의 결론**: `SAVE_FREQ=-1`(체크포인트 미저장)로 첫 실행 진행, 결과는
로그만 `colab download`로 회수. Drive는 나중에 필요해지면 사용자가 직접
`! colab drivemount -s <session>`으로 실행(Enter 직접 입력 가능한 포그라운드).

## 2026-09-27 tool_parser·response_length 미결 해소, 정정(1차 답 notebook 추정 오류)

**1차 답(오답, advisor가 잡음)**: notebook cell10의 `engine_kwargs.vllm.tool_call_parser=hermes`
/ `engine_kwargs.sglang.tool_call_parser=qwen25`만 보고 verl `tool_parser` 실측
완료라 STATE에 적음. 이건 vLLM/SGLang 엔진 자체 OpenAI 호환 서버의 파서 이름이고,
verl `ToolAgentLoop`이 실제로 쓰는 파서와는 다른 설정 키였음.

**정정(소스 확인)**: `verl/experimental/agent_loop/tool_agent_loop.py:120`
`self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, ...)`
— 선택 키는 `multi_turn.format`. 기본값은 `verl/workers/config/rollout.py:61`
`format: str = "hermes"`. Qwen2.5가 `<tool_call>{json}</tool_call>`(Hermes 스타일)로
tool call을 내므로 기본값 그대로 맞음, 오버라이드 불필요. notebook의 엔진단
`tool_call_parser`는 별개 레이어(엔진의 OpenAI 호환 서버가 자체 파싱할 때만 쓰임,
verl agent_loop 경로에선 영향 없음) — SGLang `qwen25`는 이 트랙(vLLM 사용)엔 무관.

**response_length 예산 확인**: `tool_agent_loop.py:290,394`에서 종료조건은
`len(response_mask) >= self.response_length` — tool 메시지 포함 전체 토큰 길이
기준. 하지만 `verl/utils/tokenizer/continuous_token.py:412-413`에서 context
kind(=tool 응답 병합)는 `aligned_mask += [0] * appended_token_count` — loss
mask는 0. 결론: tool 토큰이 length 예산은 소모하지만(assistant 턴 수 실질적으로
줄어듦) loss에는 기여 안 함 — "구조적으로 불리"라고 단정한 최초 표현은 과함,
정확히는 "같은 response_length 안에서 tool arm의 실제 답변 생성 여지가 tool
왕복 텍스트만큼 줄어든다"는 정도. `max_tool_response_length` 기본 256자,
truncate side `middle`(`rollout.yaml:218,221`) — tool 응답 자체도 무제한 아님.

**교훈**: notebook config override는 어느 레이어(엔진 vs 프레임워크 내부)를
겨냥한 것인지 먼저 확인 후 "실측 확정" 태그를 달 것 — config 존재 자체가
해당 레이어에서 쓰인다는 증거는 아님.

## 2026-09-27 control arm agent_loop 확정 + 10-step 파일럿 재사용 불가 + reward 설계 결정

**control arm agent_loop**: 전역 기본값 `default_agent_loop=single_turn_agent`
(`rollout.yaml:257`)이지만 이걸 no-tool arm에 쓰면 agent_loop 클래스 자체가
tool arm(`tool_agent`)과 달라져 state machine·턴 카운팅 로직까지 confound.
확정: control arm도 `tool_agent`(tools=[] 또는 tool_config_path=null)로 고정.

**10-step 파일럿 재사용 불가**: `experiments/gsm8k_grpo_0.5b/notebooks/gsm8k_grpo_pipeline_check.ipynb`
cell9가 `examples/grpo_trainer/run_qwen3_4b_fsdp.sh`를 호출하는데, 이 스크립트
자체에 `multi_turn`/`agent.default_agent_loop`/`return_raw_chat` 오버라이드가
전혀 없음(grep 확인) — 전역 기본값 `single_turn_agent`로 돌았다는 뜻. control
arm 요건과 다른 경로로 실행됐던 것 확인, 100-step 사이징 런은 이 스크립트
기반이 아니라 `default_agent_loop=tool_agent`+`data.return_raw_chat=True`+
async 모드로 새로 구성해야 함. 이전 STATE의 "~100step≈25분" 추정치는
근거(실측 로그) 없음 확인 — 폐기.

**reward 설계 확인 및 결정**: `verl/workers/reward_manager/naive.py:113-119`에서
`response_str = tokenizer.decode(valid_response_ids)` — `responses` 배치필드는
`AgentLoopOutput.response_ids`(tool_agent_loop.py:195, response_mask와 무관한
전체 토큰) 그대로. `reward_score/gsm8k.py`의 `extract_solution`은 이 전체
디코딩 텍스트 마지막 300자에서 strict `#### N` 매칭 — tool(SandboxTool)이
`print(f"#### {x}")` 형태로 출력하면 모델 자체 텍스트가 그 포맷을 안 내도
reward가 맞을 수 있음. 사용자 확인 후 **의도된 비교로 수용, reward 함수
불변** 결정(gsm8k reward는 oracle 아님 — tool이 계산 결과를 반환하고
모델이 그 결과를 그대로 `#### N`으로 출력하는 것도 tool-use의 정당한
경로로 판단, tool arm 분석 시 정답 출처(tool 응답 구간 vs assistant
자체 텍스트)를 `validation_data_dir` 덤프에서 분리 확인 필요 — 미착수).

## 2026-09-27 SandboxTool timeout 부재 확인 (advisor 지적, GPU 실행 전 블로커)

notebook 원본 `Sandbox.code_execution`(`examples/tutorial/agent_loop_get_started/
agent_loop_tutorial.ipynb` cell-18)의 `process.communicate()`에 timeout
인자 없음 — subprocess는 `asyncio.create_subprocess_exec`로 Docker/kata
없이 순수 subprocess(notebook 자체 경고: "데모용, prod엔 docker/kata
쓰라"). 모델이 `while True: pass`류 코드를 생성하면 해당 rollout 샘플이
무한 대기 — GRPO는 batch 전체가 그 샘플 완료를 기다리므로 배치 전체
정지, GPU 비용 계속 발생. **결정(승인)**: Colab 이식 시
`asyncio.wait_for(process.communicate(), timeout=10~30)`로 감싸고
`asyncio.TimeoutError` 시 `process.kill()` + 실패 응답 반환하도록 수정
— tool arm 코드 작성 시 반드시 포함, 원본 그대로 쓰지 말 것.

## 2026-09-27 K8s 요구범위 재확인 (XiaomiMiMo/verl 조사, 착수 안 함)

지난 조사에서 "code/cyber arm은 Ray만 써서 K8s 불필요"라 추정했으나 env.example
원문 재확인 결과 틀림 — code(`scripts/code/env.example`의 "task pods" 섹션,
`KUBECONFIG`+`MAX_CONCURRENT_SESSIONS`가 "cluster quota"로 제한)와
cyber(`scripts/arvo/arvo.env.example`, `KUBECONFIG` "pod-create permission")
둘 다 K8s 필요. 최소 3/5 도메인이 단일세션(Colab/RunPod) 구조적으로 불가 —
이 리포는 지금 착수 대상 아님, vault [[xiaomi-mimo-verl-oss-2026-09-26]] 참고.

## 2026-09-27 control arm 100-step 완주 성공 + 자동판정 정규식 실패(값이 np.float64() 래핑)

**실행 결과**: exit 0, step 100 완주(commit 869b9fc, Colab auto-save). strict
val(`val-core/openai/gsm8k/acc/mean@1`) 궤적: step0=0.00076, 10=0.057,
20=0.320, 30=0.448, 40=0.453, 50=0.473, 60=0.478, 70=0.470, 80=0.491,
90=0.479, 100=0.480. **판정: 학습 진행 확실** — step0 대비 압도적 상승 후
step40 근처부터 0.45~0.49 사이 plateau, 80→90 소폭 하락(0.491→0.479)은
배치 노이즈 수준. GSM8K STATE 목표(verl agent_loop 인프라 실전 경험)
달성.

**자동판정 스크립트 버그(내가 짠 코드)**: 노트북 셀11의 정규식
`val-core/openai/gsm8k/(?:acc|reward)/mean@\d+:([0-9.]+)`이 0개 매칭.
원인: 실제 로그값이 `val-core/openai/gsm8k/acc/mean@1:np.float64(0.000758...)`
형태로 `np.float64(...)` 래퍼가 씌워져 있는데, 정규식은 숫자가 바로 붙는다고
가정. metric key 이름(`val-core/openai/gsm8k/acc/mean@N`) 자체는 소스
검증대로 정확했음 — 값 포맷(numpy repr)만 미확인 상태로 코드 작성한 게
문제. **교훈**: 로그 포맷 관련 정규식은 "필드 이름이 소스에서 확인됨"과
"그 필드의 값이 어떻게 직렬화되는지"를 별개로 검증해야 함, 후자는
`print()`/`repr()` 체인 전체(logger → concat_dict_to_str → numpy scalar
repr)를 다 봐야 알 수 있어 실제 실행 로그 없이는 코드만으로 확정 불가.
다음 세션에서 노트북 정규식에 `np\.float64\(([0-9.]+)\)` 패턴 추가
필요(아직 미수정).
