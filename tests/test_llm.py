import json

import pytest

from judgepack.llm import LLM, LLMError, Provider, RawResponse, extract_json, status_of
from judgepack.schemas import Brief, PairwiseJudgement


class Scripted(Provider):
    """Replays a list of replies (strings) or exceptions."""

    name = "scripted"
    model = "test-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, *, system, prompt, schema, temperature, task, context):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return RawResponse(text=r, input_tokens=100, output_tokens=20)


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


GOOD = json.dumps({"reason": "B works end to end.", "better": "B", "confidence": "high"})


def make(replies, tmp_path, retries=3):
    return LLM(Scripted(replies), cache_dir=tmp_path, requests_per_minute=0, max_retries=retries, sleep=lambda s: None)


def call(llm, prompt="compare"):
    return llm.run(task="pairwise", system="sys", prompt=prompt, schema=PairwiseJudgement, temperature=0.2)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} Hope that helps.') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_invalid_reply_triggers_repair_then_succeeds_and_caches(tmp_path):
    llm = make(['{"better": "C"}', GOOD], tmp_path)
    out = call(llm)
    assert out.better == "B"
    assert "could not be used" in llm.provider.prompts[1]
    u = llm.usage.total()
    assert (u.calls, u.failures, u.input_tokens) == (2, 1, 200)

    # Same prompt again: answered from the cache, no new call.
    llm2 = make([], tmp_path)
    assert call(llm2).better == "B"
    assert llm2.usage.total().cache_hits == 1 and llm2.usage.total().calls == 0


def test_rate_limit_is_retried(tmp_path):
    llm = make([HTTPError(429), HTTPError(503), GOOD], tmp_path)
    assert call(llm).better == "B"
    assert llm.usage.total().failures == 2


def test_auth_and_bad_model_fail_fast_with_a_clear_message(tmp_path):
    with pytest.raises(LLMError, match="API key"):
        call(make([HTTPError(401)], tmp_path))
    with pytest.raises(LLMError, match="model"):
        call(make([HTTPError(404)], tmp_path / "b"))


def test_gives_up_after_max_retries(tmp_path):
    with pytest.raises(LLMError, match="no usable reply"):
        call(make(["nope", "still nope"], tmp_path, retries=2))


def test_check_function_rejects_semantically_bad_output(tmp_path):
    bad = json.dumps({"one_line": "", "summary": "", "how_it_works": "", "strengths": [],
                      "questions_for_team": [], "unverified_claims": [], "challenge_fits": []})
    good = json.dumps({"one_line": "x", "summary": "y", "how_it_works": "z", "strengths": [],
                       "questions_for_team": ["a", "b", "c"], "unverified_claims": [], "challenge_fits": []})

    def check(b):
        if not b.summary:
            raise ValueError("summary must not be empty")

    llm = make([bad, good], tmp_path)
    out = llm.run(task="brief", system="s", prompt="p", schema=Brief, temperature=0.2, check=check)
    assert out.summary == "y"


def test_status_of_reads_common_shapes():
    assert status_of(HTTPError(429)) == 429
    assert status_of(Exception("got 503 Service Unavailable")) == 503
    assert status_of(Exception("boom")) is None


def test_gemini_provider_reads_text_and_usage(monkeypatch):
    pytest.importorskip("google.genai")
    from judgepack.llm import GeminiProvider

    provider = GeminiProvider("gemini-test", api_key="not-a-real-key")

    class Usage:
        prompt_token_count = 11
        candidates_token_count = 7
        thoughts_token_count = 3  # thinking tokens are billed as output

    class Resp:
        text = GOOD
        usage_metadata = Usage()

    monkeypatch.setattr(provider._client.models, "generate_content", lambda **kw: Resp())
    raw = provider.generate(system="sys", prompt="hello", schema=PairwiseJudgement, temperature=0.2,
                            task="pairwise", context=None)
    assert raw.text == GOOD and raw.input_tokens == 11 and raw.output_tokens == 10


def test_gemini_request_payload_keeps_reasons_before_verdicts():
    """Stop the real SDK just before the HTTP call and inspect what it would send."""
    pytest.importorskip("google.genai")
    from judgepack.llm import GeminiProvider
    from judgepack.schemas import RubricScore

    provider = GeminiProvider("gemini-test", api_key="not-a-real-key")
    captured = {}

    class Stop(Exception):
        pass

    def fake_request(method, path, body, *args, **kwargs):
        captured.update(path=path, body=body)
        raise Stop

    provider._client._api_client.request = fake_request
    with pytest.raises(Stop):
        provider.generate(system="sys", prompt="hello", schema=PairwiseJudgement, temperature=0.2,
                          task="pairwise", context=None)
    gen = captured["body"]["generationConfig"]
    assert captured["path"].endswith("gemini-test:generateContent")
    assert gen["responseMimeType"] == "application/json" and gen["temperature"] == 0.2
    assert gen["responseSchema"]["property_ordering"] == ["reason", "better", "confidence"]
    assert "systemInstruction" in captured["body"]

    with pytest.raises(Stop):
        provider.generate(system="s", prompt="p", schema=RubricScore, temperature=0.4, task="score", context=None)
    item = captured["body"]["generationConfig"]["responseSchema"]["properties"]["criteria"]["items"]
    assert item["property_ordering"] == ["criterion_id", "evidence", "evidence_level", "score"]


def test_anthropic_provider_builds_a_valid_request(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    from judgepack.llm import AnthropicProvider

    provider = AnthropicProvider("claude-test")
    seen = {}

    class Block:
        type = "text"
        text = GOOD

    class Msg:
        content = [Block()]

        class usage:  # noqa: N801
            input_tokens = 9
            output_tokens = 4

    def fake_create(**kwargs):
        seen.update(kwargs)
        return Msg()

    monkeypatch.setattr(provider._client.messages, "create", fake_create)
    raw = provider.generate(system="sys", prompt="hello", schema=PairwiseJudgement, temperature=0.2,
                            task="pairwise", context=None)
    assert raw.text == GOOD and (raw.input_tokens, raw.output_tokens) == (9, 4)
    assert seen["system"] == "sys" and seen["messages"][0]["content"] == "hello"


def gemini_429(quota_id, value, retry="27s"):
    """An error shaped like google-genai's ClientError for a quota 429 (str is '429 RESOURCE_EXHAUSTED. {...}')."""
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "You exceeded your current quota, please check your plan and billing details.",
                      "details": [
                          {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                           "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                           "quotaId": quota_id, "quotaValue": str(value),
                                           "quotaDimensions": {"model": "gemini-3.6-flash", "location": "global"}}]},
                          {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}]}}
    exc = Exception(f"429 RESOURCE_EXHAUSTED. {body}")
    exc.code = 429
    exc.details = body
    return exc


def test_daily_quota_stops_at_once_and_says_the_limit(tmp_path):
    from judgepack.llm import QuotaExhausted

    llm = make([gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", 20), GOOD], tmp_path)
    with pytest.raises(QuotaExhausted, match=r"daily quota.*20 requests a day.*8am UK"):
        call(llm)
    assert len(llm.provider.prompts) == 1  # no retries: waiting a minute can't fix a daily quota
    with pytest.raises(QuotaExhausted):
        call(llm, prompt="another project")
    assert len(llm.provider.prompts) == 1  # later calls don't even try


def test_zero_quota_is_reported_as_no_free_quota(tmp_path):
    from judgepack.llm import QuotaExhausted

    with pytest.raises(QuotaExhausted, match="no free quota"):
        call(make([gemini_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", 0)], tmp_path))


def test_per_minute_quota_waits_as_long_as_the_server_says(tmp_path):
    waits = []
    llm = LLM(Scripted([gemini_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", 5, "27s"), GOOD]),
              cache_dir=tmp_path, requests_per_minute=0, max_retries=3, sleep=waits.append)
    assert call(llm).better == "B"
    assert len(waits) == 1 and 28 <= waits[0] <= 29


def test_a_stop_error_cancels_the_rest_of_the_batch():
    from judgepack.llm import LLMError, QuotaExhausted
    from judgepack.util import run_parallel

    started = []

    def fn(i):
        started.append(i)
        if i == 0:
            raise QuotaExhausted("daily quota used up")
        if i == 1:
            raise LLMError("one bad reply")
        return i

    with pytest.raises(QuotaExhausted):
        run_parallel(fn, list(range(50)), 1, "test", catch=(LLMError,), stop=(QuotaExhausted,))
    assert len(started) < 50  # the rest were cancelled, not run

    results, errors = run_parallel(fn, [1, 2, 3], 1, "test", catch=(LLMError,), stop=(QuotaExhausted,))
    assert results == [None, 2, 3] and errors == ["one bad reply"]
