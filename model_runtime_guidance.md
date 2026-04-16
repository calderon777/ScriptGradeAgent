# Model And Ollama Guidance

## Document Role

This document defines how local models should be used in this repository.

It answers:

1. which model is sensible for which task,
2. which inference profile should be tested,
3. how to investigate failures such as timeouts, empty JSON, or malformed structure.

It does not define:

- the ideal scoring payload shape,
- the canonical criterion design,
- the engineering work order.

Those belong in:

- [small_model_scoring_payload_design.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/small_model_scoring_payload_design.md)
- [execution_plan.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/execution_plan.md)

## How This Relates To The Other Docs

Read the working docs in this order:

1. [small_model_scoring_payload_design.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/small_model_scoring_payload_design.md)
   - decide the payload and scoring task shape first
2. this file
   - decide how to run local models against that payload
3. [execution_plan.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/execution_plan.md)
   - decide what to change and benchmark next

Short version:

- payload design first
- runtime tuning second
- implementation and benchmark sequencing third

This document explains how to use the locally installed Ollama models in this repository with more discipline.

It is not generic LLM advice. It is written for the actual tasks in `ScriptGradeAgent`:

- assessment-structure extraction
- submission section detection
- subpart refinement
- rubric verification
- per-section scoring
- moderation and final synthesis

## Why This Document Exists

A timeout, empty structure, or malformed JSON is not a confirmation that a model is weak.

It often means one of these:

- the task was too large for one call
- the prompt mixed multiple objectives
- the output contract was underspecified
- the inference profile was not appropriate
- the runtime timeout was too loose or too strict for the task

We should treat models as tools that need task-specific setup, not as interchangeable chat agents.

## Current Repository Status

Current working position in the repository:

1. deterministic assessment-structure extraction is the main baseline,
2. model-based extraction is still useful as a competitor or recovery path,
3. per-section scoring should be tuned only after the extracted assessment structure and payload are trustworthy.

So the current runtime priority is:

- keep extraction calls strict and bounded,
- keep the scoring payload clean and task-specific,
- then test stricter scoring-model inference settings on the same fixed assessment sample.

## Runtime vs Model

`Ollama` is the local runtime.

The current repo-supported models are:

- `qwen2:7b`
- `gemma3:4b`
- `mistral:7b`
- `llama3.1:8b`

Useful local commands:

```powershell
ollama list
ollama show qwen2:7b
ollama show gemma3:4b
ollama ps
```

## Current Repo Settings

The shared Ollama JSON call path is in [core.py](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/marking_pipeline/core.py#L3592).

Current defaults:

- `temperature: 0.1`
- `top_p: 0.9`
- `num_ctx: 8192`
- HTTP timeout: `300s`
- `format: "json"`

These settings are serviceable, but still generic.

They are not yet tuned by task type.

## General Working Rules

### 1. One decision per call

Do not ask a model to:

- infer the full assessment structure,
- infer task type,
- rewrite a rubric,
- and produce scoring criteria

in one large call when the document is long.

Break the work up.

### 2. Long documents should be chunked

A single full-document extraction call is often the wrong test.

For long marking documents, prefer:

1. extract top-level sections first
2. then extract subparts inside each section
3. then derive task and rubric objects from the extracted structure

### 3. Empty output is a failure mode, not a verdict

If a model returns empty sections:

- inspect the prompt
- inspect the raw document text sent
- inspect timeout and context size
- retry on a smaller slice
- compare against a different model on the same slice

Do not treat the first empty response as conclusive evidence that the model cannot do the task.

### 4. Avoid assessment-specific lexical overfitting

Use general cues such as:

- `part`, `question`, `section`, `subsection`, `subpart`, `task`, `item`
- `mark`, `marks`, `point`, `points`
- `%`, `weight`, `weights`, `weighting`
- `worth`, `allocated`, `available`, `out of`
- numbering and containment patterns like `1.`, `(a)`, `Q3`

Do not hard-code topic nouns from one assessment.

## Practical Model Roles

These are not absolute truths. They are the current working hypotheses for this repo.

### Gemma 3 4B

Best current use:

- short structured extraction
- rubric verification
- compact classification tasks
- low-latency experiments

Strengths:

- often fast
- often stable on tight JSON tasks
- good when the prompt is narrow and concrete

Risks:

- may under-extract structure on long, messy documents
- may become too conservative and return too little

Use Gemma when:

- the input is short to medium
- the schema is explicit
- the task is not document-wide and open-ended

### Qwen2 7B

Best current use:

- richer per-section scoring
- tasks where more nuanced internal comparison is useful
- harder extraction when the input is chunked

Strengths:

- can be stronger than Gemma on nuanced scoring
- often benefits from richer structured payloads

Risks:

- can become slow on long extraction tasks
- can timeout if asked to parse large documents in one shot
- can over-process and drift if the task is broad

Use Qwen when:

- the task is important enough to justify more latency
- the input has already been narrowed
- you need stronger judgement, not just label extraction

### Mistral 7B

Best current use:

- backup baseline
- simple JSON extraction
- comparison runs

Strengths:

- useful reference model in bakeoffs

Risks:

- may be less consistent than Gemma on strict structured tasks
- may be less useful than Qwen on nuanced grading tasks

Use Mistral when:

- you want a third comparison point
- you need to test whether a problem is prompt-specific rather than model-specific

### Llama 3.1 8B

Best current use:

- backup baseline
- comparison against Qwen on longer reasoning-heavy tasks

Strengths:

- can be useful as a second strong baseline

Risks:

- may not justify latency relative to Qwen for this workflow

Use Llama when:

- you are comparing structured scoring quality, not only speed

## Chinese-Language Notes For Qwen

I reviewed a small set of Chinese-language sources for Qwen-specific usage patterns. They do not overturn the main engineering direction of this repo, but they do reinforce a few useful points.

### What seems actionable

#### 1. For structured extraction, explicitly ask for JSON and use structured-output mode when available

Alibaba Cloud's Qwen structured-output documentation indicates that:

- JSON mode is intended to suppress extra text around the JSON
- JSON Schema mode is stronger than plain JSON object mode
- prompts should explicitly mention JSON
- structured output support is tied to non-thinking modes for some model families

Practical translation for this repo:

- for extraction and verifier tasks, prefer schema-constrained output over a plain "return JSON" instruction
- keep the prompt explicit that only JSON is allowed
- prefer non-thinking / non-deliberative variants when the task is strict extraction rather than open reasoning

#### 2. Keep Qwen prompts instructional and concrete

A Chinese Qwen prompt-engineering repository describes its recommended techniques as:

- explicit instructions
- zero-shot and few-shot learning
- role prompting
- chain-of-thought
- self-consistency
- retrieval augmentation
- program-aided prompting
- limiting extra tokens

Practical translation for this repo:

- for structure extraction, explicit instruction and a narrow schema are more relevant than free-form reasoning
- few-shot examples may help when labels and nesting are irregular
- limiting extra tokens matches our need to reduce rambling and schema drift
- chain-of-thought should not be requested in the output for strict JSON extraction tasks

#### 3. Do not treat "bigger" or "thinking" variants as automatically better for this workflow

One official Qwen structured-output document explicitly notes that structured output is tied to non-thinking modes for some model families.

Practical translation for this repo:

- for extraction and verifier tasks, "thinking more" may be the wrong mode
- the right target is contract-following and bounded output, not open-ended reasoning
- this helps explain why a smaller or simpler model can outperform a larger one on our structure tasks

### What I would carry into code from these sources

For Qwen extraction tasks:

- use schema-based output if the runtime supports it
- keep `temperature` at `0.0` or very close to it
- cap output with `num_predict`
- prefer staged extraction over one giant document call
- test few-shot examples only for irregular structure cases
- avoid asking for reasoning traces in the final output

### Source Notes

Useful Chinese-language references I checked:

- Alibaba Cloud Qwen structured output docs:
  - https://www.alibabacloud.com/help/zh/model-studio/qwen-structured-output
  - https://www.alibabacloud.com/help/zh/model-studio/json-mode
- Community Qwen prompt-engineering repo:
  - https://github.com/onesuper/Prompt_Engineering_with_Qwen

Treat the community material as suggestive, not authoritative. Anything taken from it should be validated on our own fixed bakeoffs.

## Recommended Task Routing

Current recommended routing to test:

- assessment structure extraction:
  - `gemma3:4b` and `qwen2:7b`, but only in staged extraction
- rubric verifier:
  - `gemma3:4b` first
- per-part scoring:
  - `qwen2:7b` first
- moderation:
  - same scorer model as the section analysis unless a benchmark shows otherwise

## Inference Profiles To Test

These are proposed profiles. They are not all implemented yet.

### Profile A: Strict JSON Extraction

Use for:

- top-level structure extraction
- subpart extraction
- rubric verifier

Recommended test settings:

- `temperature: 0.0`
- `top_p: 0.8`
- `num_ctx: 8192`
- add `seed`
- add `num_predict` cap
- shorter timeout than scoring if the input is chunked

Goal:

- reduce chattyness
- reduce schema drift
- increase reproducibility

### Profile B: Section Scoring

Use for:

- per-section scoring with compact structured payloads

Recommended test settings:

- `temperature: 0.0` or `0.05`
- `top_p: 0.85`
- `num_ctx: 8192`
- add `seed`
- add `num_predict` cap

Goal:

- preserve judgement quality
- suppress decorative variation
- keep outputs contract-bound

### Profile C: Long Structure Recovery

Use for:

- only if chunking is impossible

Recommended test settings:

- `temperature: 0.0`
- `top_p: 0.8`
- larger context only if truly needed
- explicit wall-time budget

Goal:

- keep extraction deterministic
- avoid open-ended document summarization behaviour

## Time And Processing Controls

Current HTTP timeout is `300s`, which is too blunt as a universal setting.

We should move toward task-specific limits:

- short JSON extraction:
  - lower timeout
- scoring:
  - medium timeout
- large benchmark or multi-call workflows:
  - track cumulative time, not just per-call timeout

We should also test:

- `seed` for reproducibility
- `num_predict` to cap overly long outputs
- possibly `repeat_penalty` only if models start echoing the prompt

## Failure Investigation Checklist

When a model times out or returns empty structure:

1. save the exact prompt and input text
2. record the model name and inference settings
3. check document length and extracted word count
4. rerun on a smaller slice of the same document
5. rerun the same slice on another model
6. inspect whether the schema was too broad
7. only then compare the model against heuristics

## Evaluation Standard

When comparing heuristics, `gemma`, `qwen`, `mistral`, and hybrid methods, evaluate:

- top-level section recall
- subpart recall
- mark / weight recovery
- fidelity of instruction text
- task-type usefulness downstream
- runtime
- failure mode:
  - timeout
  - empty output
  - malformed JSON
  - wrong structure

An empty or timed-out run should count as a failed run, but it should also trigger inspection.

## Immediate Next Engineering Steps

1. Introduce task-specific Ollama inference profiles in code instead of one global setting.
2. Add optional `seed` and `num_predict` in the shared Ollama call path.
3. Build a staged assessment-structure extractor:
   - top-level sections
   - subparts within each section
   - structure merge
4. Benchmark `heuristics`, `gemma`, `qwen`, `mistral`, and `hybrid` on the same assessment documents.
5. Keep all prompts assessment-generic and avoid topic-word rules.
