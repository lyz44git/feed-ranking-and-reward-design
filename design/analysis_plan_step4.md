# Analysis plan, step 4: scoring a new video from its caption

Written on 2026-11-24, after the pilot of step 4 and before step 4 was run on the final videos. It adds a step to
[`analysis_plan.md`](analysis_plan.md) and changes nothing in steps 1 to 3. The SHA-256 of this file is in
`logs/plan_hash.txt`.

## What was known when this was written

- Steps 1 to 3 on the final users (they use no caption).
- Every result of step 4 on the **pilot videos** (video_id divisible by 5; 1,505 videos with a main-feed random
  exposure): the first run in `pilot/s4_pilot_first_run.log` and `out/pilot/step4.json`, and the second in
  `pilot/pilot_s4.py`, `pilot/pilot_s4.log` and `out/pilot/pilot_s4.json`.
- Nothing of step 4 computed on the final videos.

## Question

A new video has no viewing record. A platform knows its length, category, upload type and caption. Does its caption
help predict whether users will watch it, and does a large language model's judgment of the caption agree with what
users do?

## Data and split

- Captions: KuaiRand's caption file (Zenodo record 18159199), joined on the video id; 98% of pool videos have one.
  Account mentions are removed; hashtags are kept. The text a model reads is "length in seconds, then caption".
- Videos are the unit. **Final videos:** video_id not divisible by 5. Split by video, seed 20261128: train 50%,
  validation 10%, test 40%.
- Label of a video: long views over main-feed random exposures, all users. Training uses the train videos' counts.
- Scoring: the main-feed random exposures of the test videos, within-user AUC. Intervals: 95%, 500 draws of test
  videos with replacement.

## Models

- **Metadata:** LightGBM (cross-entropy on the rate, weighted by exposures) on length, category, ad flag, age,
  music type, aspect ratio, upload type and caption length. Early stopping on the validation videos.
- **Metadata + frozen text:** the same with 32 principal components of Qwen2.5-0.5B's mean-pooled caption embedding.
- **Metadata + fine-tuned text:** Qwen2.5-0.5B with a linear head on the last token, its last four blocks fine-tuned
  (learning rate 1e-5, three epochs, the epoch with the lowest validation loss kept, three seeds averaged), with a
  binomial loss on each video's counts. Its output is added to the metadata model's out-of-fold logit, so it learns
  what the text adds. On the pilot no fine-tuning setting (three tried) lowered the validation loss, so the kept model
  equalled the metadata model; the final train set is four times larger.
- **Zero-shot judge:** Qwen2.5-7B-Instruct asked whether a typical user would watch the video for 18 seconds (or to
  its end if shorter), scored by the log-odds of "yes" against "no"; no training. A second prompt asks for a rating
  from 0 to 9, scored by the expected rating.
- **Reference, not cold start:** the video's long-view rate among the other viewers of its random exposures.

## Rule

*Rule 4:* the metadata model's within-user AUC exceeds the zero-shot judge's (yes or no prompt), with the interval of
the difference above zero. Pilot: metadata 0.611, judge 0.495, difference +0.117 (0.082 to 0.153).

Reported without a rule: every model's within-user AUC and its correlation with the observed rate over test videos;
the differences of the text models from the metadata model; the rating prompt (pilot 0.517); the reference (pilot
0.695).

## What this cannot show

- Two prompts and one judge model. A larger model, other prompts, or the video frames could do better.
- The videos are the 7,583 of the pool; about 3,000 train videos are few for fine-tuning a language model.
- A long view is one outcome; a caption could predict likes or shares better.
