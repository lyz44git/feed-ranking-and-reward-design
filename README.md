# What a feed ranker learns from matters more than the model.

[![tests](https://github.com/lyz44git/feed-ranking-and-reward-design/actions/workflows/tests.yml/badge.svg)](https://github.com/lyz44git/feed-ranking-and-reward-design/actions/workflows/tests.yml)

**On 369,177 random exposures of 8,644 held-out users of a short-video app, training a ranker on random exposures
instead of the standard feed raised the long-view rate of its top picks from 12.2% to 19.3%. Changing the model moved
it by about one point.**

When a short-video feed ranks candidate videos for a user, which choices decide whether the ranking serves the user?

In the data studied here, the logs the ranker learns from and the outcome it is rewarded for matter more than the
model. Kuaishou replaced a share of the recommendations in its app with videos drawn at random from a pool of 7,583
and released the logs (KuaiRand; [Gao et al. 2022](https://doi.org/10.1145/3511808.3557624)). The recommender did not
choose these exposures, so users' responses to them show what a ranker would obtain by showing each user its
top-ranked candidates, with no model of how the feed chooses. (A long view is a play of at least 18 seconds, or of the
whole video if it is shorter.)

Measured this way, the usual offline check points the wrong way. The same network trained on the standard feed scores
better on held-out standard-feed exposures (AUC 0.71 against 0.65), yet the videos it ranks highest are long-viewed
12.2% of the time against 19.3%, and it predicts 3.7 times the actual rate. A reward for the share of a video watched
selects 9-second videos and lowers watch time below that of videos shown at random.

This repository examines four choices in turn: the model that predicts a user's response, the logs it learns from,
the outcome it is rewarded for, and how a new video with no viewing record can be scored from its caption. All data
are public. Each of the four pass-or-fail rules was set on a pilot sample that is not part of the reported results,
then fixed and hashed before the result was seen; all four were met. Results that did not favor the more complex
method are reported alongside the rest.

**Scope.** The results are offline estimates for one app over one month in 2022. They show how a ranker's choices
change the outcomes of the videos it ranks highest among a user's randomly exposed candidates. They do not estimate
the effect of a deployed feed, in which exposures interact, or effects on later visits and retention, which the
release does not allow to be followed. What generalizes is the method: checking a ranker's offline metrics against
exposures the recommender did not choose.

<img src="figures/figure1_final.png" alt="Four panels: within-user ranking accuracy by model; long-view rate of the top-ranked videos by training logs; watch time by reward; ranking accuracy for new videos scored from metadata, caption models and a language model's judgment." width="100%">

**Figure 1. What changes the outcomes of a feed ranker's top picks.** *Notes:* Panels A to C are computed on the
main-feed random exposures of the 8,644 test users who have any, with 95% intervals from 500 bootstrap draws of users.
Panel A shows the share of pairs of the same user's exposures (one long-viewed, one not) that each model orders
correctly, and each model's AUC over all exposures. Panels B and C show, for each user, the exposures in the top 10%
of the user's random exposures by a model's score, and the mean outcome over them. Panel D is computed on the 2,413
test videos of section 4, none of which is in its training data, with intervals from 500 bootstrap draws of videos;
the dashed line is chance. Navy marks the result the study stands behind, grey a simple baseline, blue other models
and orange a number that misleads.

## 1. Which model is used changes the outcome by about one point.

Among models trained on random exposures, a Transformer over the user's last 50 events orders a user's candidates
better than gradient-boosted trees given summaries of the same history: a within-user AUC of 0.689 against 0.684, a
difference of 0.0049 (0.0027 to 0.0071). Within-user AUC is the share of pairs of the same user's exposures, one
long-viewed and one not, that a score orders correctly; it ignores which users watch more, which a ranker cannot
change.

The gain does not come from reading the history in order. The same network without the Transformer does slightly
better (0.690; difference −0.0013, −0.0024 to −0.0002). Histories in this release contain only the pool's videos, so a
user's recent viewing outside the pool is missing. What helps is the training loss. Without a pairwise loss over pairs
of the same user, the Transformer has the highest overall AUC of any model (0.781) and a lower within-user AUC
(−0.0079, −0.0097 to −0.0063): it learns which users watch a lot, which is most of what overall AUC rewards and none of
what a ranker can use.

The resulting policies differ little. The long-view rate of each model's top 10% ranges from 18.2% to 19.5%, against
8.5% for videos shown at random.

<details><summary>All models</summary>

| Model, trained on random exposures, label long view | Overall AUC | Within-user AUC (95% CI) | Long view, top 10% |
|---|---|---|---|
| LightGBM, no history | 0.677 | 0.677 (0.668 to 0.685) | 19.2% |
| LightGBM, history summaries | 0.753 | 0.684 (0.676 to 0.692) | 18.7% |
| Network without the Transformer | 0.738 | 0.690 (0.681 to 0.699) | 19.5% |
| Transformer, label loss only | 0.781 | 0.681 (0.673 to 0.689) | 18.2% |
| Transformer with the within-user loss | 0.757 | 0.689 (0.680 to 0.698) | 19.3% |

</details>

<sub><b>Methods.</b> LightGBM with and without history summaries (counts and rates of the user's earlier events, overall and in the candidate's category). A PyTorch network with embeddings of the video, its category and length, the user's features and the same summaries, and a two-layer Transformer encoder over the last 50 events, each with what the user did on it, with the candidate as the last token. Trained on the label's loss plus a pairwise logistic loss over pairs of the same user; three seeds, the epoch with the lowest validation loss kept. Users split 50/10/40 into train, validation and test; 470,538 random exposures of the train users.</sub>

## 2. Trained on the standard feed, the ranker looks better offline and recommends worse.

Trained on random exposures, the ranker's top picks are long-viewed 19.3% of the time (18.6% to 20.0%). The same
network trained on the train users' standard-feed exposures, the logs a team normally has, reaches 12.2% (11.7% to
12.8%): 7.1 points less (6.7 to 7.6), and 3.7 seconds less watch time per exposure (3.4 to 4.0).

The usual offline check prefers the worse ranker. Scored on held-out standard-feed exposures, the model trained on
the standard feed wins by 0.06 of AUC (0.713 against 0.655). It has learned which of the feed's own choices users
watch, not which videos a user would watch among those the feed did not choose; on random exposures its within-user
AUC is 0.571 against 0.689, and it predicts 3.7 times the long-view rate of a random video. A team that chose between
the two on its usual offline metric would ship the worse ranker.

<sub><b>Methods.</b> The network of section 1, trained once on 470,538 random and once on 419,032 standard-feed exposures of the same train users. Policy value: for each test user, the top 10% of the user's random exposures by score (at least one), and the mean outcome over them. Standard-feed exposures are long-viewed four times as often as random ones (33.7% and 31.3% against 8.5%), which is the selection the feed makes.</sub>

## 3. Rewarded for the share of a video watched, the ranker picks 9-second videos.

Rewarded for seconds watched, the ranker's picks are watched for 12.6 seconds per exposure, 7.8 seconds longer
(7.4 to 8.2) than when it is rewarded for the share of the video watched (4.8 seconds).

The share-watched reward is gamed by video length. It raises the share watched from 16%, for videos shown at random,
to 57% by picking videos of about 9 seconds, which are easy to finish and short to watch: 2.2 seconds less than a
random video (2.0 to 2.3). The metric it optimizes rises while the time users spend falls.

Rewards that do not depend on length trade off rather than dominate. The watch-time reward gives the most seconds,
from videos of 110 seconds, and fewer long views than the long-view reward (16.9% against 19.3%). Likes and dislikes
are rare (about 6 and 1 per 1,000 exposures) and move little under any reward.

<sub><b>Methods.</b> The network of section 1 trained on random exposures with three labels: the share of the video watched (watch time over length, capped at one), long view, and the log of seconds watched. Each policy shows each user the top 10% of the user's random exposures by its score; "shown at random" shows the same number drawn at random.</sub>

## 4. For a new video, a language model's judgment of its caption is close to chance.

A new video has no viewing record. A platform knows its length, category, upload type and caption. Scored on the
377,132 random exposures of 2,413 test videos to 26,857 users, a tree model on that metadata orders a user's new
candidates with a within-user AUC of 0.622 (0.614 to 0.631). Qwen2.5-7B-Instruct, asked whether a typical user would
watch the video for 18 seconds given its length and caption, reaches 0.504 (0.493 to 0.514), close to chance and 0.118
below the metadata model (0.103 to 0.133). It answers yes for 42% of videos, and its answers have a correlation of
0.03 with the videos' long-view rates; asking it for a rating from 0 to 9 instead gives 0.523. A language model's
opinion of a caption is not a measure of what users watch, and random exposures show this before such an opinion is
used as a label or a reward.

The caption itself carries a little information. Adding 32 components of a small language model's embedding of it to
the metadata raises the within-user AUC by 0.0065 (0.0013 to 0.0124), about a tenth of the gap between metadata and a
video's long-view rate among its other viewers (0.695), which is available only once the video has been shown.

Fine-tuning that model did not help with 3,000 training videos. With its last four blocks fine-tuned on each video's
long views out of its exposures, Qwen2.5-0.5B never had a validation loss below that of the metadata model it starts
from, in any of three seeds; the kept model is therefore the metadata model. The same happened on the pilot with
every setting tried.

<details><summary>All scores</summary>

| Score of a new video | Within-user AUC (95% CI) | Correlation with the video's long-view rate |
|---|---|---|
| Metadata (LightGBM) | 0.622 (0.614 to 0.631) | 0.49 |
| Metadata + caption embedding (Qwen2.5-0.5B, 32 components) | 0.628 (0.620 to 0.638) | 0.53 |
| Metadata + fine-tuned Qwen2.5-0.5B | 0.622 (0.614 to 0.631) | 0.49 |
| Qwen2.5-7B-Instruct, yes or no | 0.504 (0.493 to 0.514) | 0.03 |
| Qwen2.5-7B-Instruct, rating from 0 to 9 | 0.523 (0.513 to 0.535) | 0.15 |
| Reference, not cold start: rate among the video's other viewers | 0.695 (0.688 to 0.703) | |

</details>

<sub><b>Methods.</b> Videos split 50/10/40 into train (3,016), validation (603) and test (2,413); each video's label is its long-view rate over its main-feed random exposures. Metadata: LightGBM with a cross-entropy loss on the rate, weighted by exposures, on length, category, ad flag, age, music type, aspect ratio, upload type and caption length. Caption models read "length in seconds, then caption", with account mentions removed. The fine-tuned model adds its output to the metadata model's out-of-fold logit, so it learns what the text adds, with a binomial loss on each video's counts. The judge is scored by the log-odds of "yes" against "no", with no training. Intervals from 500 bootstrap draws of test videos.</sub>

## 5. Published research reports the same pattern.

- **Logged feedback is not a random sample.** Treating recommendations as treatments, feedback observed under a
  recommender gives biased estimates for the items it did not show
  ([Schnabel et al. 2016](https://arxiv.org/abs/1602.05352)). KuaiRand was released to allow unbiased evaluation with
  randomly exposed videos ([Gao et al. 2022](https://doi.org/10.1145/3511808.3557624)); section 2 measures the size of
  the bias for one ranker.
- **Watch time over clicks.** YouTube's recommender ranks by expected watch time rather than by the probability of a
  click, because clicks favor videos that users do not finish
  ([Covington, Adams, and Sargin 2016](https://doi.org/10.1145/2959100.2959190)). Section 3 finds the converse for a
  share-watched reward, which favors videos that are short.
- **Sequence models.** Self-attention over a user's sequence of actions is the basis of recent sequential and
  generative recommenders ([Kang and McAuley 2018](https://arxiv.org/abs/1808.09781);
  [Zhai et al. 2024](https://arxiv.org/abs/2402.17152)), trained on long, complete histories. With histories restricted
  to a pool of 7,583 videos, section 1 finds no gain from the order of the history.
- **Simple baselines.** Several neural recommenders did not beat well-tuned simple baselines when re-run
  ([Ferrari Dacrema, Cremonesi, and Jannach 2019](https://doi.org/10.1145/3298689.3347058)). Here the network beat the
  trees, and the gain came from its loss, not its architecture.
- **Reward hacking.** An agent that optimizes a proxy can raise the proxy while lowering the objective
  ([Amodei et al. 2016](https://arxiv.org/abs/1606.06565)), as the share-watched reward does in section 3.
- **Language models as judges.** Strong language models agree with human preferences on chat answers about as often
  as humans agree with each other ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685)). Predicting what other people
  will watch is a different task, and section 4 finds that a 7-billion-parameter model's judgment does not predict it.

## Design

The analysis plan for sections 1 to 3, [`design/analysis_plan.md`](design/analysis_plan.md), was fixed before any
model was run on the final users. The settings and the three thresholds were chosen on the **pilot users** (user ids
divisible by five; 5,457 users), whose scripts, log and results are kept in [`pilot/`](pilot) and
[`out/pilot/results.json`](out/pilot/results.json). The final users, the other 21,828, are split by user into train
(10,914), validation (2,182) and test (8,732, of whom 8,644 have random exposures in the main feed). Examples are
exposures in the main feed, where 99% of random exposures happen; the history of an exposure is the same user's
earlier events of either kind, with a median of 86 events at the test exposures.

Section 4 was planned after sections 1 to 3 had been run, in its own plan,
[`design/analysis_plan_step4.md`](design/analysis_plan_step4.md). Its pilot is the videos with ids divisible by five;
the pilot's two runs, in which no fine-tuning setting lowered the validation loss and the judge showed no signal, are
in [`pilot/`](pilot) and [`out/pilot/`](out/pilot). The SHA-256 of both plans, with the time each was recorded, is in
[`logs/plan_hash.txt`](logs/plan_hash.txt).

## Data sources and terms

Data are not included in this repository.

| Data | Used in | Terms |
|---|---|---|
| KuaiRand-Pure ([Gao et al. 2022](https://doi.org/10.1145/3511808.3557624)), [Zenodo 10439422](https://zenodo.org/records/10439422): logs of 27,285 users from 8 April to 8 May 2022 on a pool of 7,583 videos, with 1,186,059 random exposures from 22 April | 1 to 4 | CC BY 4.0. |
| KuaiRand video captions, [Zenodo 18159199](https://zenodo.org/records/18159199) (`kuairand_video_captions.csv`); 98% of pool videos have an entry | 4 | CC BY 4.0. |
| Qwen2.5-0.5B and Qwen2.5-7B-Instruct, from Hugging Face | 4 | Apache 2.0. |

| Log | Exposures | Users | Long view | Seconds watched |
|---|---|---|---|---|
| Standard feed, 8–21 April | 1,141,112 | 26,210 | 33.7% | 23.3 |
| Standard feed, 22 April–8 May | 295,497 | 25,877 | 31.3% | 21.5 |
| Random exposures, 22 April–8 May | 1,186,059 | 27,285 | 8.5% | 6.9 |

**Use.** This is a non-commercial research project. The data and all derived outputs in this repository are used for
research only.

## Repository layout

```
design/      analysis plans, each hashed before its run on the final data
pilot/       the exploratory scripts and logs that chose the settings and thresholds
src/kr/      data preparation, the network, unbiased evaluation, the caption models' data and evaluation
steps/       one script per stage, in run order; results in out/<sample>/
out/         results.json (sections 1 to 3) and step4.json (section 4), for the pilot and the final sample
figures/     Figure 1, drawn from the result files
logs/        run logs and the SHA-256 of the plans
tests/       unit tests (no data needed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `KuaiRand-Pure.tar.gz` and `kuairand_video_captions.csv` from the Zenodo records above, then run:

```bash
python steps/s1_baselines.py <KuaiRand-Pure/data> --sample final
python steps/s123_networks.py <KuaiRand-Pure/data> --sample final
python steps/s_summary.py <KuaiRand-Pure/data> --sample final
python steps/s4_extract_captions.py <kuairand_video_captions.csv> <KuaiRand-Pure/data> pool_captions.parquet
python steps/s4_captions.py <KuaiRand-Pure/data> pool_captions.parquet --sample final
python steps/s_figures.py --sample final
```

`--sample pilot` runs the same steps on the pilot. Tested with Python 3.13 on macOS (Apple M3 Ultra, PyTorch MPS).
The networks of sections 1 to 3 take about 30 minutes on the final users, and section 4, which loads both language
models from Hugging Face, about 25 minutes.

## Tests

```bash
pytest -q
```

The tests need none of the data. They check that the splits of users and of videos are deterministic and keep their
shares, and that pilot and final samples do not overlap; that a history reads only the same user's earlier events and
that no history summary uses the current or a later outcome; that within-user AUC equals the usual AUC for one user,
ignores shifts in a user's scores, gives ties half credit, and gives the same value with weights as with repeated
rows; that a policy picks each user's top share; and that captions lose account mentions but keep hashtags. They run
on every push (see the badge above).

## License

Code is released under the MIT License (see `LICENSE`). The datasets and models keep their own terms, listed above.

## Notes

Developed with an AI-assisted coding workflow. The research questions, analysis plans and interpretation of results
were led by the author. All results are reproducible from the scripts and public data referenced above.
