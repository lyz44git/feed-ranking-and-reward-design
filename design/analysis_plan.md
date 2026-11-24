# Analysis plan

Written on 2026-11-24, after the pilot and before any model was run on the final users. The SHA-256 of this file is
in `logs/plan_hash.txt`.

## What was known when this was written

- Descriptive counts of the whole release (rows, users, feedback rates by log, exposures per video).
- Every result on the **pilot users** (user_id divisible by 5; 5,457 users), in `pilot/` and `out/pilot/results.json`.
  The pilot was used to choose the model settings and the three rules below. Two choices came from it: the
  within-user pairwise loss, added after a Transformer trained on the label's loss alone ranked videos within a user
  worse than the tree baseline; and the history summaries as inputs to the networks.
- Nothing computed on the final users beyond the descriptive counts.

## Data

KuaiRand-Pure (Gao et al. 2022): logs of 27,285 users of the Kuaishou app from 8 April to 8 May 2022 on a pool of
7,583 videos. From 22 April to 8 May a small share of recommendations were replaced by a video drawn uniformly at
random from the pool (1,186,059 exposures, `is_rand = 1`). The other exposures come from the standard feed.

- **Final users:** user_id not divisible by 5. Split by user, seed 20261124: train 50%, validation 10%, test 40%.
- **Exposures used as training examples and for scoring:** those in the main feed (`tab = 1`).
- **History of an exposure:** the same user's earlier events of either kind, by timestamp.

## Evaluation

All scoring is on the test users' random exposures. A user's random exposures are a uniform sample of the pool, so
ranking them needs no model of the logging policy.

- **Within-user AUC:** the share of pairs (long view, no long view) of the same user that a score orders correctly,
  pooled over users.
- **Policy value:** for each user, the exposures in the top 10% of the user's random exposures by score (at least
  one); the mean outcome over them, pooled over users.
- **Intervals:** 95%, from 500 bootstrap draws of test users.

## Models

- LightGBM without history; LightGBM with history summaries (counts and rates over the user's earlier events,
  overall and within the candidate's category). Early stopping on the validation users.
- The network: embeddings of the candidate video, its category and length, the user's features, the history
  summaries, and a two-layer Transformer encoder over the last 50 events with the candidate as the last token.
  Loss: the label's loss plus a pairwise logistic loss over pairs of the same user. Three seeds, at most five epochs,
  the epoch with the lowest validation loss kept, predictions averaged over seeds.
- Variants: the network without the Transformer; the network without the pairwise loss.

## Steps and rules

**Step 1, prediction.** Label: long view. Trained on the train users' random exposures.
*Rule 1:* the network's within-user AUC exceeds that of LightGBM with history summaries, with the interval of the
difference above zero. Pilot: +0.011 (0.007 to 0.015).
Reported without a rule: the difference from the network without the Transformer (pilot: 0.000, −0.001 to 0.001)
and from the network without the pairwise loss (pilot: +0.008), and the overall AUC of every model.

**Step 2, which logs to learn from.** The same network trained on the train users' standard-feed exposures instead.
*Rule 2:* the policy from the network trained on random exposures has a higher long-view rate than the policy from
the network trained on the standard feed, with the interval of the difference above zero. Pilot: +4.3 points
(3.5 to 5.2).
Reported without a rule: each network's AUC on the standard feed and on random exposures, and its mean prediction
over the actual long-view rate on random exposures.

**Step 3, which reward.** The same network trained on random exposures with the completion share (watch time over
video length, capped at one) and with log watch time as labels.
*Rule 3:* the policy trained on watch time has more watch seconds than the policy trained on completion, with the
interval of the difference above zero. Pilot: +7.1 seconds (6.3 to 7.7).
Reported without a rule: every outcome (long view, watch seconds, completion, valid play, like, dislike, video
length) under each reward and under showing the same number of exposures at random.

## What this cannot show

- The policy value is that of choosing among a user's randomly exposed candidates, one exposure at a time. It is not
  an estimate of a deployed feed, where exposures interact.
- The outcomes are per exposure. The release does not allow the rest of a session to be followed, so effects on later
  behaviour are not measured.
- Histories in this release contain only pool videos; the comparison with and without the Transformer is conditional
  on that.
- One app, one month.
