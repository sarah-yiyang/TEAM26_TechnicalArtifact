# Working notes

My notes from BuildFest, grouped by topic. The README covers what the tool is and how to run it.

## The challenge

Question: How can EveryMind use its 988 training feedback data to develop an actionable solution for evaluating training effectiveness, identifying gaps across communities, and continuously improving future trainings?

- Prizes to aim for: Most Effective Visualization and EveryMind's Choice.
- EveryMind runs 988 Awareness Training and community education sessions. They help people understand how the 988 Suicide & Crisis Lifeline works and how to support themselves or others.
- The sessions are live, interactive, and virtual.

## Reading the challenge

- Two keywords guide the build: continuous and actionable.
- Continuous: an automated pipeline fits best, because EveryMind wants this to keep running. Insights must be reproducible.
- Actionable: the tool should suggest actions, not just show numbers.
- The survey design itself may need changing. A post-only review can hide bias that was there from the start.

## The dataset

- It is people's feedback on EveryMind's trainings.
- The Email and Name columns are not useful for analysis.

## What EveryMind needs

- A tool that visualizes and summarizes feedback automatically, so staff can adjust trainings and see how each change lands.
- Fast enough to act on before the next round of school sessions.
- Output in a form they can paste into a grant report.
- Business value: staff save time reading responses.
- The pipeline is the center of the artifact, but key insights need real effort too.

## Feature ideas

- See how the audience mix (demographics) changes over time.
- Summarize what people valued most, so staff know what to keep.
- Summarize the main improvement suggestions.
- Maybe include a redesigned survey.

Stretch goals:
- Concrete impact targets, for example: "Goal: raise the student likely-to-call rate from 41% to 55% by next spring, tracked monthly in the tool."
- A "know it vs. would use it" chart.

## Suggestions for EveryMind

- Add a line to the survey intro such as: "This information is used only to improve the training, not to assess you personally."

## Open items

- Comparison mode. Right now it is hard to see what changed after new results come in. Idea: same dataset plus a cutoff date as input, output a before/after comparison of the metrics. Call the current view "Cumulative overview".
- AI summaries (milestone 5): if funding allows, AI could give more specific feedback and summaries. The current tool uses keyword rules only.

## Key decisions while building with AI

- app.py cleanup: went from one messy page to a sidebar nav with dropdowns, and made comments readable on the comment board.
- M2: reworked the "what changed for each person" chart; swapped confusing graphs for easier ones.
- M4: rewrote vague wording in the insight checks.
- M4: changed the whole insight card structure.
- M5: changed the comment board layout and added sample questions.
- AI proposed an AI API section, but it repeated the M4 insight cards, so I removed it and turned it into the Program brief.

## What I learned

- Start the product (app.py) from the business side first, so the layout suits the people who will use it.
- dropping ai component
