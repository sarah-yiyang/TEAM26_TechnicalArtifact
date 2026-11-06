# BuildFest

Question: How can EveryMind use its 988 training feedback data to develop an actionable solution for evaluating training effectiveness, identifying gaps across communities, and continuously improving future trainings?

## Goal
* aim for: Most Effective Visualization & EveryMind's choice

## EveryMind
* EveryMind offers 988 Awareness Training and community education sessions to help individuals understand how the 988 Suicide & Crisis Lifeline operates and how to support themselves or others.
* interactive, live virtual sessions

## notes about challenge
* automated pipeline may be the best choice cuz they want the thing to be continuous. insights should be reproduciable 
* actionable! ? so artifact should suggest actions
* may include that the survey structure should be changed in the first place? post review may ignore bias in the first place
* limit: AI api...

## notes about dataset
* the dataset is people's feedback on EveryMind's training
* Email	Name useless

## conclusions real
2 keywords in building my artifact: continuous & actionable.
* everymind needs a tool that automatically visualizes and summarizes feedback where they can then adjust training if they want and automatically give feedback for every changes to the training.
* fast enough to act on before the next round of school sessions, and in a form they can paste into a grant report.
* maybe include a redesigned survey
* stretch goal: 
  * concrete impact targets: "goal: raise the student likely-to-call rate from 41% to 55% by next spring, tracked monthly in the tool"
  * know it vs. would use it chart
* ok so i center around my artifact to the pipeline but i also need efforts on making key insights. 

## business suggestions
* should include explanation in title in survey that "this info is used for improvements only not assessing ur personal preference..."

idea
* 1 feature to see trend of demographics
* 1 feature to summarize people's most valuable part: so where to keep in the training
* 1 feature to summarize main improvement suggestions
* to explain in business value: staff will save time reading responses. 

lingering changes
* add a nav bar

* reorganize layout to emphasize the 3 features
* you can't really tell difference after a set of new stuff came in.  need a comparison mode: compare to prev training results. program input: same dataset, input cutoff date. output: comparison of metrics. name current mode cumulative overview.

## key ai decisions
* cleanup app.py
* deleting some sections, recognized that app.py includes repeated info because i wanted to see app.py running after every milestones where m4 builds on top of m2 and m3; only m4 should really be kept.
* original graphs are difficult to understand... changing.

## learned
* need to start the product (app.py) from business perspective first so layout is friendly to targeted audience