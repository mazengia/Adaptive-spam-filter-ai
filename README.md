# Adaptive-spam-filter-ai
Feedback & Learning and Model Versions
Spam filter service: what the two tabs do and how to use them together
Both tabs support one loop: people correct the model's mistakes, you retrain, and you check whether the new model is better or roll back.
Feedback & learning tab
This tab turns your corrections into a better model.
1. Shows the corrections collected so far
• Corrections saved: the total number of times someone clicked "This is spam" or "This is NOT spam" on the Check tab.
• Marked as spam / Marked as not spam: how those corrections split by label. If almost all are "not spam", the model is being too strict. If almost all are "spam", it is missing too much.
• Waiting for retraining: corrections saved since the current model was trained. These have not been learned yet.
2. Lists recent corrections
A table of the last 20 corrections shows the date, the subject, and what the person marked it as. Use it to spot wrong or suspicious corrections before they reach the model.
3. Retrains the model
The Retrain model with corrections button does the following:
• It asks you to confirm, then trains a new model on the original examples plus all corrections.
• Corrections count 5 times more than ordinary examples, so a few of them can change the result.
• If a correction disagrees with an old example of the same email, the correction wins.
• It tests the new model on emails it did not train on and shows the results: legitimate emails wrongly blocked, spam missed, and the AUC (a quality score where 1.0 is perfect).
• It switches to the new model immediately, with no restart. The button is disabled when nothing is waiting.
4. Refreshes
The Refresh button reloads the numbers, for example if a colleague has added corrections.
Model versions tab
This tab is the history and the safety net. Every retrain creates a new saved version instead of overwriting the old one.
1. Lists every saved version
• Version: timestamp when it was trained. The one in use says "(active)".
• Trained: date and time of training.
• AUC: overall quality score. Higher is better.
• Wrongly blocked: legitimate emails the model flagged as spam in testing. This is the most important number, because blocking real mail is the costly mistake.
• Missed spam: spam the model let through in testing.
• Corrections used: how many feedback corrections went into that version.
2. Compares versions
After a retrain, compare the new row with the old one. If "Wrongly blocked" went up, the new version is worse for real mail.
3. Activates (rolls back) a version
Click Activate on any older version and confirm, and the system switches to it immediately. The chosen version is remembered after a restart.
A typical cycle
1.  On Check an email, the model marks a real tombola message as spam.
2.  You click This is NOT spam, and the correction is saved.
3.  On Feedback & learning, "Waiting for retraining" shows 1. Collect more corrections first, since a single one changes very little.
4.  You click Retrain, and a new version becomes active.
5.  On Model versions, you compare the new version with the old one. If "Wrongly blocked" got worse, you click Activate on the previous version.
Use several corrections of the same kind, such as 10 to 20 varied tombola messages, before retraining. A single example rarely teaches the model a lasting rule.