# Adaptive-spam-filter-ai

## Running the service

Run the API from the repository root:

```bash
MODEL_DIR=models/v1 API_KEY=your-secret-key .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the web UI. The UI supports both email and SMS. The API accepts `POST /predict` requests with a JSON body such as:

```json
{
  "sender": "\"Enat Bank\" <not-enat@example.net>",
  "subject": "Please review",
  "body": "Message text",
  "channel": "email",
  "reply_to": "different-address@example.net",
  "headers": {"spf": "fail", "dkim": "pass", "dmarc": "fail"},
  "links": ["https://example.net/login"],
  "attachments": ["statement.xlsm"],
  "sender_seen_count": 2,
  "redirect_count": 1
}
```

The sender, authentication results, and other metadata are optional. Missing authentication results are treated as unknown, not as failures. Authentication results should come from a trusted mail gateway; do not treat values supplied directly by an untrusted client as verified headers. Attachment entries are filenames/metadata only; the service does not inspect attachment contents or determine password protection from the file itself.

### Sender rules

Set comma-separated email addresses or domains to override the model:

```bash
SPAM_ALLOWLIST=alerts@example.org,partner.example.org \
SPAM_BLOCKLIST=known-bad@example.net,bad-domain.example \
MODEL_DIR=models/v1 API_KEY=your-secret-key \
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

An allowlist entry can be an exact address or domain. A blocklist entry can likewise be an address or domain; blocklist rules take precedence. `enatbank.com` senders are currently trusted by default, so configure trusted upstream validation before using that rule with untrusted sender values.

### Existing model files

The updated service remains able to load older saved pipelines whose metadata feature vector has fewer columns. This avoids a feature-count error during prediction after upgrading the code. Older pipelines do not learn from the new sender/authentication/link/attachment signals, however; retrain a model using labeled data with those metadata columns to activate those signals. Restart the service after upgrading the code.

Retraining accepts optional columns in `data/emails.csv` and feedback data: `sender`, `channel`, `reply_to`, `headers`, `links`, `attachments`, `sender_seen_count`, and `redirect_count`. If omitted, these fields use empty or neutral values.

Feedback CSV files created with the earlier four-column schema (`subject`, `body`, `label`, `date`) remain supported. The reader also accepts appended metadata rows from an intermediate version with the old header. When new feedback is saved, the file is normalized to the current 12-column schema so retraining and feedback statistics can continue to read it.

## Feedback and model versions

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