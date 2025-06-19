from pathlib import Path
import email
import email.policy

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

def load_emails(email_dir: Path):
    return [
        load_email(f) for f in sorted(email_dir.iterdir()) if len(f.name) > 20
    ]
