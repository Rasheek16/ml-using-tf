import tarfile
from pathlib import Path
import urllib.request

def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path("data") / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)

    for dir_name, tar_name, url in (("easy_ham", "ham", ham_url), ("spam", "spam", spam_url)):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            with tarfile.open(path) as tar_bz2_file:
                tar_bz2_file.extractall(path=spam_path)

    return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]
