# Bureaucracy Navigator — עוזר מסמכים ובירוקרטיה

A Telegram bot that explains Israeli bureaucracy in simple Hebrew, powered by a small Hebrew LLM (DictaLM 3.0 1.7B) fine-tuned for the job, with answers grounded in Kol Zchut articles.

> Work in progress: 3-day portfolio build. Full README (architecture, dataset, training, results) comes at the end.

## Data and license

The knowledge base is built from [Kol Zchut (כל-זכות)](https://www.kolzchut.org.il), whose content is licensed under [Creative Commons BY-NC-SA 2.5 IL](https://creativecommons.org/licenses/by-nc-sa/2.5/il/). This project is non-commercial, credits Kol Zchut, and links to the source articles in every answer. Data derived from Kol Zchut in this repository (e.g. `data/out/chunks.jsonl`) is shared under the same license.

The bot gives general information, not legal advice.
