# Third party materials and publication boundaries

This package adds no blanket licence to third-party assets and assigns no new licence to the student's original code. The repository owner should select an appropriate code licence before inviting reuse.

- MARC: dataset mirror `buruzaemon/amazon_reviews_multi`, pinned revision in `data/provenance/downloads.json`. The retained dataset card declares `license: other`; the downloaded loader contains no specific licence text. Redistribution permission is not confirmed by this packaging process. Review bodies, titles, raw data and processed copies are kept local, not published here. Review IDs and experimental aggregates are retained for reproducibility. See the source links in `references/REFERENCES.md` and review current terms before use or redistribution.
- mBERT: `google-bert/bert-base-multilingual-cased`; XLM-R: `FacebookAI/xlm-roberta-base`. Source model cards, configurations and tokenizers are retained for attribution and compatibility. Exact revisions and hashes are recorded. Respect the upstream terms; this repository does not relicense them. Base and fine-tuned tensor weights are not part of the ordinary Git upload.
- Papers: bibliographic entries and links are in `references/REFERENCES.md`. Downloaded PDFs remain in the local archive, not this repository.
- Python dependencies: install using their upstream package distributions and licences. The environment and wheel installer are not redistributed here.

The local fine-tuned checkpoints are research artifacts. Before publishing them separately, review the base model terms and dataset conditions. A transfer channel such as LFS does not supply redistribution permission.
