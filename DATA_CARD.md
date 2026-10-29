# Data Card — HateGuard Code-Mixed Hate Speech Moderation Corpus
This document summarizes the provenance, partition composition, cleaning protocols, limitations, and ethical risks of the corpus used in the **HateGuard AI** research.

---
## 1. Summary of Corpus Specifications
| Component | Description & Protocol |
| :--- | :--- |
| **Primary Task** | Binary hate speech classification (*Hate Speech* = `1` vs *Non-Hate Speech* = `0`) on Indonesian code-mixed social media text. |
| **Data Source 1** | Indonesian Twitter Corpus by Ibrohim & Budi (2019) (`re_dataset.csv`, SHA-256: `44c04e31ad4b7ee4a95f1884e7af4da2c44b69762143eb2de0ede7f90502735e`) and informal language normalization dictionary (`new_kamusalay.csv`, SHA-256: `fe00f0c728ee8ecc40dca3c3001ff2701dba7d4417a4178fe5b5745243564f9d`). Each unique tweet represents 1 independent group (`group_id`). |
| **Data Source 2** | Hugging Face Dataset `aiatums/codemixed-id-hate-speech` (*Indonesian-Javanese* and *Indonesian-Sundanese* configurations, CC BY 4.0 license, commit `19a03eba294171be64a5fcb5dcf7f36b6e86d72f`). Each record contains 1 original text (`original`) and 2 augmented variations (`augmented_1`, `augmented_2`) bound to the same `group_id`. |
| **Cleaning Procedure** | From 283,887 raw combined rows: (1) removal of empty text, (2) removal of 954 unique texts with contradictory cross-row labels (11,503 rows removed), and (3) deduplication based on `text_clean` (72,648 duplicates removed), yielding **199,360 clean rows** (111,761 Non-HS [56.06%] and 87,599 HS [43.94%]). |
| **Internal Partition Protocol** | Leak-free splitting using `GroupShuffleSplit` (`seed = 42`) based on `group_id`: **Train = 142,647 rows**, **Validation = 15,779 rows**, and **Internal Test = 40,934 rows (4,100 unique groups; consisting of 3,907 original texts and 37,027 augmented texts)**. |
| **Official Held-Out Test** | Extracted from the official `test` split of the Javanese and Sundanese datasets (original texts only), filtered for empty text, duplicates, and overlaps against all internal partitions, resulting in **1,995 original texts** (996 Javanese and 999 Sundanese). |
| **Normalization & Masking** | Model input normalization combines the alay dictionary (15,195 entries) and a manual 4-language lexicon (Indonesian, Slang English, Javanese, Sundanese). Public display censorship (*lexical masking*) uses a separate list of 47 sensitive words. |
---

## 2. Partition Composition Details and Class Prevalence

| Partition | Source Domain | Row Count | Unique Group Count (`group_id`) | Rows/Group Ratio | HS Prevalence (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `codemixed_javanese` (Aug) | 64,007 | 2,678 | 23.90 | 62.39% |
| **Train** | `codemixed_javanese_orig` | 2,677 | 2,677 | 1.00 | 62.38% |
| **Train** | `codemixed_sundanese` (Aug) | 64,565 | 2,698 | 23.93 | 26.30% |
| **Train** | `codemixed_sundanese_orig` | 2,016 | 2,016 | 1.00 | 18.15% |
| **Train** | `indonesian_twitter` (Orig) | 9,382 | 9,382 | 1.00 | 42.02% |
| **Validation** | `codemixed_javanese` (Aug) | 6,833 | 285 | 23.98 | 55.55% |
| **Validation** | `codemixed_javanese_orig` | 285 | 285 | 1.00 | 55.44% |
| **Validation** | `codemixed_sundanese` (Aug) | 7,384 | 309 | 23.90 | 22.85% |
| **Validation** | `codemixed_sundanese_orig` | 231 | 231 | 1.00 | 17.32% |
| **Validation** | `indonesian_twitter` (Orig) | 1,046 | 1,046 | 1.00 | 42.93% |
| **Internal Test** | `codemixed_javanese` (Aug) | 18,965 | 793 | 23.92 | 61.71% |
| **Internal Test** | `codemixed_javanese_orig` | 792 | 792 | 1.00 | 61.74% |
| **Internal Test** | `codemixed_sundanese` (Aug) | 18,062 | 754 | 23.95 | 28.61% |
| **Internal Test** | `codemixed_sundanese_orig` | 561 | 561 | 1.00 | 20.50% |
| **Internal Test** | `indonesian_twitter` (Orig) | 2,554 | 2,554 | 1.00 | 42.99% |
| **Official Held-Out** | `official_test_jv_orig` | 996 | 996 | 1.00 | 60.54% |
| **Official Held-Out** | `official_test_su_orig` | 999 | 999 | 1.00 | 34.53% |
---

## 3. Limitations & Ethical Risks
1. **Augmented Text Ratio to Independent Units:** In the *Internal Test* partition (40,934 rows), 90.5% of the rows (37,027 rows) consist of augmented variations originating from 1,546 code-mixed groups (~23.9 rows per group). Therefore, evaluation must utilize *grouped bootstrap* at the `group_id` level and report performance on the original text subset (`ORIGINAL_ONLY`, N = 3,907) as the primary benchmark.
2. **Augmentation Label Assumption:** Augmented variations are assumed to preserve the label of their original text (*label-preserving*) as provided by the source dataset, without manual re-verification by human annotators.
3. **Cross-Domain Class Prevalence Differences:** Hate speech class prevalence in this research corpus (~43.9%–45.4%, with Javanese ~61.7% and Sundanese original ~20.5%) is significantly higher than real-world social media traffic on average.
4. **Linguistic Scope & Annotation:** The corpus only covers variations of Indonesian, Slang English, Javanese, and Sundanese from the utilized sources, and thus does not automatically represent other regional dialects or different social media platforms.
5. **Handling Sensitive Content:** As the dataset contains real examples of hate speech and profanity, qualitative examples displayed in publications and user interfaces undergo lexical masking (*censored*).