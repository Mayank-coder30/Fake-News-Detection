"""Expand EvidenceLens Datasets — FEVER Claims, Evidence Corpus, and Baseline Fake News.

This script:
1. Downloads the official FEVER v1.0 dev split (~20K records) from AWS S3.
2. Creates a larger curated FEVER sample (default 500 records, balanced labels).
3. Expands the evidence corpus with additional Wikipedia-sourced passages.
4. Expands the baseline fake-news dataset with more REAL/FAKE articles.
5. Re-runs dataset preparation (label mapping) on the expanded sample.

Usage:
    python scripts/expand_dataset.py                     # Default 500 FEVER claims
    python scripts/expand_dataset.py --fever-size 1000   # Custom size
    python scripts/expand_dataset.py --skip-download      # Skip download if already done
"""

import json
import sys
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_DIR, SAMPLE_DATA_DIR, PROCESSED_DATA_DIR,
    LABEL_SUPPORTED, LABEL_REFUTED, LABEL_UNCERTAIN
)
from src.utils.logging_utils import get_logger

logger = get_logger("expand_dataset")

# ─────────────────────────────────────────────────────────────────────
# Step 1: Download FEVER official dev split
# ─────────────────────────────────────────────────────────────────────

FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"


def download_fever_split(url: str, dest_path: Path):
    """Downloads a FEVER split from the official fever.ai repository."""
    import urllib.request

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Already downloaded: {dest_path}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading FEVER split from {url} ...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            totalsize = int(response.headers.get("Content-Length", 0))
            blocksize = 8192
            downloaded = 0
            while True:
                chunk = response.read(blocksize)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if totalsize > 0 and (downloaded // blocksize) % 100 == 0:
                    percent = min(100.0, downloaded * 100 / totalsize)
                    mb = downloaded / (1024 * 1024)
                    print(f"\rProgress: {percent:.1f}% ({mb:.1f} MB)", end="", flush=True)
            print()
        logger.info(f"Successfully downloaded to: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Step 2: Create larger curated FEVER sample with balanced labels
# ─────────────────────────────────────────────────────────────────────

def create_curated_fever_sample(raw_file: Path, output_file: Path, target_size: int = 500):
    """
    Reads the full official FEVER JSONL, stratifies by label, and creates
    a balanced curated sample of `target_size` records.
    """
    logger.info(f"Loading raw FEVER data from {raw_file} ...")

    # Read and bucket by label
    buckets: Dict[str, List[Dict]] = {
        "SUPPORTS": [],
        "REFUTES": [],
        "NOT ENOUGH INFO": []
    }

    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                label = rec.get("label", "").strip()
                if label in buckets:
                    buckets[label].append(rec)
            except json.JSONDecodeError:
                continue

    for label, records in buckets.items():
        logger.info(f"  {label}: {len(records)} records available")

    # Balanced sampling: split target evenly, remainder goes to SUPPORTS
    per_class = target_size // 3
    remainder = target_size - (per_class * 3)

    sampled = []
    random.seed(42)  # reproducible

    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        n = per_class + (remainder if label == "SUPPORTS" else 0)
        pool = buckets[label]
        if len(pool) < n:
            logger.warning(f"  Only {len(pool)} records for {label}, using all.")
            sampled.extend(pool)
        else:
            sampled.extend(random.sample(pool, n))

    random.shuffle(sampled)

    # Re-number IDs sequentially
    for i, rec in enumerate(sampled, start=1001):
        rec["id"] = i

    # Write curated sample
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    label_counts = {}
    for rec in sampled:
        lbl = rec.get("label", "UNKNOWN")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    logger.info(f"Curated sample saved: {len(sampled)} records -> {output_file}")
    logger.info(f"  Label distribution: {label_counts}")
    return len(sampled)


# ─────────────────────────────────────────────────────────────────────
# Step 3: Expand the evidence corpus with more Wikipedia passages
# ─────────────────────────────────────────────────────────────────────

ADDITIONAL_EVIDENCE_PASSAGES = [
    {
        "passage_id": "doc_021", "title": "Amazon (company)",
        "text": "Amazon.com, Inc. is an American multinational technology company focusing on e-commerce, cloud computing (AWS), online advertising, digital streaming, and artificial intelligence. Founded by Jeff Bezos in 1994, it is one of the Big Five American information technology companies.",
        "source_id": "wiki_amazon", "source_url": "https://en.wikipedia.org/wiki/Amazon_(company)"
    },
    {
        "passage_id": "doc_022", "title": "Artificial intelligence",
        "text": "Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence of living beings, primarily humans. It is a field of study in computer science that develops and studies intelligent machines. Major subfields include machine learning, natural language processing, computer vision, and robotics.",
        "source_id": "wiki_ai", "source_url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
    },
    {
        "passage_id": "doc_023", "title": "Bitcoin",
        "text": "Bitcoin is a decentralized digital currency that can be transferred on the peer-to-peer bitcoin network. Bitcoin transactions are verified by network nodes through cryptography and recorded in a distributed public ledger called a blockchain. It was invented in 2008 by an unknown person or group using the pseudonym Satoshi Nakamoto.",
        "source_id": "wiki_bitcoin", "source_url": "https://en.wikipedia.org/wiki/Bitcoin"
    },
    {
        "passage_id": "doc_024", "title": "Human trafficking",
        "text": "Human trafficking is the trade of humans for the purpose of forced labour, sexual slavery, or commercial sexual exploitation. It is considered a form of modern slavery and is a serious violation of human rights, affecting millions of people worldwide according to the International Labour Organization.",
        "source_id": "wiki_human_trafficking", "source_url": "https://en.wikipedia.org/wiki/Human_trafficking"
    },
    {
        "passage_id": "doc_025", "title": "International Space Station",
        "text": "The International Space Station (ISS) is a modular space station in low Earth orbit. It is a multinational collaborative project involving five space agencies: NASA, Roscosmos, JAXA, ESA, and CSA. The ISS orbits Earth approximately every 90 minutes at an altitude of about 400 kilometres.",
        "source_id": "wiki_iss", "source_url": "https://en.wikipedia.org/wiki/International_Space_Station"
    },
    {
        "passage_id": "doc_026", "title": "Vaccine",
        "text": "A vaccine is a biological preparation that provides active acquired immunity to a particular infectious or malignant disease. The safety and effectiveness of vaccines has been widely studied and verified. A vaccine typically contains an agent that resembles a disease-causing microorganism and is often made from weakened or killed forms of the microbe.",
        "source_id": "wiki_vaccine", "source_url": "https://en.wikipedia.org/wiki/Vaccine"
    },
    {
        "passage_id": "doc_027", "title": "Solar System",
        "text": "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. The largest of such objects are the eight planets, in order from the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. It formed 4.6 billion years ago from the gravitational collapse of a giant molecular cloud.",
        "source_id": "wiki_solar_system", "source_url": "https://en.wikipedia.org/wiki/Solar_System"
    },
    {
        "passage_id": "doc_028", "title": "World War II",
        "text": "World War II was a global conflict that lasted from 1939 to 1945. It involved the vast majority of the world's countries, including all the great powers, forming two opposing military alliances: the Allies and the Axis powers. It was the deadliest conflict in human history, resulting in 70 to 85 million fatalities.",
        "source_id": "wiki_ww2", "source_url": "https://en.wikipedia.org/wiki/World_War_II"
    },
    {
        "passage_id": "doc_029", "title": "DNA",
        "text": "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. DNA carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms and many viruses. It was first identified by Friedrich Miescher in 1869.",
        "source_id": "wiki_dna", "source_url": "https://en.wikipedia.org/wiki/DNA"
    },
    {
        "passage_id": "doc_030", "title": "United Nations",
        "text": "The United Nations (UN) is an intergovernmental organization whose stated purposes are maintaining international peace and security, developing friendly relations among nations, and achieving international cooperation. It was established on 24 October 1945, after World War II, with the aim of preventing future wars.",
        "source_id": "wiki_un", "source_url": "https://en.wikipedia.org/wiki/United_Nations"
    },
    {
        "passage_id": "doc_031", "title": "Large Hadron Collider",
        "text": "The Large Hadron Collider (LHC) is the world's largest and highest-energy particle collider. It was built by the European Organization for Nuclear Research (CERN) between 1998 and 2008, situated beneath the France-Switzerland border near Geneva. The LHC confirmed the existence of the Higgs boson in 2012.",
        "source_id": "wiki_lhc", "source_url": "https://en.wikipedia.org/wiki/Large_Hadron_Collider"
    },
    {
        "passage_id": "doc_032", "title": "Penicillin",
        "text": "Penicillin is a group of antibiotics that are widely used to treat bacterial infections. It was discovered by Alexander Fleming in 1928 when he noticed that a mold called Penicillium notatum had contaminated a culture plate and killed the surrounding bacteria. This discovery revolutionized modern medicine.",
        "source_id": "wiki_penicillin", "source_url": "https://en.wikipedia.org/wiki/Penicillin"
    },
    {
        "passage_id": "doc_033", "title": "Quantum computing",
        "text": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena such as superposition and entanglement. A quantum computer uses quantum bits (qubits) which can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1. Major companies like IBM, Google, and Microsoft are actively developing quantum processors.",
        "source_id": "wiki_quantum_computing", "source_url": "https://en.wikipedia.org/wiki/Quantum_computing"
    },
    {
        "passage_id": "doc_034", "title": "Internet",
        "text": "The Internet is a global system of interconnected computer networks that uses the TCP/IP protocol suite to communicate between networks and devices. It originated in the United States as ARPANET in the late 1960s and became commercially available in the early 1990s. Tim Berners-Lee invented the World Wide Web in 1989.",
        "source_id": "wiki_internet", "source_url": "https://en.wikipedia.org/wiki/Internet"
    },
    {
        "passage_id": "doc_035", "title": "Photosynthesis",
        "text": "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose. The process primarily takes place in the chloroplasts of plant cells using chlorophyll, water, and carbon dioxide, releasing oxygen as a byproduct.",
        "source_id": "wiki_photosynthesis", "source_url": "https://en.wikipedia.org/wiki/Photosynthesis"
    },
    {
        "passage_id": "doc_036", "title": "Human genome",
        "text": "The human genome is the complete set of nucleic acid sequences for humans, encoded as DNA within the 23 chromosome pairs in cell nuclei. The Human Genome Project, completed in 2003, determined the sequence of the entire human genome, identifying approximately 20,000-25,000 protein-coding genes.",
        "source_id": "wiki_human_genome", "source_url": "https://en.wikipedia.org/wiki/Human_genome"
    },
    {
        "passage_id": "doc_037", "title": "Hubble Space Telescope",
        "text": "The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990 and remains in operation. Named after astronomer Edwin Hubble, it is one of NASA's Great Observatories. Hubble has made more than 1.5 million observations and has been instrumental in numerous astronomical discoveries.",
        "source_id": "wiki_hubble", "source_url": "https://en.wikipedia.org/wiki/Hubble_Space_Telescope"
    },
    {
        "passage_id": "doc_038", "title": "CRISPR gene editing",
        "text": "CRISPR gene editing is a genetic engineering technique in molecular biology by which the genomes of living organisms may be modified. It is based on a simplified version of the bacterial CRISPR-Cas9 antiviral defense system. Emmanuelle Charpentier and Jennifer Doudna were awarded the 2020 Nobel Prize in Chemistry for developing this technology.",
        "source_id": "wiki_crispr", "source_url": "https://en.wikipedia.org/wiki/CRISPR_gene_editing"
    },
    {
        "passage_id": "doc_039", "title": "Black hole",
        "text": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light or other electromagnetic waves, has enough energy to escape. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole. The first image of a black hole was captured by the Event Horizon Telescope in 2019.",
        "source_id": "wiki_black_hole", "source_url": "https://en.wikipedia.org/wiki/Black_hole"
    },
    {
        "passage_id": "doc_040", "title": "Mahatma Gandhi",
        "text": "Mohandas Karamchand Gandhi, honorifically known as Mahatma Gandhi, was an Indian lawyer, anti-colonial nationalist, and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule. He inspired movements for civil rights and freedom across the world.",
        "source_id": "wiki_gandhi", "source_url": "https://en.wikipedia.org/wiki/Mahatma_Gandhi"
    },
    {
        "passage_id": "doc_041", "title": "Titanic",
        "text": "RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean on 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of the deadliest peacetime maritime disasters in history.",
        "source_id": "wiki_titanic", "source_url": "https://en.wikipedia.org/wiki/Titanic"
    },
    {
        "passage_id": "doc_042", "title": "Mount Everest",
        "text": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border runs across its summit point. Its peak elevation is 8,848.86 metres above sea level. Tenzing Norgay and Edmund Hillary were the first confirmed climbers to reach the summit in 1953.",
        "source_id": "wiki_everest", "source_url": "https://en.wikipedia.org/wiki/Mount_Everest"
    },
    {
        "passage_id": "doc_043", "title": "Mars",
        "text": "Mars is the fourth planet from the Sun. It is often called the 'Red Planet' because iron oxide on its surface gives it a reddish appearance. Mars has two small natural satellites, Phobos and Deimos. The planet has a thin atmosphere composed primarily of carbon dioxide, with surface temperatures ranging from −143°C to 35°C.",
        "source_id": "wiki_mars", "source_url": "https://en.wikipedia.org/wiki/Mars"
    },
    {
        "passage_id": "doc_044", "title": "Periodic table",
        "text": "The periodic table is a tabular display of the chemical elements organized by atomic number, electron configuration, and recurring chemical properties. Dmitri Mendeleev is generally credited with the creation of the first periodic table in 1869. As of 2023, the table contains 118 confirmed elements.",
        "source_id": "wiki_periodic_table", "source_url": "https://en.wikipedia.org/wiki/Periodic_table"
    },
    {
        "passage_id": "doc_045", "title": "Nelson Mandela",
        "text": "Nelson Rolihlahla Mandela was a South African anti-apartheid activist and politician who served as the first president of South Africa from 1994 to 1999. He was imprisoned for 27 years for his activism, and after his release became a symbol of the fight against racial oppression. He was awarded the Nobel Peace Prize in 1993.",
        "source_id": "wiki_mandela", "source_url": "https://en.wikipedia.org/wiki/Nelson_Mandela"
    },
    {
        "passage_id": "doc_046", "title": "Electricity",
        "text": "Electricity is the set of physical phenomena associated with the presence and motion of matter that has a property of electric charge. Electricity is related to magnetism, both being part of the phenomenon of electromagnetism. Benjamin Franklin's famous kite experiment in 1752 demonstrated the electrical nature of lightning.",
        "source_id": "wiki_electricity", "source_url": "https://en.wikipedia.org/wiki/Electricity"
    },
    {
        "passage_id": "doc_047", "title": "Amazon rainforest",
        "text": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km² of which 5,500,000 km² are covered by the rainforest. It represents over half of the planet's remaining rainforests.",
        "source_id": "wiki_amazon_rainforest", "source_url": "https://en.wikipedia.org/wiki/Amazon_rainforest"
    },
    {
        "passage_id": "doc_048", "title": "William Shakespeare",
        "text": "William Shakespeare was an English playwright, poet, and actor. He is widely regarded as the greatest writer in the English language and the world's greatest dramatist. He wrote approximately 39 plays, 154 sonnets, and several longer poems. His plays have been translated into every major language and are performed more often than those of any other playwright.",
        "source_id": "wiki_shakespeare", "source_url": "https://en.wikipedia.org/wiki/William_Shakespeare"
    },
    {
        "passage_id": "doc_049", "title": "Gravity",
        "text": "Gravity is a fundamental interaction that causes mutual attraction between all things that have mass. It is by far the weakest of the four fundamental interactions. Isaac Newton's law of universal gravitation, published in 1687, describes gravity as a force. Albert Einstein's general theory of relativity (1915) describes it as a consequence of the curvature of spacetime.",
        "source_id": "wiki_gravity", "source_url": "https://en.wikipedia.org/wiki/Gravity"
    },
    {
        "passage_id": "doc_050", "title": "Malaria",
        "text": "Malaria is a mosquito-borne infectious disease that affects humans and other animals. It is caused by single-celled microorganisms of the Plasmodium group, spread through bites of infected female Anopheles mosquitoes. According to the WHO, there were an estimated 247 million cases and 619,000 deaths from malaria worldwide in 2021.",
        "source_id": "wiki_malaria", "source_url": "https://en.wikipedia.org/wiki/Malaria"
    },
]


def expand_evidence_corpus(existing_file: Path, output_file: Path):
    """Appends additional evidence passages to the existing corpus."""
    existing_passages = []
    if existing_file.exists():
        with open(existing_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_passages.append(json.loads(line))

    existing_ids = {p["passage_id"] for p in existing_passages}
    new_passages = [p for p in ADDITIONAL_EVIDENCE_PASSAGES if p["passage_id"] not in existing_ids]

    all_passages = existing_passages + new_passages

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Evidence corpus expanded: {len(existing_passages)} -> {len(all_passages)} passages ({len(new_passages)} new)")
    return len(all_passages)


# ─────────────────────────────────────────────────────────────────────
# Step 4: Expand baseline fake news dataset
# ─────────────────────────────────────────────────────────────────────

ADDITIONAL_BASELINE_ARTICLES = [
    # REAL articles
    {"id": "base_026", "text": "The European Space Agency successfully launches the Euclid space telescope to map the dark universe and study dark energy.", "label": "REAL"},
    {"id": "base_027", "text": "New Zealand becomes the first country to require financial institutions to disclose climate-related financial risks.", "label": "REAL"},
    {"id": "base_028", "text": "Scientists at DeepMind develop AlphaFold 2, solving the protein structure prediction problem with remarkable accuracy.", "label": "REAL"},
    {"id": "base_029", "text": "The UN General Assembly adopts a global resolution on artificial intelligence promoting safe and trustworthy AI systems.", "label": "REAL"},
    {"id": "base_030", "text": "India successfully launches the Chandrayaan-3 mission, making a soft landing near the lunar south pole.", "label": "REAL"},
    {"id": "base_031", "text": "Ocean surface temperatures reach record highs according to data from NOAA and the European Copernicus Climate Service.", "label": "REAL"},
    {"id": "base_032", "text": "Researchers demonstrate a room-temperature semiconductor that could transform the electronics industry, published in Nature.", "label": "REAL"},
    {"id": "base_033", "text": "The International Energy Agency reports global solar energy installations exceeded 300 gigawatts of new capacity in 2023.", "label": "REAL"},
    {"id": "base_034", "text": "Denmark announces plans to build the world's first energy island in the North Sea to generate offshore wind power.", "label": "REAL"},
    {"id": "base_035", "text": "Archaeologists uncover a previously unknown section of the Dead Sea Scrolls in a cave near Qumran.", "label": "REAL"},
    {"id": "base_036", "text": "The World Bank issues new report showing global extreme poverty rate fell below eight percent for the first time.", "label": "REAL"},
    {"id": "base_037", "text": "A team of marine biologists discovers a new species of deep-sea coral in the Mariana Trench at record depths.", "label": "REAL"},
    {"id": "base_038", "text": "Switzerland votes by referendum to adopt net-zero carbon emissions target by 2050.", "label": "REAL"},
    # FAKE articles
    {"id": "base_039", "text": "Government admits to spraying mind-altering chemicals in commercial airline contrails to reduce population growth.", "label": "FAKE"},
    {"id": "base_040", "text": "Scientists discover that the Earth is actually flat according to leaked classified documents from top research labs.", "label": "FAKE"},
    {"id": "base_041", "text": "Eating only raw crystals for thirty days will completely detoxify your liver and reverse aging by twenty years.", "label": "FAKE"},
    {"id": "base_042", "text": "Secret underground tunnels connecting all world capitals discovered, built by ancient alien civilization.", "label": "FAKE"},
    {"id": "base_043", "text": "Smartphone screens emit invisible radiation that causes instant permanent brain damage if used more than one hour daily.", "label": "FAKE"},
    {"id": "base_044", "text": "A farmer in rural Ohio discovers a fully functional nuclear reactor buried in his backyard dating to the Roman Empire.", "label": "FAKE"},
    {"id": "base_045", "text": "Major tech company CEO reveals their products contain hidden microphones that record all private conversations.", "label": "FAKE"},
    {"id": "base_046", "text": "Scientists prove that all dinosaurs were actually robots created by a prehistoric human civilization.", "label": "FAKE"},
    {"id": "base_047", "text": "Mixing common household bleach with orange juice creates a superfood that cures every known disease instantly.", "label": "FAKE"},
    {"id": "base_048", "text": "The moon is actually a hologram projected by a secret space station to hide alien mining operations.", "label": "FAKE"},
    {"id": "base_049", "text": "World's richest billionaire admits to controlling all weather patterns using orbital microwave satellites.", "label": "FAKE"},
    {"id": "base_050", "text": "Walking barefoot on wet grass for exactly seven minutes at midnight will permanently cure all allergies.", "label": "FAKE"},
    {"id": "base_051", "text": "Top university proves that listening to classical music while sleeping doubles your IQ permanently after one night.", "label": "FAKE"},
]


def expand_baseline_dataset(existing_file: Path, output_file: Path):
    """Appends additional REAL/FAKE articles to baseline dataset."""
    existing = []
    if existing_file.exists():
        with open(existing_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    existing_ids = {r["id"] for r in existing}
    new_articles = [a for a in ADDITIONAL_BASELINE_ARTICLES if a["id"] not in existing_ids]

    all_articles = existing + new_articles

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for a in all_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    real_count = sum(1 for a in all_articles if a["label"] == "REAL")
    fake_count = sum(1 for a in all_articles if a["label"] == "FAKE")
    logger.info(f"Baseline dataset expanded: {len(existing)} -> {len(all_articles)} articles ({real_count} REAL, {fake_count} FAKE)")
    return len(all_articles)


# ─────────────────────────────────────────────────────────────────────
# Step 5: Re-prepare (standardize labels) for the expanded sample
# ─────────────────────────────────────────────────────────────────────

FEVER_LABEL_MAPPING = {
    "SUPPORTS": LABEL_SUPPORTED,
    "REFUTES": LABEL_REFUTED,
    "NOT ENOUGH INFO": LABEL_UNCERTAIN
}


def prepare_expanded_fever(input_file: Path, output_file: Path):
    """Standardize labels on the expanded FEVER sample (mirrors prepare_dataset.py logic)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "valid": 0, "skipped": 0}
    label_counts = {}
    processed = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                rec = json.loads(line)
                claim_id = rec.get("id")
                claim_text = rec.get("claim", "").strip()
                raw_label = rec.get("label", "").strip()

                if not claim_id or not claim_text or not raw_label:
                    stats["skipped"] += 1
                    continue

                mapped_label = FEVER_LABEL_MAPPING.get(raw_label.upper())
                if not mapped_label:
                    stats["skipped"] += 1
                    continue

                # Process evidence references
                evidence_entries = []
                raw_evidence = rec.get("evidence", [])
                if isinstance(raw_evidence, list):
                    for ev in raw_evidence:
                        if isinstance(ev, list) and len(ev) > 0:
                            doc_title = ev[2] if len(ev) > 2 and ev[2] is not None else None
                            line_no = ev[3] if len(ev) > 3 and ev[3] is not None else None
                            if doc_title:
                                evidence_entries.append({
                                    "doc_title": str(doc_title),
                                    "line_number": line_no
                                })

                processed.append({
                    "id": claim_id,
                    "claim": claim_text,
                    "label": mapped_label,
                    "original_label": raw_label,
                    "evidence_count": len(evidence_entries),
                    "evidence_refs": evidence_entries
                })
                stats["valid"] += 1
                label_counts[mapped_label] = label_counts.get(mapped_label, 0) + 1

            except Exception:
                stats["skipped"] += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Processed {stats['valid']} records -> {output_file}")
    logger.info(f"  Label distribution: {label_counts}")
    logger.info(f"  Skipped: {stats['skipped']}")
    return stats


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expand EvidenceLens datasets")
    parser.add_argument("--fever-size", type=int, default=500,
                        help="Number of FEVER records to sample (default: 500)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading FEVER from S3")
    parser.add_argument("--use-train", action="store_true",
                        help="Download and sample from train split instead of dev")
    args = parser.parse_args()

    print("=" * 60)
    print("EVIDENCELENS DATASET EXPANSION")
    print("=" * 60)

    # ── Step 1: Download ──
    if not args.skip_download:
        if args.use_train:
            raw_file = RAW_DATA_DIR / "train.jsonl"
            download_fever_split(FEVER_TRAIN_URL, raw_file)
        else:
            raw_file = RAW_DATA_DIR / "dev.jsonl"
            download_fever_split(FEVER_DEV_URL, raw_file)
    else:
        raw_file = RAW_DATA_DIR / ("train.jsonl" if args.use_train else "dev.jsonl")
        if not raw_file.exists():
            logger.error(f"Raw file not found: {raw_file}. Run without --skip-download first.")
            return

    # ── Step 2: Create curated FEVER sample ──
    fever_sample_out = SAMPLE_DATA_DIR / "fever_sample.jsonl"
    if raw_file.exists():
        create_curated_fever_sample(raw_file, fever_sample_out, target_size=args.fever_size)
    else:
        logger.error(f"Cannot create curated sample — raw file missing: {raw_file}")

    # ── Step 3: Expand evidence corpus ──
    evidence_file = SAMPLE_DATA_DIR / "evidence_corpus.jsonl"
    expand_evidence_corpus(evidence_file, evidence_file)

    # ── Step 4: Expand baseline dataset ──
    baseline_file = SAMPLE_DATA_DIR / "baseline_fake_news.jsonl"
    expand_baseline_dataset(baseline_file, baseline_file)

    # ── Step 5: Re-prepare processed FEVER ──
    processed_out = PROCESSED_DATA_DIR / "fever_processed.jsonl"
    prepare_expanded_fever(fever_sample_out, processed_out)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("EXPANSION COMPLETE — Summary")
    print("=" * 60)

    for label, path in [
        ("FEVER Sample", fever_sample_out),
        ("FEVER Processed", processed_out),
        ("Evidence Corpus", evidence_file),
        ("Baseline Dataset", baseline_file),
    ]:
        if path.exists():
            count = sum(1 for _ in open(path, "r", encoding="utf-8") if _.strip())
            size_kb = path.stat().st_size / 1024
            print(f"  {label:<20} {count:>6} records  ({size_kb:.1f} KB)  -> {path}")
        else:
            print(f"  {label:<20}  MISSING  -> {path}")

    print("=" * 60)
    print("\nNext steps:")
    print("  1. Rebuild evidence index:  python scripts/build_evidence_index.py")
    print("  2. Retrain baseline:        python scripts/train_baseline.py")
    print("  3. Run evaluation:          python scripts/run_evaluation.py")


if __name__ == "__main__":
    main()

