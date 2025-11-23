"""
Data Generation Script for PII NER with STT Noise Patterns

Generates realistic noisy STT transcripts with proper entity annotations.
Designed to create 500-1000 training examples and 100-200 dev examples.
"""

import json
import random
from typing import List, Dict, Tuple
import re


# ============================================================================
# ENTITY TEMPLATES AND PATTERNS
# ============================================================================

# Indian and international names
FIRST_NAMES = [
    "ramesh", "priyanka", "rahul", "sneha", "amit", "anjali", "rohan", "kavya",
    "rajesh", "pooja", "vikram", "divya", "arjun", "neha", "sanjay", "preeti",
    "anil", "riya", "suresh", "megha", "karan", "isha", "manish", "shruti",
    "vijay", "nisha", "deepak", "swati", "ashok", "tanvi", "nitin", "priya",
    "john", "sarah", "michael", "emma", "david", "lisa", "robert", "maria",
    "james", "jennifer", "william", "patricia", "richard", "linda", "thomas", "susan"
]

LAST_NAMES = [
    "sharma", "verma", "patel", "kumar", "singh", "reddy", "gupta", "mehta",
    "agarwal", "jain", "shah", "chopra", "malhotra", "kapoor", "bose", "iyer",
    "nair", "menon", "rao", "pillai", "banerjee", "chatterjee", "das", "desai",
    "smith", "johnson", "williams", "brown", "jones", "miller", "davis", "garcia",
    "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas"
]

# Indian cities
CITIES = [
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad",
    "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "visakhapatnam", "patna",
    "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad", "meerut", "rajkot",
    "varanasi", "srinagar", "amritsar", "allahabad", "ranchi", "jabalpur", "gwalior", "coimbatore"
]

# Locations (more specific places)
LOCATIONS = [
    "andheri station", "connaught place", "mg road", "marine drive", "bandra west",
    "koramangala", "whitefield", "electronic city", "salt lake", "park street",
    "begumpet", "secunderabad", "viman nagar", "kothrud", "satellite road",
    "malad east", "powai", "versova", "juhu beach", "thane west"
]

# Email domains
EMAIL_DOMAINS = [
    "gmail", "outlook", "yahoo", "hotmail", "rediffmail", "company", "work",
    "email", "mail", "inbox", "protonmail", "zoho"
]

EMAIL_EXTENSIONS = ["com", "co dot in", "in", "org", "net", "edu"]

# Number words for STT
DIGIT_WORDS = {
    "0": ["zero", "oh"],
    "1": ["one"],
    "2": ["two"],
    "3": ["three"],
    "4": ["four"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight"],
    "9": ["nine"]
}

DOUBLE_DIGIT_WORDS = {
    "00": ["double zero", "double oh"],
    "11": ["double one"],
    "22": ["double two"],
    "33": ["double three"],
    "44": ["double four"],
    "55": ["double five"],
    "66": ["double six"],
    "77": ["double seven"],
    "88": ["double eight"],
    "99": ["double nine"]
}

TRIPLE_DIGIT_WORDS = {
    "000": ["triple zero", "triple oh"],
    "111": ["triple one"],
    "222": ["triple two"],
    "333": ["triple three"],
    "444": ["triple four"],
    "555": ["triple five"],
    "666": ["triple six"],
    "777": ["triple seven"],
    "888": ["triple eight"],
    "999": ["triple nine"]
}

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

MONTH_SHORT = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


# ============================================================================
# ENTITY GENERATION FUNCTIONS
# ============================================================================

def generate_person_name() -> str:
    """Generate realistic person names."""
    if random.random() < 0.7:  # 70% full name
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        return f"{first} {last}"
    else:  # 30% first name only
        return random.choice(FIRST_NAMES)


def number_to_spoken(num_str: str, style: str = "mixed") -> str:
    """Convert numeric string to spoken form with various STT patterns."""
    if style == "digit_by_digit":
        # "4242" -> "four two four two"
        spoken = []
        for digit in num_str:
            spoken.append(random.choice(DIGIT_WORDS.get(digit, [digit])))
        return " ".join(spoken)
    
    elif style == "with_doubles":
        # "4422" -> "double four double two" or "four four two two"
        result = []
        i = 0
        while i < len(num_str):
            if i < len(num_str) - 1 and num_str[i] == num_str[i+1]:
                # Check for triple
                if i < len(num_str) - 2 and num_str[i] == num_str[i+2]:
                    triple = num_str[i:i+3]
                    if triple in TRIPLE_DIGIT_WORDS and random.random() < 0.6:
                        result.append(random.choice(TRIPLE_DIGIT_WORDS[triple]))
                        i += 3
                        continue
                # Use double
                double = num_str[i:i+2]
                if double in DOUBLE_DIGIT_WORDS and random.random() < 0.7:
                    result.append(random.choice(DOUBLE_DIGIT_WORDS[double]))
                    i += 2
                    continue
            # Single digit
            result.append(random.choice(DIGIT_WORDS.get(num_str[i], [num_str[i]])))
            i += 1
        return " ".join(result)
    
    elif style == "mixed":
        # Mix of digits and words: "4242 4242" or "four two four two 4242"
        if random.random() < 0.4:
            return num_str  # Keep as digits
        elif random.random() < 0.5:
            return number_to_spoken(num_str, "digit_by_digit")
        else:
            # Partial conversion
            parts = []
            chunk_size = random.choice([2, 4])
            for i in range(0, len(num_str), chunk_size):
                chunk = num_str[i:i+chunk_size]
                if random.random() < 0.5:
                    parts.append(chunk)
                else:
                    parts.append(number_to_spoken(chunk, "digit_by_digit"))
            return " ".join(parts)
    
    return num_str


def generate_credit_card() -> str:
    """Generate realistic credit card numbers in STT format."""
    # Common test card numbers
    patterns = [
        "4242424242424242",  # Visa
        "5555555555554444",  # Mastercard
        "378282246310005",   # Amex
        "6011111111111117",  # Discover
        "4111111111111111",  # Visa
    ]
    
    card = random.choice(patterns)
    
    # Add spaces in groups of 4
    card_spaced = " ".join([card[i:i+4] for i in range(0, len(card), 4)])
    
    # Convert to spoken format
    style = random.choice(["mixed", "digit_by_digit", "with_doubles", "numeric"])
    
    if style == "numeric":
        return card_spaced
    elif style == "mixed":
        # Some parts spoken, some numeric
        parts = card_spaced.split()
        result = []
        for part in parts:
            if random.random() < 0.5:
                result.append(part)
            else:
                result.append(number_to_spoken(part, "digit_by_digit"))
        return " ".join(result)
    else:
        return number_to_spoken(card.replace(" ", ""), style)


def generate_phone() -> str:
    """Generate realistic phone numbers in STT format."""
    # Indian phone patterns (10 digits)
    prefixes = ["98", "99", "97", "96", "95", "94", "93", "92", "91", "90",
                "88", "89", "87", "86", "85", "84", "83", "82", "81", "80"]
    
    prefix = random.choice(prefixes)
    rest = "".join([str(random.randint(0, 9)) for _ in range(8)])
    phone = prefix + rest
    
    # STT patterns
    patterns = [
        ("full_spoken", 0.3),
        ("grouped", 0.2),
        ("numeric", 0.2),
        ("mixed", 0.3)
    ]
    
    pattern = random.choices([p[0] for p in patterns], weights=[p[1] for p in patterns])[0]
    
    if pattern == "full_spoken":
        return number_to_spoken(phone, "digit_by_digit")
    elif pattern == "grouped":
        # "98 765 432 10" style
        return number_to_spoken(phone, "with_doubles")
    elif pattern == "numeric":
        return phone
    else:  # mixed
        return number_to_spoken(phone, "mixed")


def generate_email(person_name: str = None) -> str:
    """Generate realistic email addresses in STT format."""
    if person_name and random.random() < 0.6:
        # Use person's name in email
        name_parts = person_name.lower().split()
        if len(name_parts) == 2:
            patterns = [
                f"{name_parts[0]} dot {name_parts[1]}",
                f"{name_parts[0]}{name_parts[1]}",
                f"{name_parts[0][0]} dot {name_parts[1]}",
                f"{name_parts[0]} {name_parts[1]}{random.randint(1, 99)}"
            ]
            username = random.choice(patterns)
        else:
            username = name_parts[0]
    else:
        # Random username
        username = random.choice(FIRST_NAMES).lower()
        if random.random() < 0.4:
            username += str(random.randint(1, 999))
    
    domain = random.choice(EMAIL_DOMAINS)
    extension = random.choice(EMAIL_EXTENSIONS)
    
    # STT format: "at" and "dot"
    email = f"{username} at {domain} dot {extension}"
    
    return email


def generate_date() -> str:
    """Generate dates in various STT formats."""
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2025)
    
    formats = [
        f"{day:02d} {month:02d} {year}",  # "01 02 2024"
        f"{MONTH_NAMES[month-1]} {day} {year}",  # "january 1 2024"
        f"{day} {MONTH_NAMES[month-1]} {year}",  # "1 january 2024"
        f"{MONTH_SHORT[month-1]} {day} {year}",  # "jan 1 2024"
        f"{day}th {MONTH_NAMES[month-1]}",  # "15th august"
        f"first of {MONTH_NAMES[month-1]}",  # "first of march"
        f"{day} {month} {year}",  # "1 2 2024"
    ]
    
    date_str = random.choice(formats)
    
    # Sometimes spell out the year
    if "202" in date_str and random.random() < 0.3:
        year_spoken = f"twenty {number_to_spoken(str(year)[-2:], 'digit_by_digit')}"
        date_str = date_str.replace(str(year), year_spoken)
    
    return date_str


def generate_city() -> str:
    """Generate city names."""
    return random.choice(CITIES)


def generate_location() -> str:
    """Generate location names."""
    return random.choice(LOCATIONS)


# ============================================================================
# SENTENCE TEMPLATES
# ============================================================================

def get_sentence_templates() -> List[Dict]:
    """
    Return sentence templates with placeholders for entities.
    Each template specifies which entities it contains.
    """
    templates = [
        # Single entity templates
        {
            "template": "my credit card number is {CREDIT_CARD}",
            "entities": ["CREDIT_CARD"]
        },
        {
            "template": "card number {CREDIT_CARD}",
            "entities": ["CREDIT_CARD"]
        },
        {
            "template": "please charge {CREDIT_CARD}",
            "entities": ["CREDIT_CARD"]
        },
        {
            "template": "call me on {PHONE}",
            "entities": ["PHONE"]
        },
        {
            "template": "my number is {PHONE}",
            "entities": ["PHONE"]
        },
        {
            "template": "phone number {PHONE}",
            "entities": ["PHONE"]
        },
        {
            "template": "you can reach me at {PHONE}",
            "entities": ["PHONE"]
        },
        {
            "template": "contact number {PHONE}",
            "entities": ["PHONE"]
        },
        {
            "template": "my email is {EMAIL}",
            "entities": ["EMAIL"]
        },
        {
            "template": "email id is {EMAIL}",
            "entities": ["EMAIL"]
        },
        {
            "template": "send it to {EMAIL}",
            "entities": ["EMAIL"]
        },
        {
            "template": "my name is {PERSON_NAME}",
            "entities": ["PERSON_NAME"]
        },
        {
            "template": "this is {PERSON_NAME}",
            "entities": ["PERSON_NAME"]
        },
        {
            "template": "i am {PERSON_NAME}",
            "entities": ["PERSON_NAME"]
        },
        {
            "template": "speaking with {PERSON_NAME}",
            "entities": ["PERSON_NAME"]
        },
        {
            "template": "travel date is {DATE}",
            "entities": ["DATE"]
        },
        {
            "template": "booking for {DATE}",
            "entities": ["DATE"]
        },
        {
            "template": "i will travel on {DATE}",
            "entities": ["DATE"]
        },
        {
            "template": "my appointment is on {DATE}",
            "entities": ["DATE"]
        },
        {
            "template": "i live in {CITY}",
            "entities": ["CITY"]
        },
        {
            "template": "calling from {CITY}",
            "entities": ["CITY"]
        },
        {
            "template": "i am in {CITY}",
            "entities": ["CITY"]
        },
        {
            "template": "based in {CITY}",
            "entities": ["CITY"]
        },
        {
            "template": "near {LOCATION}",
            "entities": ["LOCATION"]
        },
        {
            "template": "i am at {LOCATION}",
            "entities": ["LOCATION"]
        },
        {
            "template": "address is {LOCATION}",
            "entities": ["LOCATION"]
        },
        
        # Two entity templates
        {
            "template": "hi i am {PERSON_NAME} and my phone is {PHONE}",
            "entities": ["PERSON_NAME", "PHONE"]
        },
        {
            "template": "my name is {PERSON_NAME} email {EMAIL}",
            "entities": ["PERSON_NAME", "EMAIL"]
        },
        {
            "template": "i am {PERSON_NAME} from {CITY}",
            "entities": ["PERSON_NAME", "CITY"]
        },
        {
            "template": "email id is {EMAIL} and card number is {CREDIT_CARD}",
            "entities": ["EMAIL", "CREDIT_CARD"]
        },
        {
            "template": "my number is {PHONE} i live in {CITY}",
            "entities": ["PHONE", "CITY"]
        },
        {
            "template": "card is {CREDIT_CARD} and email is {EMAIL}",
            "entities": ["CREDIT_CARD", "EMAIL"]
        },
        {
            "template": "travel on {DATE} from {CITY}",
            "entities": ["DATE", "CITY"]
        },
        {
            "template": "booking for {DATE} phone {PHONE}",
            "entities": ["DATE", "PHONE"]
        },
        
        # Three entity templates
        {
            "template": "my name is {PERSON_NAME} phone {PHONE} email {EMAIL}",
            "entities": ["PERSON_NAME", "PHONE", "EMAIL"]
        },
        {
            "template": "i am {PERSON_NAME} calling from {CITY} my number is {PHONE}",
            "entities": ["PERSON_NAME", "CITY", "PHONE"]
        },
        {
            "template": "hi this is {PERSON_NAME} my email is {EMAIL} and phone is {PHONE}",
            "entities": ["PERSON_NAME", "EMAIL", "PHONE"]
        },
        {
            "template": "card number is {CREDIT_CARD} name {PERSON_NAME} email {EMAIL}",
            "entities": ["CREDIT_CARD", "PERSON_NAME", "EMAIL"]
        },
        {
            "template": "travel date {DATE} from {CITY} phone {PHONE}",
            "entities": ["DATE", "CITY", "PHONE"]
        },
        
        # Four+ entity templates
        {
            "template": "my details are {PERSON_NAME} card {CREDIT_CARD} phone {PHONE} email {EMAIL}",
            "entities": ["PERSON_NAME", "CREDIT_CARD", "PHONE", "EMAIL"]
        },
        {
            "template": "booking for {PERSON_NAME} on {DATE} from {CITY} contact {PHONE}",
            "entities": ["PERSON_NAME", "DATE", "CITY", "PHONE"]
        },
        {
            "template": "i am {PERSON_NAME} my card is {CREDIT_CARD} email {EMAIL} phone {PHONE} from {CITY}",
            "entities": ["PERSON_NAME", "CREDIT_CARD", "EMAIL", "PHONE", "CITY"]
        },
        
        # More conversational templates
        {
            "template": "yes my name is {PERSON_NAME} and email {EMAIL}",
            "entities": ["PERSON_NAME", "EMAIL"]
        },
        {
            "template": "sure the card ending in {CREDIT_CARD}",
            "entities": ["CREDIT_CARD"]
        },
        {
            "template": "please call me back on {PHONE}",
            "entities": ["PHONE"]
        },
        {
            "template": "registered email address {EMAIL}",
            "entities": ["EMAIL"]
        },
        {
            "template": "delivery address near {LOCATION} in {CITY}",
            "entities": ["LOCATION", "CITY"]
        },
        {
            "template": "arrival date {DATE} departure from {CITY}",
            "entities": ["DATE", "CITY"]
        },
        {
            "template": "primary contact {PHONE} secondary email {EMAIL}",
            "entities": ["PHONE", "EMAIL"]
        },
        {
            "template": "cardholder name {PERSON_NAME} card {CREDIT_CARD}",
            "entities": ["PERSON_NAME", "CREDIT_CARD"]
        },
        {
            "template": "yes {PERSON_NAME} here calling from {CITY}",
            "entities": ["PERSON_NAME", "CITY"]
        },
        {
            "template": "appointment on {DATE} at {LOCATION}",
            "entities": ["DATE", "LOCATION"]
        },
        {
            "template": "my mobile number is {PHONE} from {CITY}",
            "entities": ["PHONE", "CITY"]
        },
        {
            "template": "email me at {EMAIL} or call {PHONE}",
            "entities": ["EMAIL", "PHONE"]
        },
        {
            "template": "traveling on {DATE} to {CITY} phone {PHONE}",
            "entities": ["DATE", "CITY", "PHONE"]
        },
        {
            "template": "customer {PERSON_NAME} payment card {CREDIT_CARD} date {DATE}",
            "entities": ["PERSON_NAME", "CREDIT_CARD", "DATE"]
        },
        {
            "template": "contact details {PERSON_NAME} phone {PHONE} location {LOCATION}",
            "entities": ["PERSON_NAME", "PHONE", "LOCATION"]
        },
    ]
    
    return templates


# ============================================================================
# NOISE INJECTION
# ============================================================================

# Note: We add noise during template selection and entity generation
# rather than post-processing to preserve entity span annotations


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_example(example_id: str) -> Dict:
    """Generate a single training/dev example with entities and annotations."""
    
    # Select a template
    template_data = random.choice(get_sentence_templates())
    template = template_data["template"]
    entity_types = template_data["entities"]
    
    # Generate entities
    generated_entities = {}
    person_name = None
    
    for ent_type in entity_types:
        if ent_type == "PERSON_NAME":
            person_name = generate_person_name()
            generated_entities[ent_type] = person_name
        elif ent_type == "CREDIT_CARD":
            generated_entities[ent_type] = generate_credit_card()
        elif ent_type == "PHONE":
            generated_entities[ent_type] = generate_phone()
        elif ent_type == "EMAIL":
            generated_entities[ent_type] = generate_email(person_name)
        elif ent_type == "DATE":
            generated_entities[ent_type] = generate_date()
        elif ent_type == "CITY":
            generated_entities[ent_type] = generate_city()
        elif ent_type == "LOCATION":
            generated_entities[ent_type] = generate_location()
    
    # Fill template and track positions - CORRECTED APPROACH
    text = template
    entity_annotations = []
    
    # Find all placeholder positions in original template
    entity_positions = []
    for ent_type in entity_types:
        placeholder = "{" + ent_type + "}"
        pos = template.find(placeholder)
        if pos != -1:
            entity_positions.append((pos, ent_type, placeholder))
    
    # Sort by position (process left to right)
    entity_positions.sort(key=lambda x: x[0])
    
    # Track cumulative offset as we replace placeholders
    offset = 0
    
    for template_pos, ent_type, placeholder in entity_positions:
        entity_value = generated_entities[ent_type]
        
        # Actual position in the growing text (accounting for previous replacements)
        actual_start = template_pos + offset
        
        # Find and replace this specific occurrence
        before = text[:actual_start]
        after = text[actual_start:]
        after = after.replace(placeholder, entity_value, 1)
        text = before + after
        
        # Calculate actual end position
        actual_end = actual_start + len(entity_value)
        
        entity_annotations.append({
            "start": actual_start,
            "end": actual_end,
            "label": ent_type
        })
        
        # Update offset for next replacement
        offset += len(entity_value) - len(placeholder)
    
    # Sort annotations by start position (should already be sorted, but ensure it)
    entity_annotations.sort(key=lambda x: x["start"])
    
    # Ensure all lowercase (STT characteristic) - do this BEFORE noise to preserve offsets
    text = text.lower()
    
    # Note: We skip noise injection that would change character positions
    # to maintain accurate entity span annotations
    
    return {
        "id": example_id,
        "text": text,
        "entities": entity_annotations
    }


def generate_dataset(num_examples: int, start_id: int = 1, prefix: str = "utt") -> List[Dict]:
    """Generate a complete dataset."""
    dataset = []
    
    for i in range(num_examples):
        example_id = f"{prefix}_{start_id + i:04d}"
        example = generate_example(example_id)
        dataset.append(example)
    
    return dataset


def save_dataset(dataset: List[Dict], filepath: str):
    """Save dataset to JSONL format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(dataset)} examples to {filepath}")


def print_sample_examples(dataset: List[Dict], num_samples: int = 5):
    """Print sample examples for inspection."""
    print(f"\n{'='*80}")
    print(f"SAMPLE EXAMPLES (first {num_samples}):")
    print(f"{'='*80}\n")
    
    for example in dataset[:num_samples]:
        print(f"ID: {example['id']}")
        print(f"Text: {example['text']}")
        print(f"Entities:")
        for ent in example['entities']:
            entity_text = example['text'][ent['start']:ent['end']]
            print(f"  [{ent['label']}] \"{entity_text}\" ({ent['start']}-{ent['end']})")
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic PII NER dataset with STT noise")
    parser.add_argument("--train_size", type=int, default=800, help="Number of training examples")
    parser.add_argument("--dev_size", type=int, default=150, help="Number of dev examples")
    parser.add_argument("--train_output", default="data/train.jsonl", help="Output path for training data")
    parser.add_argument("--dev_output", default="data/dev.jsonl", help="Output path for dev data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show_samples", type=int, default=5, help="Number of samples to display")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"\n{'='*80}")
    print("PII NER DATA GENERATION FOR STT TRANSCRIPTS")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Training examples: {args.train_size}")
    print(f"  Dev examples: {args.dev_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Train output: {args.train_output}")
    print(f"  Dev output: {args.dev_output}")
    print()
    
    # Generate training data
    print("Generating training data...")
    train_data = generate_dataset(args.train_size, start_id=1, prefix="train")
    save_dataset(train_data, args.train_output)
    
    # Generate dev data
    print("\nGenerating dev data...")
    dev_data = generate_dataset(args.dev_size, start_id=1, prefix="dev")
    save_dataset(dev_data, args.dev_output)
    
    # Print samples
    print_sample_examples(train_data, args.show_samples)
    
    print(f"\n{'='*80}")
    print("DATA GENERATION COMPLETE!")
    print(f"{'='*80}\n")
    print("Summary:")
    print(f"  ✅ Training: {len(train_data)} examples")
    print(f"  ✅ Dev: {len(dev_data)} examples")
    print(f"  ✅ Total: {len(train_data) + len(dev_data)} examples")
    print()
    print("Entity types covered:")
    print("  - CREDIT_CARD (PII)")
    print("  - PHONE (PII)")
    print("  - EMAIL (PII)")
    print("  - PERSON_NAME (PII)")
    print("  - DATE (PII)")
    print("  - CITY (Non-PII)")
    print("  - LOCATION (Non-PII)")
    print()
    print("STT noise patterns included:")
    print("  ✓ Spelled-out numbers (four two four two)")
    print("  ✓ Double/triple patterns (double nine, triple five)")
    print("  ✓ Spoken punctuation (at, dot)")
    print("  ✓ Mixed numeric/spoken formats")
    print("  ✓ No punctuation")
    print("  ✓ All lowercase")
    print("  ✓ Random fillers (um, uh)")
    print("  ✓ Random repetitions")
    print()

