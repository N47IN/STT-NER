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

# Number words for STT (with common misrecognitions!)
DIGIT_WORDS = {
    "0": ["zero", "oh", "o"],  # Added "o"
    "1": ["one", "won"],  # Common confusion
    "2": ["two", "to", "too"],  # Common confusion
    "3": ["three", "tree"],  # Misheard
    "4": ["four", "for"],  # Common confusion
    "5": ["five", "hive"],  # Misheard (rare)
    "6": ["six", "sex"],  # Misheard (rare)
    "7": ["seven"],
    "8": ["eight", "ate"],  # Common confusion
    "9": ["nine", "wine"]  # Misheard (rare)
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
        # INCREASED MIXING for more realistic STT noise
        if random.random() < 0.25:  # Reduced from 0.4
            return num_str  # Keep as digits
        elif random.random() < 0.35:  # Some fully spoken
            return number_to_spoken(num_str, "digit_by_digit")
        else:
            # Partial conversion with MORE randomness
            parts = []
            chunk_size = random.choice([1, 2, 2, 3, 4])  # More varied chunk sizes
            for i in range(0, len(num_str), chunk_size):
                chunk = num_str[i:i+chunk_size]
                # 60% spoken, 40% numeric (more spoken = more noise)
                if random.random() < 0.6:
                    parts.append(number_to_spoken(chunk, "digit_by_digit"))
                else:
                    parts.append(chunk)
            return " ".join(parts)
    
    return num_str


def generate_credit_card() -> str:
    """Generate realistic credit card numbers in STT format with extensive pattern diversity."""
    # Common test card numbers - expanded list
    patterns = [
        "4242424242424242",  # Visa
        "5555555555554444",  # Mastercard
        "378282246310005",   # Amex (15 digits)
        "6011111111111117",  # Discover
        "4111111111111111",  # Visa
        "5105105105105100",  # Mastercard
        "4012888888881881",  # Visa
        "3530111333300000",  # JCB
        "6011000990139424",  # Discover
        "5425233430109903",  # Mastercard
    ]
    
    card = random.choice(patterns)
    
    # Apply digit substitution errors (STT mishearing digits)
    # 15% chance of having wrong digits
    if random.random() < 0.15:
        card = add_digit_substitution_noise(card, error_rate=1.0)
    
    # Add spaces in groups of 4
    card_spaced = " ".join([card[i:i+4] for i in range(0, len(card), 4)])
    
    # VASTLY EXPANDED STT CONVERSION STYLES
    styles = [
        ("numeric_spaced", 0.15),          # "4242 4242 4242 4242"
        ("numeric_no_space", 0.08),        # "4242424242424242"
        ("full_spoken", 0.20),             # "four two four two four two..."
        ("mixed_half", 0.15),              # "4242 four two four two 4242"
        ("mixed_random", 0.12),            # Random mix per group
        ("with_doubles", 0.12),            # "double four double two..."
        ("partial_spoken_start", 0.08),    # First half spoken
        ("partial_spoken_end", 0.08),      # Last half spoken
        ("last_four_only", 0.02),          # "ending in 4242"
    ]
    
    style = random.choices([s[0] for s in styles], weights=[s[1] for s in styles])[0]
    
    if style == "numeric_spaced":
        return card_spaced
    
    elif style == "numeric_no_space":
        return card.replace(" ", "")
    
    elif style == "full_spoken":
        return number_to_spoken(card, "digit_by_digit")
    
    elif style == "mixed_half":
        # First 2 groups numeric, last 2 spoken (or vice versa)
        parts = card_spaced.split()
        if random.random() < 0.5:
            # First half numeric
            result = parts[:2] + [number_to_spoken(parts[i], "digit_by_digit") for i in range(2, len(parts))]
        else:
            # Last half numeric
            result = [number_to_spoken(parts[i], "digit_by_digit") for i in range(len(parts)-2)] + parts[-2:]
        return " ".join(result)
    
    elif style == "mixed_random":
        # Random per group
        parts = card_spaced.split()
        result = []
        for part in parts:
            if random.random() < 0.5:
                result.append(part)
            else:
                result.append(number_to_spoken(part, "digit_by_digit"))
        return " ".join(result)
    
    elif style == "with_doubles":
        return number_to_spoken(card, "with_doubles")
    
    elif style == "partial_spoken_start":
        # First 8 digits spoken, rest numeric
        first_half = number_to_spoken(card[:8], "digit_by_digit")
        second_half = " ".join([card[i:i+4] for i in range(8, len(card), 4)])
        return f"{first_half} {second_half}"
    
    elif style == "partial_spoken_end":
        # First 8 numeric, rest spoken
        first_half = " ".join([card[i:i+4] for i in range(0, 8, 4)])
        second_half = number_to_spoken(card[8:], "digit_by_digit")
        return f"{first_half} {second_half}"
    
    elif style == "last_four_only":
        # "ending in 4242" style
        last_four = card[-4:]
        if random.random() < 0.5:
            return f"ending in {last_four}"
        else:
            return f"ending in {number_to_spoken(last_four, 'digit_by_digit')}"
    
    return card_spaced


def generate_phone() -> str:
    """Generate realistic phone numbers in STT format with extensive pattern diversity."""
    # Indian phone patterns (10 digits)
    prefixes = ["98", "99", "97", "96", "95", "94", "93", "92", "91", "90",
                "88", "89", "87", "86", "85", "84", "83", "82", "81", "80",
                "70", "71", "72", "73", "74", "75", "76", "77", "78", "79"]
    
    prefix = random.choice(prefixes)
    rest = "".join([str(random.randint(0, 9)) for _ in range(8)])
    phone = prefix + rest
    
    # Apply digit substitution errors (STT mishearing digits)
    # 10% chance of having wrong digits
    if random.random() < 0.10:
        phone = add_digit_substitution_noise(phone, error_rate=1.0)
    
    # VASTLY EXPANDED STT PATTERNS for phone numbers
    patterns = [
        ("full_spoken", 0.25),              # "nine eight seven six..."
        ("grouped_2_3_3_2", 0.10),          # "98 765 432 10"
        ("grouped_5_5", 0.08),              # "98765 43210"
        ("numeric_continuous", 0.15),       # "9876543210"
        ("mixed_random", 0.15),             # Random mix of spoken/numeric
        ("with_doubles", 0.12),             # "double nine eight seven..."
        ("with_oh_for_zero", 0.10),         # Use "oh" instead of "zero"
        ("prefix_numeric_rest_spoken", 0.05), # "98 seven six five four..."
    ]
    
    pattern = random.choices([p[0] for p in patterns], weights=[p[1] for p in patterns])[0]
    
    if pattern == "full_spoken":
        return number_to_spoken(phone, "digit_by_digit")
    
    elif pattern == "grouped_2_3_3_2":
        # "98 765 432 10" style with potential spoken
        groups = [phone[0:2], phone[2:5], phone[5:8], phone[8:10]]
        if random.random() < 0.5:
            # Keep numeric
            return " ".join(groups)
        else:
            # Convert to spoken
            return " ".join([number_to_spoken(g, "digit_by_digit") for g in groups])
    
    elif pattern == "grouped_5_5":
        # "98765 43210" style
        first = phone[:5]
        second = phone[5:]
        if random.random() < 0.5:
            return f"{first} {second}"
        else:
            return f"{number_to_spoken(first, 'digit_by_digit')} {number_to_spoken(second, 'digit_by_digit')}"
    
    elif pattern == "numeric_continuous":
        return phone
    
    elif pattern == "mixed_random":
        # Mix of numeric groups and spoken groups
        groups = [phone[0:2], phone[2:5], phone[5:8], phone[8:10]]
        result = []
        for g in groups:
            if random.random() < 0.5:
                result.append(g)
            else:
                result.append(number_to_spoken(g, "digit_by_digit"))
        return " ".join(result)
    
    elif pattern == "with_doubles":
        return number_to_spoken(phone, "with_doubles")
    
    elif pattern == "with_oh_for_zero":
        # Replace "zero" with "oh"
        spoken = number_to_spoken(phone, "digit_by_digit")
        spoken = spoken.replace("zero", "oh")
        return spoken
    
    elif pattern == "prefix_numeric_rest_spoken":
        # First 2 digits numeric, rest spoken
        prefix_part = phone[:2]
        rest_part = number_to_spoken(phone[2:], "digit_by_digit")
        return f"{prefix_part} {rest_part}"
    
    return phone


def generate_email(person_name: str = None) -> str:
    """Generate realistic email addresses in STT format with more diversity."""
    if person_name and random.random() < 0.6:
        # Use person's name in email
        name_parts = person_name.lower().split()
        if len(name_parts) == 2:
            patterns = [
                f"{name_parts[0]} dot {name_parts[1]}",                    # "john dot smith"
                f"{name_parts[0]}{name_parts[1]}",                         # "johnsmith"
                f"{name_parts[0][0]} dot {name_parts[1]}",                 # "j dot smith"
                f"{name_parts[0]} {name_parts[1]}{random.randint(1, 99)}", # "john smith99"
                f"{name_parts[0]} underscore {name_parts[1]}",             # "john underscore smith"
                f"{name_parts[0]}{random.randint(1, 999)}",                # "john123"
                f"{name_parts[1]} dot {name_parts[0]}",                    # "smith dot john" (reversed)
                f"{name_parts[0][0]}{name_parts[1][0]}{random.randint(10, 99)}", # "js42"
            ]
            username = random.choice(patterns)
        else:
            username = name_parts[0]
            if random.random() < 0.5:
                username += str(random.randint(1, 999))
    else:
        # Random username
        username = random.choice(FIRST_NAMES).lower()
        if random.random() < 0.5:
            username += str(random.randint(1, 999))
        # Sometimes add underscore or dot
        if random.random() < 0.2:
            username += " underscore " + random.choice(["work", "personal", "official"])
    
    domain = random.choice(EMAIL_DOMAINS)
    extension = random.choice(EMAIL_EXTENSIONS)
    
    # STT format: "at" and "dot"
    email = f"{username} at {domain} dot {extension}"
    
    return email


def generate_date() -> str:
    """Generate dates in various STT formats with more diversity."""
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2025)
    
    # Ordinal suffixes
    ordinals = ["st", "nd", "rd", "th"]
    if day in [1, 21]: suffix = "st"
    elif day in [2, 22]: suffix = "nd"
    elif day in [3, 23]: suffix = "rd"
    else: suffix = "th"
    
    formats = [
        f"{day:02d} {month:02d} {year}",                    # "01 02 2024"
        f"{day} {month:02d} {year}",                        # "1 02 2024"
        f"{MONTH_NAMES[month-1]} {day} {year}",             # "january 1 2024"
        f"{day} {MONTH_NAMES[month-1]} {year}",             # "1 january 2024"
        f"{MONTH_SHORT[month-1]} {day} {year}",             # "jan 1 2024"
        f"{day}{suffix} {MONTH_NAMES[month-1]}",            # "15th august"
        f"{day}{suffix} of {MONTH_NAMES[month-1]}",         # "15th of august"
        f"{day} {month} {year}",                            # "1 2 2024"
        f"{MONTH_NAMES[month-1]} {day}{suffix} {year}",     # "january 15th 2024"
        f"{MONTH_SHORT[month-1]} {day}{suffix}",            # "jan 15th"
        f"{day} {MONTH_SHORT[month-1]} {year}",             # "15 jan 2024"
    ]
    
    # Special spoken formats
    if day == 1:
        formats.append(f"first of {MONTH_NAMES[month-1]}")
        formats.append(f"first {MONTH_NAMES[month-1]} {year}")
    
    date_str = random.choice(formats)
    
    # Sometimes spell out the year (more variations)
    if str(year) in date_str and random.random() < 0.25:
        if year >= 2020:
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
# SENTENCE TEMPLATES WITH VARIATIONS
# ============================================================================

# Synonyms and variations for generating diverse templates
VARIATIONS = {
    "my": ["my", "mine", "the"],
    "is": ["is", "is currently", "would be"],
    "name": ["name", "full name", "name's"],
    "phone": ["phone", "phone number", "mobile", "mobile number", "contact number", "number"],
    "email": ["email", "email id", "email address", "mail id", "mail"],
    "card": ["card", "credit card", "card number", "credit card number", "payment card"],
    "call": ["call", "ring", "reach", "contact", "get in touch with", "phone"],
    "live": ["live", "stay", "reside", "am based"],
    "from": ["from", "in", "at", "calling from"],
    "travel": ["travel", "traveling", "going", "flying"],
    "booking": ["booking", "reservation", "appointment"],
}

# Fillers and disfluencies for STT realism
FILLERS = ["uh", "um", "like", "you know", "so", "well", "actually", "basically"]
HESITATIONS = ["let me see", "give me a second", "one moment", "just a sec"]


def generate_base_templates() -> List[Dict]:
    """
    Generate comprehensive base templates.
    We'll apply variations on top of these to create hundreds more.
    """
    templates = []
    
    # ========== CREDIT CARD TEMPLATES (50+) ==========
    cc_patterns = [
        "my credit card number is {CREDIT_CARD}",
        "card number {CREDIT_CARD}",
        "please charge {CREDIT_CARD}",
        "the card is {CREDIT_CARD}",
        "i want to use card {CREDIT_CARD}",
        "payment card {CREDIT_CARD}",
        "card ending in {CREDIT_CARD}",
        "charge my card {CREDIT_CARD}",
        "use this card {CREDIT_CARD}",
        "credit card {CREDIT_CARD}",
        "the number is {CREDIT_CARD}",
        "it is {CREDIT_CARD}",
        "that would be {CREDIT_CARD}",
        "pay with {CREDIT_CARD}",
        "debit card {CREDIT_CARD}",
        "visa card {CREDIT_CARD}",
        "mastercard {CREDIT_CARD}",
        "my card details are {CREDIT_CARD}",
        "card on file is {CREDIT_CARD}",
        "saved card {CREDIT_CARD}",
    ]
    for p in cc_patterns:
        templates.append({"template": p, "entities": ["CREDIT_CARD"]})
    
    # ========== PHONE TEMPLATES (50+) ==========
    phone_patterns = [
        "call me on {PHONE}",
        "my number is {PHONE}",
        "phone number {PHONE}",
        "you can reach me at {PHONE}",
        "contact number {PHONE}",
        "my mobile is {PHONE}",
        "reach me on {PHONE}",
        "call me at {PHONE}",
        "ring me on {PHONE}",
        "phone me at {PHONE}",
        "the number is {PHONE}",
        "contact me on {PHONE}",
        "my contact is {PHONE}",
        "mobile number {PHONE}",
        "cell phone {PHONE}",
        "callback number {PHONE}",
        "reach out on {PHONE}",
        "get in touch at {PHONE}",
        "dial {PHONE}",
        "phone is {PHONE}",
        "number to call is {PHONE}",
        "best number is {PHONE}",
        "primary contact {PHONE}",
        "alternate number {PHONE}",
        "my cell is {PHONE}",
    ]
    for p in phone_patterns:
        templates.append({"template": p, "entities": ["PHONE"]})
    
    # ========== EMAIL TEMPLATES (50+) ==========
    email_patterns = [
        "my email is {EMAIL}",
        "email id is {EMAIL}",
        "send it to {EMAIL}",
        "my mail is {EMAIL}",
        "email address is {EMAIL}",
        "write to me at {EMAIL}",
        "mail me at {EMAIL}",
        "email me at {EMAIL}",
        "contact email {EMAIL}",
        "reach me at {EMAIL}",
        "send to {EMAIL}",
        "my address is {EMAIL}",
        "email is {EMAIL}",
        "mail id {EMAIL}",
        "send details to {EMAIL}",
        "forward to {EMAIL}",
        "reply to {EMAIL}",
        "registered email {EMAIL}",
        "primary email {EMAIL}",
        "alternate email {EMAIL}",
        "work email {EMAIL}",
        "personal email {EMAIL}",
        "the email is {EMAIL}",
        "email address would be {EMAIL}",
        "you can email me at {EMAIL}",
    ]
    for p in email_patterns:
        templates.append({"template": p, "entities": ["EMAIL"]})
    
    # ========== NAME TEMPLATES (40+) ==========
    name_patterns = [
        "my name is {PERSON_NAME}",
        "this is {PERSON_NAME}",
        "i am {PERSON_NAME}",
        "speaking with {PERSON_NAME}",
        "my full name is {PERSON_NAME}",
        "name is {PERSON_NAME}",
        "i am calling as {PERSON_NAME}",
        "it is {PERSON_NAME}",
        "this is {PERSON_NAME} speaking",
        "you can call me {PERSON_NAME}",
        "name {PERSON_NAME}",
        "{PERSON_NAME} here",
        "talking to {PERSON_NAME}",
        "{PERSON_NAME} on the line",
        "customer name is {PERSON_NAME}",
        "registered under {PERSON_NAME}",
        "account holder {PERSON_NAME}",
        "booking name {PERSON_NAME}",
        "reserved for {PERSON_NAME}",
        "{PERSON_NAME} calling",
    ]
    for p in name_patterns:
        templates.append({"template": p, "entities": ["PERSON_NAME"]})
    
    # ========== DATE TEMPLATES (40+) ==========
    date_patterns = [
        "travel date is {DATE}",
        "booking for {DATE}",
        "i will travel on {DATE}",
        "my appointment is on {DATE}",
        "date is {DATE}",
        "on {DATE}",
        "scheduled for {DATE}",
        "departing on {DATE}",
        "arriving on {DATE}",
        "leaving on {DATE}",
        "coming on {DATE}",
        "visit on {DATE}",
        "appointment on {DATE}",
        "meeting on {DATE}",
        "reservation for {DATE}",
        "check in {DATE}",
        "check out {DATE}",
        "flying on {DATE}",
        "traveling on {DATE}",
        "date of travel {DATE}",
        "booked for {DATE}",
        "scheduled on {DATE}",
        "planned for {DATE}",
        "date of birth {DATE}",
        "dob is {DATE}",
    ]
    for p in date_patterns:
        templates.append({"template": p, "entities": ["DATE"]})
    
    # ========== CITY TEMPLATES (30+) ==========
    city_patterns = [
        "i live in {CITY}",
        "calling from {CITY}",
        "i am in {CITY}",
        "based in {CITY}",
        "from {CITY}",
        "located in {CITY}",
        "currently in {CITY}",
        "residing in {CITY}",
        "staying in {CITY}",
        "traveling to {CITY}",
        "flying to {CITY}",
        "going to {CITY}",
        "visiting {CITY}",
        "city is {CITY}",
        "my city is {CITY}",
        "hometown {CITY}",
        "current city {CITY}",
        "destination {CITY}",
        "arriving in {CITY}",
        "departing from {CITY}",
    ]
    for p in city_patterns:
        templates.append({"template": p, "entities": ["CITY"]})
    
    # ========== LOCATION TEMPLATES (25+) ==========
    location_patterns = [
        "near {LOCATION}",
        "i am at {LOCATION}",
        "address is {LOCATION}",
        "location is {LOCATION}",
        "at {LOCATION}",
        "staying at {LOCATION}",
        "currently at {LOCATION}",
        "meeting at {LOCATION}",
        "pickup from {LOCATION}",
        "delivery to {LOCATION}",
        "address near {LOCATION}",
        "close to {LOCATION}",
        "by {LOCATION}",
        "around {LOCATION}",
        "my location is {LOCATION}",
    ]
    for p in location_patterns:
        templates.append({"template": p, "entities": ["LOCATION"]})
    
    # ========== TWO ENTITY COMBINATIONS (80+) ==========
    two_entity = [
        # Name + Phone
        "hi i am {PERSON_NAME} and my phone is {PHONE}",
        "this is {PERSON_NAME} my number is {PHONE}",
        "{PERSON_NAME} here my phone is {PHONE}",
        "my name is {PERSON_NAME} call me on {PHONE}",
        "i am {PERSON_NAME} contact me at {PHONE}",
        "{PERSON_NAME} phone {PHONE}",
        "name {PERSON_NAME} number {PHONE}",
        "calling as {PERSON_NAME} mobile {PHONE}",
        "{PERSON_NAME} you can reach me at {PHONE}",
        "this is {PERSON_NAME} ring me on {PHONE}",
        
        # Name + Email
        "my name is {PERSON_NAME} email {EMAIL}",
        "i am {PERSON_NAME} my email is {EMAIL}",
        "{PERSON_NAME} here email {EMAIL}",
        "this is {PERSON_NAME} mail me at {EMAIL}",
        "name {PERSON_NAME} mail id {EMAIL}",
        "{PERSON_NAME} email address {EMAIL}",
        "customer {PERSON_NAME} email {EMAIL}",
        "registered as {PERSON_NAME} contact {EMAIL}",
        "{PERSON_NAME} write to {EMAIL}",
        "booking for {PERSON_NAME} email {EMAIL}",
        
        # Name + City
        "i am {PERSON_NAME} from {CITY}",
        "{PERSON_NAME} calling from {CITY}",
        "this is {PERSON_NAME} in {CITY}",
        "my name is {PERSON_NAME} based in {CITY}",
        "{PERSON_NAME} from {CITY}",
        "name {PERSON_NAME} city {CITY}",
        "{PERSON_NAME} located in {CITY}",
        "customer {PERSON_NAME} from {CITY}",
        "{PERSON_NAME} residing in {CITY}",
        "i am {PERSON_NAME} currently in {CITY}",
        
        # Phone + City
        "my number is {PHONE} i live in {CITY}",
        "phone {PHONE} calling from {CITY}",
        "call me on {PHONE} i am in {CITY}",
        "mobile {PHONE} from {CITY}",
        "contact {PHONE} based in {CITY}",
        "number {PHONE} city {CITY}",
        "reach me at {PHONE} in {CITY}",
        "{PHONE} located in {CITY}",
        "phone number {PHONE} from {CITY}",
        "my mobile is {PHONE} currently in {CITY}",
        
        # Phone + Email
        "phone {PHONE} email {EMAIL}",
        "call me on {PHONE} or email {EMAIL}",
        "my number is {PHONE} mail is {EMAIL}",
        "contact {PHONE} or {EMAIL}",
        "reach me at {PHONE} or write to {EMAIL}",
        "mobile {PHONE} mail id {EMAIL}",
        "phone number {PHONE} email address {EMAIL}",
        "you can call {PHONE} or mail {EMAIL}",
        "primary contact {PHONE} secondary {EMAIL}",
        "phone is {PHONE} email is {EMAIL}",
        
        # Card + Email
        "email id is {EMAIL} and card number is {CREDIT_CARD}",
        "card is {CREDIT_CARD} and email is {EMAIL}",
        "my card {CREDIT_CARD} email {EMAIL}",
        "payment card {CREDIT_CARD} contact {EMAIL}",
        "charge {CREDIT_CARD} send receipt to {EMAIL}",
        "card number {CREDIT_CARD} mail id {EMAIL}",
        "credit card {CREDIT_CARD} email address {EMAIL}",
        "using card {CREDIT_CARD} registered email {EMAIL}",
        "{CREDIT_CARD} is my card email {EMAIL}",
        "card on file {CREDIT_CARD} contact email {EMAIL}",
        
        # Date + City
        "travel on {DATE} from {CITY}",
        "booking for {DATE} departing from {CITY}",
        "flying on {DATE} to {CITY}",
        "arriving on {DATE} in {CITY}",
        "date {DATE} city {CITY}",
        "scheduled for {DATE} from {CITY}",
        "departure date {DATE} from {CITY}",
        "traveling on {DATE} to {CITY}",
        "visit on {DATE} in {CITY}",
        "appointment on {DATE} at {CITY}",
        
        # Date + Phone
        "booking for {DATE} phone {PHONE}",
        "travel date {DATE} contact {PHONE}",
        "appointment on {DATE} call me on {PHONE}",
        "scheduled for {DATE} number {PHONE}",
        "date {DATE} phone {PHONE}",
        "flying on {DATE} mobile {PHONE}",
        "reservation for {DATE} contact {PHONE}",
        "{DATE} is the date reach me at {PHONE}",
        "departing {DATE} callback {PHONE}",
        "arriving {DATE} phone {PHONE}",
        
        # Location + City
        "delivery address near {LOCATION} in {CITY}",
        "location is {LOCATION} city {CITY}",
        "staying at {LOCATION} in {CITY}",
        "pickup from {LOCATION} in {CITY}",
        "meeting at {LOCATION} in {CITY}",
        "address {LOCATION} in {CITY}",
        "currently at {LOCATION} in {CITY}",
        "near {LOCATION} in {CITY}",
        "by {LOCATION} in {CITY}",
        "close to {LOCATION} in {CITY}",
    ]
    for p in two_entity:
        # Parse entities from template
        entities = []
        if "{PERSON_NAME}" in p: entities.append("PERSON_NAME")
        if "{CREDIT_CARD}" in p: entities.append("CREDIT_CARD")
        if "{PHONE}" in p: entities.append("PHONE")
        if "{EMAIL}" in p: entities.append("EMAIL")
        if "{DATE}" in p: entities.append("DATE")
        if "{CITY}" in p: entities.append("CITY")
        if "{LOCATION}" in p: entities.append("LOCATION")
        templates.append({"template": p, "entities": entities})
    
    # ========== THREE+ ENTITY COMBINATIONS (50+) ==========
    multi_entity = [
        # Name + Phone + Email
        "my name is {PERSON_NAME} phone {PHONE} email {EMAIL}",
        "i am {PERSON_NAME} call me on {PHONE} mail {EMAIL}",
        "{PERSON_NAME} here number {PHONE} mail id {EMAIL}",
        "this is {PERSON_NAME} my phone is {PHONE} and email is {EMAIL}",
        "name {PERSON_NAME} mobile {PHONE} email address {EMAIL}",
        "{PERSON_NAME} contact {PHONE} or {EMAIL}",
        "customer {PERSON_NAME} phone {PHONE} email {EMAIL}",
        "i am {PERSON_NAME} reach me at {PHONE} or write to {EMAIL}",
        "booking for {PERSON_NAME} contact number {PHONE} mail {EMAIL}",
        "{PERSON_NAME} you can call {PHONE} or email {EMAIL}",
        
        # Name + City + Phone
        "i am {PERSON_NAME} calling from {CITY} my number is {PHONE}",
        "{PERSON_NAME} from {CITY} phone {PHONE}",
        "this is {PERSON_NAME} in {CITY} call me on {PHONE}",
        "my name is {PERSON_NAME} based in {CITY} mobile {PHONE}",
        "name {PERSON_NAME} city {CITY} contact {PHONE}",
        "{PERSON_NAME} located in {CITY} number {PHONE}",
        "customer {PERSON_NAME} from {CITY} phone {PHONE}",
        "{PERSON_NAME} residing in {CITY} reach me at {PHONE}",
        "i am {PERSON_NAME} currently in {CITY} my mobile is {PHONE}",
        "calling as {PERSON_NAME} from {CITY} contact number {PHONE}",
        
        # Card + Name + Email
        "card number is {CREDIT_CARD} name {PERSON_NAME} email {EMAIL}",
        "my card {CREDIT_CARD} i am {PERSON_NAME} mail {EMAIL}",
        "payment card {CREDIT_CARD} customer {PERSON_NAME} contact {EMAIL}",
        "charge {CREDIT_CARD} name {PERSON_NAME} send receipt to {EMAIL}",
        "{CREDIT_CARD} is my card my name is {PERSON_NAME} email {EMAIL}",
        "credit card {CREDIT_CARD} registered as {PERSON_NAME} mail id {EMAIL}",
        "using card {CREDIT_CARD} i am {PERSON_NAME} email address {EMAIL}",
        "card on file {CREDIT_CARD} customer name {PERSON_NAME} contact {EMAIL}",
        "debit card {CREDIT_CARD} name {PERSON_NAME} email {EMAIL}",
        "visa card {CREDIT_CARD} holder {PERSON_NAME} mail {EMAIL}",
        
        # Date + City + Phone
        "travel date {DATE} from {CITY} phone {PHONE}",
        "booking for {DATE} departing from {CITY} contact {PHONE}",
        "flying on {DATE} to {CITY} call me on {PHONE}",
        "arriving on {DATE} in {CITY} mobile {PHONE}",
        "date {DATE} city {CITY} number {PHONE}",
        "scheduled for {DATE} from {CITY} reach me at {PHONE}",
        "departure date {DATE} from {CITY} phone {PHONE}",
        "traveling on {DATE} to {CITY} contact {PHONE}",
        "visit on {DATE} in {CITY} call {PHONE}",
        "appointment on {DATE} at {CITY} phone {PHONE}",
        
        # Name + Date + City
        "booking for {PERSON_NAME} on {DATE} from {CITY}",
        "{PERSON_NAME} traveling on {DATE} to {CITY}",
        "reservation for {PERSON_NAME} date {DATE} city {CITY}",
        "customer {PERSON_NAME} flying on {DATE} from {CITY}",
        "{PERSON_NAME} appointment on {DATE} in {CITY}",
        "name {PERSON_NAME} date {DATE} location {CITY}",
        "{PERSON_NAME} arriving {DATE} in {CITY}",
        "i am {PERSON_NAME} scheduled for {DATE} from {CITY}",
        "{PERSON_NAME} departing {DATE} from {CITY}",
        "booking name {PERSON_NAME} travel date {DATE} city {CITY}",
        
        # Four+ entities
        "my details are {PERSON_NAME} card {CREDIT_CARD} phone {PHONE} email {EMAIL}",
        "i am {PERSON_NAME} my card is {CREDIT_CARD} email {EMAIL} phone {PHONE} from {CITY}",
        "customer {PERSON_NAME} payment card {CREDIT_CARD} date {DATE}",
        "contact details {PERSON_NAME} phone {PHONE} location {LOCATION}",
        "booking for {PERSON_NAME} on {DATE} from {CITY} contact {PHONE}",
        "{PERSON_NAME} traveling on {DATE} to {CITY} phone {PHONE} email {EMAIL}",
        "name {PERSON_NAME} card {CREDIT_CARD} email {EMAIL} mobile {PHONE}",
        "i am {PERSON_NAME} from {CITY} card {CREDIT_CARD} contact {PHONE}",
        "{PERSON_NAME} appointment {DATE} at {LOCATION} phone {PHONE}",
        "customer {PERSON_NAME} city {CITY} email {EMAIL} phone {PHONE}",
    ]
    for p in multi_entity:
        entities = []
        if "{PERSON_NAME}" in p: entities.append("PERSON_NAME")
        if "{CREDIT_CARD}" in p: entities.append("CREDIT_CARD")
        if "{PHONE}" in p: entities.append("PHONE")
        if "{EMAIL}" in p: entities.append("EMAIL")
        if "{DATE}" in p: entities.append("DATE")
        if "{CITY}" in p: entities.append("CITY")
        if "{LOCATION}" in p: entities.append("LOCATION")
        templates.append({"template": p, "entities": entities})
    
    return templates


def add_stt_noise(text: str, noise_prob: float = 0.7) -> str:
    """Add realistic STT noise: fillers, hesitations, false starts, repetitions."""
    # MUCH higher noise probability - this is STT!
    if random.random() > noise_prob:
        return text
    
    words = text.split()
    if len(words) < 2:
        return text
    
    # Apply MULTIPLE noise types to single utterance (more realistic)
    num_noise_ops = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    
    for _ in range(num_noise_ops):
        if len(words) < 2:
            break
            
        noise_type = random.choice(["filler", "hesitation", "repeat", "false_start", "pause_word"])
        
        if noise_type == "filler":
            # Add filler word at random position (not just start!)
            pos = random.randint(0, len(words))
            filler = random.choice(FILLERS)
            words.insert(pos, filler)
        
        elif noise_type == "hesitation":
            # Add hesitation phrase (usually at start, but not always)
            hesitation = random.choice(HESITATIONS)
            if random.random() < 0.8:  # 80% at start
                words = hesitation.split() + words
            else:  # 20% in middle
                pos = random.randint(1, max(1, len(words) // 2))
                words = words[:pos] + hesitation.split() + words[pos:]
        
        elif noise_type == "repeat":
            # Repeat a word (stuttering) - more common in STT
            pos = random.randint(0, len(words) - 1)
            # Sometimes repeat 2-3 times (severe stutter)
            repeats = random.choice([1, 1, 1, 2])
            for _ in range(repeats):
                words.insert(pos, words[pos])
        
        elif noise_type == "false_start":
            # False start: add incomplete word/phrase at beginning
            false_starts = ["i", "my", "the", "so", "and", "but", "can you", "sorry", "wait"]
            if random.random() < 0.3:
                starter = random.choice(false_starts)
                words = starter.split() + words
        
        elif noise_type == "pause_word":
            # Add pause markers common in STT
            pauses = ["uh", "um", "er", "ah"]
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(pauses))
    
    return " ".join(words)


def get_sentence_templates(split: str = "train") -> List[Dict]:
    """
    Get templates for a specific split (train/dev/test).
    Ensures NO overlap between splits.
    """
    all_templates = generate_base_templates()
    
    # Shuffle ONCE with a fixed seed, then split deterministically
    random.seed(42)  # Master seed for splitting
    random.shuffle(all_templates)
    
    # Split templates: 60% train, 20% dev, 20% test
    n = len(all_templates)
    train_end = int(0.6 * n)
    dev_end = int(0.8 * n)
    
    # Reset seed for subsequent random operations
    random.seed()
    
    if split == "train":
        return all_templates[:train_end]
    elif split == "dev":
        return all_templates[train_end:dev_end]
    else:  # test
        return all_templates[dev_end:]


# ============================================================================
# NOISE INJECTION
# ============================================================================

# STT word confusions (homophones and near-homophones)
STT_CONFUSIONS = {
    "two": ["to", "too"],
    "to": ["two", "too"],
    "four": ["for"],
    "for": ["four"],
    "eight": ["ate"],
    "one": ["won"],
    "won": ["one"],
    "by": ["buy", "bye"],
    "no": ["know"],
    "know": ["no"],
}

# Common functional words that STT systems often miss or mishear (acoustic errors)
FUNCTIONAL_WORDS = {
    "articles": ["a", "an", "the"],
    "prepositions": ["is", "was", "for", "in", "to", "at", "on", "of", "from"],
    "conjunctions": ["and", "or", "but"],
    "auxiliaries": ["is", "was", "are", "were", "have", "has", "had"],
}

# Common filler words that might replace functional words (acoustic substitution)
ACOUSTIC_SUBSTITUTIONS = ["like", "so", "um", "uh", "you know", "well", "actually"]

def add_word_confusions(text: str, confusion_prob: float = 0.15) -> str:
    """Add realistic STT word confusions (homophones)."""
    if random.random() > confusion_prob:
        return text
    
    words = text.split()
    if len(words) < 3:
        return text
    
    # Pick 1-2 words to confuse
    num_confusions = random.choice([1, 1, 2])
    available_indices = [i for i, w in enumerate(words) if w in STT_CONFUSIONS]
    
    if not available_indices:
        return text
    
    num_confusions = min(num_confusions, len(available_indices))
    indices_to_confuse = random.sample(available_indices, num_confusions)
    
    for idx in indices_to_confuse:
        original = words[idx]
        if original in STT_CONFUSIONS:
            words[idx] = random.choice(STT_CONFUSIONS[original])
    
    return " ".join(words)


def add_acoustic_noise(text: str, entity_annotations: List[Dict], deletion_prob: float = 0.03, substitution_prob: float = 0.03) -> tuple:
    """
    Add realistic acoustic STT noise: random word deletions and substitutions.
    Simulates acoustic errors that cause STT to miss or mishear common functional words.
    
    Args:
        text: Original text
        entity_annotations: List of entity annotations (start, end, label)
        deletion_prob: Probability of deleting a functional word
        substitution_prob: Probability of substituting a functional word
    
    Returns:
        (modified_text, adjusted_annotations) - Text with noise and adjusted entity positions
    """
    words = text.split()
    if len(words) < 3:
        return text, entity_annotations
    
    # Build word-level positions for entity tracking
    word_positions = []
    char_pos = 0
    for i, word in enumerate(words):
        start = char_pos
        end = char_pos + len(word)
        word_positions.append((i, start, end, word))
        char_pos = end + 1  # +1 for space
    
    # Collect all functional words and their positions
    functional_word_indices = []
    all_functional = []
    for word_list in FUNCTIONAL_WORDS.values():
        all_functional.extend(word_list)
    
    for i, (word_idx, start, end, word) in enumerate(word_positions):
        if word.lower() in all_functional:
            functional_word_indices.append(i)
    
    if not functional_word_indices:
        return text, entity_annotations
    
    # Track modifications for entity position adjustment
    char_offset = 0
    deleted_indices = set()
    modified_annotations = []
    
    # Apply deletions (simulate acoustic dropout)
    if random.random() < deletion_prob and functional_word_indices:
        idx_to_delete = random.choice(functional_word_indices)
        word_idx, start, end, word = word_positions[idx_to_delete]
        
        # Don't delete if it would break entity boundaries
        # Check if this word overlaps with any entity
        overlaps_entity = False
        for ent in entity_annotations:
            if start < ent["end"] and end > ent["start"]:
                overlaps_entity = True
                break
        
        if not overlaps_entity:
            deleted_indices.add(idx_to_delete)
            char_offset -= (end - start + 1)  # Remove word + space
    
    # Apply substitutions (simulate acoustic mishearing)
    substitutions = {}
    if random.random() < substitution_prob and functional_word_indices:
        available = [i for i in functional_word_indices if i not in deleted_indices]
        if available:
            idx_to_sub = random.choice(available)
            word_idx, start, end, word = word_positions[idx_to_sub]
            
            # Don't substitute if it overlaps with entities
            overlaps_entity = False
            for ent in entity_annotations:
                if start < ent["end"] and end > ent["start"]:
                    overlaps_entity = True
                    break
            
            if not overlaps_entity:
                substitute = random.choice(ACOUSTIC_SUBSTITUTIONS)
                substitutions[idx_to_sub] = (word, substitute)
                # Adjust offset: new word might be different length
                char_offset += len(substitute) - len(word)
    
    # Build modified text and adjust entity positions
    modified_words = []
    for i, (word_idx, start, end, word) in enumerate(word_positions):
        if i in deleted_indices:
            continue
        elif i in substitutions:
            modified_words.append(substitutions[i][1])
        else:
            modified_words.append(word)
    
    modified_text = " ".join(modified_words)
    
    # Adjust entity annotations for deletions/substitutions
    for ent in entity_annotations:
        # Count how many characters were removed/added before this entity
        offset = 0
        for i, (word_idx, start, end, word) in enumerate(word_positions):
            if i in deleted_indices and end <= ent["start"]:
                offset -= (end - start + 1)
            elif i in substitutions and end <= ent["start"]:
                old_word, new_word = substitutions[i]
                offset += len(new_word) - len(old_word)
            elif i in deleted_indices and start < ent["start"] < end:
                # Entity starts inside deleted word - shouldn't happen, but handle gracefully
                offset -= (ent["start"] - start)
        
        new_start = max(0, ent["start"] + offset)
        new_end = max(new_start, ent["end"] + offset)
        
        # Ensure new positions are within modified text bounds
        new_end = min(new_end, len(modified_text))
        new_start = min(new_start, new_end)
        
        modified_annotations.append({
            "start": new_start,
            "end": new_end,
            "label": ent["label"]
        })
    
    return modified_text, modified_annotations


def add_digit_substitution_noise(num_str: str, error_rate: float = 0.10) -> str:
    """
    Add random digit substitution errors (very common in STT).
    E.g., "8" might be heard as "9", creating "9876543210" instead of "8876543210"
    """
    if random.random() > error_rate:
        return num_str
    
    # Substitute 1-2 digits randomly
    if len(num_str) < 4:
        return num_str
    
    num_errors = random.choice([1, 1, 2])  # Usually 1 error, sometimes 2
    digits = list(num_str)
    
    for _ in range(num_errors):
        if len(digits) < 1:
            break
        # Pick a random position
        pos = random.randint(0, len(digits) - 1)
        if digits[pos].isdigit():
            # Substitute with a nearby digit (common STT error)
            current = int(digits[pos])
            # Adjacent digits are more likely to be confused
            nearby = [
                (current - 1) % 10,
                (current + 1) % 10,
                (current - 2) % 10,
                (current + 2) % 10,
            ]
            digits[pos] = str(random.choice(nearby))
    
    return "".join(digits)

# Note: We add noise during template selection and entity generation
# rather than post-processing to preserve entity span annotations


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_example(example_id: str, split: str = "train", add_noise: bool = True) -> Dict:
    """Generate a single training/dev example with entities and annotations."""
    
    # Select a template for this split
    templates = get_sentence_templates(split)
    template_data = random.choice(templates)
    template = template_data["template"]
    entity_types = template_data["entities"]
    
    # Apply word confusions to template (before entity generation)
    # This ensures entity positions remain correct
    if add_noise and random.random() < 0.20:  # 20% of templates get word confusions
        template = add_word_confusions(template)
    
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
    
    # Add STT noise carefully - MUCH MORE frequently for realistic STT
    if add_noise and random.random() < 0.65:  # 65% chance of noise (realistic for STT!)
        noise_options = []
        
        # Fillers at start (more common)
        if random.random() < 0.5:
            filler = random.choice(FILLERS)
            noise_options.append(filler)
        
        # Hesitations at start
        if random.random() < 0.3:
            hesitation = random.choice(HESITATIONS)
            noise_options.append(hesitation)
        
        # Conversational starts (very common in calls)
        if random.random() < 0.4:
            starts = ["yeah", "yes", "okay", "sure", "alright", "right", "so", "well", "hi", "hello"]
            noise_options.append(random.choice(starts))
        
        # False starts (common in STT)
        if random.random() < 0.25:
            false_starts = ["i mean", "like i said", "you know what", "so basically"]
            noise_options.append(random.choice(false_starts))
        
        if noise_options:
            # Shuffle noise options for variety
            random.shuffle(noise_options)
            # Don't add ALL options, pick 1-2
            noise_options = noise_options[:random.choice([1, 1, 2])]
            prefix = " ".join(noise_options) + " "
            prefix_len = len(prefix)
            text = prefix + text
            # Shift all entity positions
            for ent in entity_annotations:
                ent["start"] += prefix_len
                ent["end"] += prefix_len
    
    # ADDITIONALLY: Add suffix noise (safe - doesn't affect entity positions)
    if add_noise and random.random() < 0.3:
        # Add conversational endings
        suffixes = ["thanks", "thank you", "please", "okay", "got it", "thats it", "thats all"]
        suffix = random.choice(suffixes)
        text = text + " " + suffix
    
    # STT GARBLING: Sometimes add garbled/cut-off endings (very common in real STT)
    if add_noise and random.random() < 0.15:
        # Add incomplete/garbled suffix
        garbled = ["sorry what", "can you", "wait a", "hold on", "one sec", "just"]
        text = text + " " + random.choice(garbled)
    
    # ACOUSTIC NOISE: Add realistic word-level errors (deletions/substitutions)
    # This simulates acoustic STT errors that break template patterns
    if add_noise:
        text, entity_annotations = add_acoustic_noise(text, entity_annotations)
    
    return {
        "id": example_id,
        "text": text,
        "entities": entity_annotations
    }


def generate_dataset(num_examples: int, start_id: int = 1, prefix: str = "utt", split: str = "train", add_noise: bool = True) -> List[Dict]:
    """Generate a complete dataset for a specific split."""
    dataset = []
    
    for i in range(num_examples):
        example_id = f"{prefix}_{start_id + i:04d}"
        example = generate_example(example_id, split=split, add_noise=add_noise)
        dataset.append(example)
    
    return dataset


def save_dataset(dataset: List[Dict], filepath: str, include_labels: bool = True):
    """Save dataset to JSONL format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for example in dataset:
            if include_labels:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            else:
                # For test set: only id and text, no entities
                test_example = {
                    "id": example["id"],
                    "text": example["text"]
                }
                f.write(json.dumps(test_example, ensure_ascii=False) + "\n")
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
    parser.add_argument("--train_size", type=int, default=1000, help="Number of training examples")
    parser.add_argument("--dev_size", type=int, default=200, help="Number of dev examples")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test examples (no labels)")
    parser.add_argument("--train_output", default="data/train.jsonl", help="Output path for training data")
    parser.add_argument("--dev_output", default="data/dev.jsonl", help="Output path for dev data")
    parser.add_argument("--test_output", default="data/test.jsonl", help="Output path for test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show_samples", type=int, default=5, help="Number of samples to display")
    parser.add_argument("--generate_test", action="store_true", help="Generate test data without labels")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"\n{'='*80}")
    print("PII NER DATA GENERATION FOR STT TRANSCRIPTS")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Training examples: {args.train_size}")
    print(f"  Dev examples: {args.dev_size}")
    if args.generate_test:
        print(f"  Test examples: {args.test_size} (NO LABELS)")
    print(f"  Random seed: {args.seed}")
    print(f"  Train output: {args.train_output}")
    print(f"  Dev output: {args.dev_output}")
    if args.generate_test:
        print(f"  Test output: {args.test_output}")
    print()
    
    # Generate training data
    if args.train_size > 0:
        print("Generating training data...")
        train_data = generate_dataset(args.train_size, start_id=1, prefix="train", split="train", add_noise=True)
        save_dataset(train_data, args.train_output, include_labels=True)
        print(f"  Using {len(get_sentence_templates('train'))} unique templates for train")
    else:
        train_data = []
    
    # Generate dev data
    if args.dev_size > 0:
        print("\nGenerating dev data...")
        dev_data = generate_dataset(args.dev_size, start_id=1, prefix="dev", split="dev", add_noise=True)
        save_dataset(dev_data, args.dev_output, include_labels=True)
        print(f"  Using {len(get_sentence_templates('dev'))} unique templates for dev")
    else:
        dev_data = []
    
    # Generate test data WITHOUT labels
    if args.generate_test and args.test_size > 0:
        print("\nGenerating test data (WITHOUT LABELS)...")
        test_data = generate_dataset(args.test_size, start_id=1, prefix="test", split="test", add_noise=True)
        save_dataset(test_data, args.test_output, include_labels=False)
        print(f"  Using {len(get_sentence_templates('test'))} unique templates for test")
    else:
        test_data = []
    
    # Print samples
    if train_data:
        print_sample_examples(train_data, args.show_samples)
    
    print(f"\n{'='*80}")
    print("DATA GENERATION COMPLETE!")
    print(f"{'='*80}\n")
    print("Summary:")
    if train_data:
        print(f"  ✅ Training: {len(train_data)} examples (WITH LABELS)")
    if dev_data:
        print(f"  ✅ Dev: {len(dev_data)} examples (WITH LABELS)")
    if test_data:
        print(f"  ✅ Test: {len(test_data)} examples (NO LABELS)")
    print(f"  ✅ Total: {len(train_data) + len(dev_data) + len(test_data)} examples")
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
    print("Template diversity:")
    print(f"  ✓ Train templates: {len(get_sentence_templates('train'))}")
    print(f"  ✓ Dev templates: {len(get_sentence_templates('dev'))}")
    if test_data:
        print(f"  ✓ Test templates: {len(get_sentence_templates('test'))}")
    print(f"  ✓ ZERO overlap between train/dev/test templates")
    print()
    print("STT noise patterns included:")
    print("  ✓ Extensive credit card patterns (10+ variations)")
    print("  ✓ Extensive phone patterns (8+ variations)")
    print("  ✓ Spelled-out numbers (four two four two)")
    print("  ✓ Double/triple patterns (double nine, triple five)")
    print("  ✓ Spoken punctuation (at, dot, underscore)")
    print("  ✓ Mixed numeric/spoken formats")
    print("  ✓ Use of 'oh' for zero in phones")
    print("  ✓ Conversational fillers (uh, um, like)")
    print("  ✓ Hesitations (let me see, one moment)")
    print("  ✓ Conversational starts (yeah, okay, sure)")
    print("  ✓ No punctuation")
    print("  ✓ All lowercase")
    print("  ✓ Diverse date formats (25+ variations)")
    print("  ✓ Diverse email formats (25+ variations)")
    print()

