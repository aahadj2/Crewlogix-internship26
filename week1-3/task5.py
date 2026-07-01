import json
from ollama import chat

REQUIRED_KEYS = ["role_title", "required_skills", "seniority_level", "estimated_salary_pkr_range"]

job_descriptions = [
    """
    We are hiring a Senior Python Developer at a fintech startup in Karachi.
    The candidate must have 5+ years of experience with Python, Django, REST APIs,
    PostgreSQL, and AWS. Strong knowledge of microservices architecture required.
    Leadership experience is a plus.
    """,
    """
    Looking for a Junior Graphic Designer in Lahore. Fresh graduates welcome.
    Must know Adobe Photoshop, Illustrator, and Canva. Basic knowledge of UI/UX
    is preferred. Portfolio required.
    """,
    """
    Urgently needed: Data Analyst for an e-commerce company in Islamabad.
    3 years experience required. Skills needed: Excel, Power BI, SQL, Python basics.
    Will be responsible for sales reporting and dashboard creation.
    """
]

prompt_template = """
Analyze the following Pakistani job description and return ONLY a JSON object with no extra text, 
no markdown, no explanation. Just raw JSON.

Job Description:
{job_description}

Return exactly this structure:
{{
  "role_title": "string",
  "required_skills": ["skill1", "skill2"],
  "seniority_level": "Junior | Mid | Senior",
  "estimated_salary_pkr_range": "e.g. 80,000 - 120,000 PKR/month"
}}
"""

def safe_parse_json(text, required_keys):
    """Parse JSON and validate required keys exist."""
    try:
        data = json.loads(text)
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        return data
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return None
    except ValueError as e:
        print(f"Validation error: {e}")
        return None

def analyze_job(description):
    prompt = prompt_template.format(job_description=description.strip())
    response = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a Pakistani HR analyst. Always respond with raw JSON only."},
            {"role": "user",   "content": prompt}
        ],
        options={"temperature": 0.3, "top_p": 0.8}
    )
    return safe_parse_json(response.message.content, REQUIRED_KEYS)

def print_table(results):
    print(f"{'ROLE TITLE':<25} {'SENIORITY':<10} {'SALARY RANGE':<30} {'SKILLS'}")
    print("=" * 80)
    for r in results:
        if r:
            skills = ", ".join(r["required_skills"][:3]) 
            print(f"{r['role_title']:<25} {r['seniority_level']:<10} {r['estimated_salary_pkr_range']:<30} {skills}")
    

results = []
for i, jd in enumerate(job_descriptions, 1):
    result = analyze_job(jd)
    results.append(result)

print_table(results)