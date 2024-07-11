import json
import random
from datetime import datetime, timedelta

# Function to generate random date
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Function to generate random visit type and details
def random_visit():
    visits = [
        ("Routine Checkup", "Patient reports headache, advised to rest and hydrate.", {"advice": "rest, hydrate", "symptoms": "headache"}),
        ("Routine Checkup", "Patient reports chest pain, scheduled for EKG and stress test.", {"tests": "EKG, stress test", "symptoms": "chest pain"}),
        ("Diabetes Follow-up", "HbA1c levels reviewed, metformin prescribed.", {"HbA1c": "7.2", "condition": "diabetes", "medications": "metformin"}),
        ("Sick Visit", "Patient diagnosed with bronchitis, prescribed antibiotics.", {"symptoms": "cough, wheezing", "diagnosis": "bronchitis", "treatment": "antibiotics"}),
        ("Specialist Referral", "Referred to dermatology for suspicious mole.", {"referral": "dermatology", "condition": "suspicious mole"}),
        ("Wellness Visit", "Vital signs normal, flu vaccination administered.", {"vital_signs": "normal", "preventive_care": "flu vaccination"}),
        ("Annual Physical", "Blood pressure and cholesterol levels normal.", {"vital_signs": "normal", "tests": "blood pressure, cholesterol"}),
        ("Pediatric Checkup", "Child's growth and development on track.", {"development": "normal", "advice": "balanced diet"}),
        ("Cardiology Follow-up", "Echo and ECG normal, continue medication.", {"tests": "Echo, ECG", "condition": "heart disease", "medications": "continue"}),
        ("Orthopedic Visit", "Patient has knee pain, advised physiotherapy.", {"symptoms": "knee pain", "treatment": "physiotherapy"}),
        ("Gynecology Visit", "Routine pelvic exam, Pap smear conducted.", {"tests": "Pap smear", "advice": "routine follow-up"}),
        ("ENT Visit", "Patient reports sinusitis symptoms, prescribed nasal spray.", {"symptoms": "sinusitis", "treatment": "nasal spray"}),
        ("Allergy Test", "Patient tested for seasonal allergies, prescribed antihistamines.", {"tests": "allergy test", "treatment": "antihistamines"}),
        ("Dermatology Follow-up", "Skin rash examined, prescribed topical cream.", {"symptoms": "skin rash", "treatment": "topical cream"}),
        ("Psychiatric Evaluation", "Patient reports anxiety, prescribed medication.", {"condition": "anxiety", "treatment": "medication"}),
        ("Nephrology Visit", "Kidney function tests normal, continue current treatment.", {"tests": "kidney function", "advice": "continue treatment"}),
        ("Gastroenterology Visit", "Patient reports abdominal pain, scheduled for endoscopy.", {"symptoms": "abdominal pain", "tests": "endoscopy"}),
        ("Neurology Visit", "Patient reports migraines, prescribed medication.", {"symptoms": "migraines", "treatment": "medication"}),
        ("Ophthalmology Visit", "Routine eye exam, vision correction updated.", {"tests": "eye exam", "treatment": "vision correction"}),
        ("Urology Visit", "Patient reports frequent urination, scheduled for tests.", {"symptoms": "frequent urination", "tests": "urological tests"})
    ]
    return random.choice(visits)

# Generate sample data
start_date = datetime(2024, 7, 1)
end_date = datetime(2024, 7, 11)
num_records = 20

data = []
for i in range(1, num_records + 1):
    patient_id = f'P{i:03}'  # Patient ID with leading zeros
    date = random_date(start_date, end_date).strftime("%Y-%m-%d")
    visit_type, notes, details = random_visit()
    record = {
        "patient_id": patient_id,
        "date": date,
        "type_of_visit": visit_type,
        "note": notes,
        "metadata": details
    }
    data.append(record)
    print(f"Record {i}: {record}")

# Write to JSONL
with open("data/sample_generated_data.jsonl", "w") as jsonlfile:
    for record in data:
        jsonlfile.write(json.dumps(record) + "\n")

print("Sample data generated and saved to sample_data.jsonl")
