"""
Skin Disease Detection - Healthcare Recommendations Module

This module contains healthcare recommendations and utility functions
for the AI-based skin disease detection system.

Author: Elizabeth Chacko
License: MIT
"""

# Healthcare recommendations database
HEALTHCARE_RECOMMENDATIONS = {
    'Acne': {
        'description': 'Acne is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells.',
        'severity': 'Mild to Moderate',
        'recommendations': [
            'Wash face twice daily with gentle cleanser',
            'Use non-comedogenic skincare products',
            'Apply topical treatments with benzoyl peroxide or salicylic acid',
            'Avoid touching or picking at acne',
            'Consider consulting a dermatologist for persistent acne',
            'Maintain a healthy diet and manage stress levels'
        ],
        'when_to_see_doctor': 'If acne is severe, painful, or leaves scars',
        'treatment_options': [
            'Over-the-counter: Benzoyl peroxide, salicylic acid',
            'Prescription: Retinoids, antibiotics',
            'Procedures: Chemical peels, laser therapy'
        ]
    },
    'Dermatitis': {
        'description': 'Dermatitis is a general term for skin inflammation that can cause redness, itching, and swelling.',
        'severity': 'Moderate',
        'recommendations': [
            'Identify and avoid triggers (allergens, irritants)',
            'Keep skin moisturized with fragrance-free lotions',
            'Use mild, hypoallergenic soaps and detergents',
            'Apply cool compresses to affected areas',
            'Consider over-the-counter hydrocortisone cream',
            'Wear soft, breathable fabrics'
        ],
        'when_to_see_doctor': 'If symptoms persist for more than a week or worsen',
        'treatment_options': [
            'Topical corticosteroids',
            'Antihistamines for itching',
            'Moisturizers and emollients',
            'Phototherapy in severe cases'
        ]
    },
    'Eczema': {
        'description': 'Eczema (atopic dermatitis) causes dry, itchy, and inflamed skin patches.',
        'severity': 'Moderate to Severe',
        'recommendations': [
            'Apply moisturizer frequently throughout the day',
            'Take short, lukewarm baths or showers',
            'Use fragrance-free, gentle cleansers',
            'Avoid known triggers (stress, certain fabrics, soaps)',
            'Use prescribed topical corticosteroids if recommended',
            'Keep nails short to prevent scratching damage',
            'Consider using a humidifier in dry environments'
        ],
        'when_to_see_doctor': 'If eczema interferes with daily activities or sleep',
        'treatment_options': [
            'Topical corticosteroids',
            'Calcineurin inhibitors',
            'Systemic medications for severe cases',
            'Phototherapy',
            'Biologics (for severe eczema)'
        ]
    },
    'Melanoma': {
        'description': 'Melanoma is a serious type of skin cancer that develops in melanocytes.',
        'severity': 'High - Requires Immediate Medical Attention',
        'recommendations': [
            '⚠️ URGENT: Consult a dermatologist or oncologist IMMEDIATELY',
            'Do NOT attempt self-treatment',
            'Schedule a professional skin biopsy',
            'Avoid sun exposure and use high SPF sunscreen',
            'Monitor any changes in moles or skin lesions',
            'Early detection is crucial for successful treatment',
            'Discuss treatment options with your healthcare provider'
        ],
        'when_to_see_doctor': 'IMMEDIATELY - This requires urgent medical evaluation',
        'treatment_options': [
            'Surgical excision',
            'Sentinel lymph node biopsy',
            'Immunotherapy',
            'Targeted therapy',
            'Radiation therapy',
            'Chemotherapy (in advanced cases)'
        ],
        'warning_signs': [
            'Asymmetry in mole shape',
            'Border irregularity',
            'Color variation',
            'Diameter greater than 6mm',
            'Evolution or changes over time'
        ]
    },
    'Normal': {
        'description': 'Healthy skin with no apparent disease or condition detected.',
        'severity': 'None',
        'recommendations': [
            'Maintain your current skincare routine',
            'Use sunscreen daily (SPF 30 or higher)',
            'Stay hydrated and eat a balanced diet',
            'Get adequate sleep and manage stress',
            'Cleanse face daily and moisturize',
            'Perform regular skin self-examinations',
            'Schedule annual dermatology check-ups'
        ],
        'when_to_see_doctor': 'For routine annual skin check or if you notice any changes',
        'prevention_tips': [
            'Avoid excessive sun exposure',
            'Wear protective clothing',
            'Stay hydrated',
            'Eat antioxidant-rich foods',
            'Avoid smoking'
        ]
    },
    'Psoriasis': {
        'description': 'Psoriasis is a chronic autoimmune condition causing rapid skin cell buildup, resulting in scaling and inflammation.',
        'severity': 'Moderate to Severe',
        'recommendations': [
            'Keep skin moisturized with thick creams or ointments',
            'Take daily lukewarm baths with colloidal oatmeal',
            'Get regular, brief sun exposure (with caution)',
            'Avoid triggers like stress, infections, and skin injuries',
            'Use prescribed topical treatments consistently',
            'Consider phototherapy under medical supervision',
            'Join support groups for coping strategies'
        ],
        'when_to_see_doctor': 'For proper diagnosis and prescription treatments',
        'treatment_options': [
            'Topical treatments (corticosteroids, vitamin D analogs)',
            'Phototherapy (UVB, PUVA)',
            'Systemic medications (methotrexate, cyclosporine)',
            'Biologic drugs (TNF-alpha inhibitors)',
            'Lifestyle modifications'
        ]
    },
    'Warts': {
        'description': 'Warts are small, rough growths caused by human papillomavirus (HPV) infection.',
        'severity': 'Mild',
        'recommendations': [
            'Try over-the-counter salicylic acid treatments',
            'Keep the affected area clean and dry',
            'Avoid picking or scratching warts',
            'Cover warts with bandages to prevent spreading',
            'Do not share personal items like towels',
            'Consider cryotherapy (freezing) by a healthcare provider',
            'Boost immune system with healthy lifestyle'
        ],
        'when_to_see_doctor': 'If warts multiply, are painful, or persist after treatment',
        'treatment_options': [
            'Salicylic acid (over-the-counter)',
            'Cryotherapy (freezing)',
            'Electrosurgery',
            'Laser treatment',
            'Immunotherapy',
            'Chemical peels'
        ]
    }
}


def get_recommendation(disease_name):
    """
    Get healthcare recommendation for a specific disease.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        dict: Healthcare recommendations or None if not found
    """
    return HEALTHCARE_RECOMMENDATIONS.get(disease_name, None)


def format_recommendation(disease_name):
    """
    Format healthcare recommendation as a readable string.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        str: Formatted recommendation text
    """
    rec = get_recommendation(disease_name)
    
    if not rec:
        return f"No recommendations available for {disease_name}"
    
    output = []
    output.append("=" * 80)
    output.append(f"HEALTHCARE RECOMMENDATIONS FOR {disease_name.upper()}")
    output.append("=" * 80)
    output.append(f"\n📝 Description:")
    output.append(f"   {rec['description']}")
    output.append(f"\n⚠️  Severity Level: {rec['severity']}")
    output.append(f"\n💊 Recommendations:")
    
    for recommendation in rec['recommendations']:
        output.append(f"   {recommendation}")
    
    if 'treatment_options' in rec:
        output.append(f"\n🏥 Treatment Options:")
        for option in rec['treatment_options']:
            output.append(f"   • {option}")
    
    if 'warning_signs' in rec:
        output.append(f"\n⚠️  Warning Signs (ABCDE):")
        for sign in rec['warning_signs']:
            output.append(f"   • {sign}")
    
    output.append(f"\n🏥 When to See a Doctor:")
    output.append(f"   {rec['when_to_see_doctor']}")
    
    output.append("\n" + "=" * 80)
    output.append("⚠️  DISCLAIMER: This is an AI-based prediction for educational purposes.")
    output.append("   Always consult with qualified healthcare professionals for accurate")
    output.append("   diagnosis and treatment. Do not rely solely on this system.")
    output.append("=" * 80)
    
    return "\n".join(output)


def get_all_diseases():
    """
    Get list of all supported diseases.
    
    Returns:
        list: List of disease names
    """
    return list(HEALTHCARE_RECOMMENDATIONS.keys())


def get_severity_level(disease_name):
    """
    Get severity level for a specific disease.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        str: Severity level or None
    """
    rec = get_recommendation(disease_name)
    return rec['severity'] if rec else None


def is_emergency(disease_name):
    """
    Check if a disease requires immediate medical attention.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        bool: True if emergency, False otherwise
    """
    emergency_keywords = ['IMMEDIATE', 'URGENT', 'High']
    rec = get_recommendation(disease_name)
    
    if not rec:
        return False
    
    severity = rec.get('severity', '')
    doctor = rec.get('when_to_see_doctor', '')
    
    return any(keyword in severity or keyword in doctor 
               for keyword in emergency_keywords)


# Example usage
if __name__ == "__main__":
    print("Skin Disease Healthcare Recommendations Module\n")
    
    # List all diseases
    print("Supported diseases:")
    for i, disease in enumerate(get_all_diseases(), 1):
        severity = get_severity_level(disease)
        emergency = "⚠️ EMERGENCY" if is_emergency(disease) else ""
        print(f"  {i}. {disease} - {severity} {emergency}")
    
    print("\n" + "=" * 80)
    
    # Example: Get recommendation for Melanoma
    print("\nExample: Melanoma Recommendations")
    print(format_recommendation('Melanoma'))
