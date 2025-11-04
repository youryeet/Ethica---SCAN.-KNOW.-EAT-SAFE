from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
from dotenv import load_dotenv 
import requests
from google.cloud import vision_v1 as vision
import google.generativeai as genai
import json
import re

# ----------------- LOAD ENVIRONMENT VARIABLES -----------------
load_dotenv()
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
project_id = os.getenv('PROJECT_ID')

# Clear any API key that might be set
if 'GOOGLE_API_KEY' in os.environ:
    del os.environ['GOOGLE_API_KEY']

print("📝 Loading configuration...")
print(f"Credentials path: {credentials_path}")
print(f"Project ID: {project_id}")

# ----------------- VALIDATE CONFIGURATION -----------------
if not credentials_path or not os.path.exists(credentials_path):
    raise ValueError(
        "❌ Invalid GOOGLE_APPLICATION_CREDENTIALS path. "
        "Check your .env file and ensure the service account JSON exists."
    )

# ----------------- INITIALIZE APP -----------------
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5501"], supports_credentials=True)

# ----------------- INITIALIZE GOOGLE CLIENTS -----------------
try:
    # Initialize Vision API client
    vision_client = vision.ImageAnnotatorClient()
    print("✅ Vision API client initialized")
    
    # Initialize Gemini AI with explicit credentials
    import google.auth
    credentials, project = google.auth.default()
    genai.configure(credentials=credentials)
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    print("✅ Gemini AI client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Google clients: {str(e)}")
    raise


# ----------------- HELPER FUNCTIONS -----------------
def extract_text_from_image(base64_image: str) -> str:
    """Extract text from image using Google Cloud Vision."""
    try:
        content = base64.b64decode(base64_image)
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        
        if not response.text_annotations:
            return ""
        
        return response.text_annotations[0].description
    except Exception as e:
        print(f"❌ OCR error: {str(e)}")
        raise

def process_ingredients_with_gemini(text: str) -> list:
    """Process OCR text with Gemini AI to extract ingredients in English."""
    prompt = f"""
    You are an expert food label analyzer. Extract ALL ingredients from this food product label text.

    CRITICAL INSTRUCTIONS:
    1. Find the "INGREDIENTS:" section (may be in any language)
    2. List EVERY SINGLE ingredient, including sub-ingredients in parentheses
    3. For compound ingredients like "ENRICHED WHEAT FLOUR (WHEAT FLOUR, NIACIN, REDUCED IRON, THIAMINE MONONITRATE, RIBOFLAVIN, FOLIC ACID)", list each component
    4. **TRANSLATE ALL INGREDIENTS TO ENGLISH** if they are in another language
    5. Return as a simple comma-separated list IN ENGLISH
    6. Use lowercase
    7. Keep full ingredient names (don't abbreviate)
    
    Example:
    If the label says "INGREDIENTES: HARINA DE TRIGO, QUESO CHEDDAR, ACEITE DE PALMA"
    Return: wheat flour, cheddar cheese, palm oil
    
    If the label says "成分: 小麦粉、チーズ、パーム油"
    Return: wheat flour, cheese, palm oil

    Now extract from this text and return ONLY in English:
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            return []
        
        # Get the response text and clean it
        raw_text = response.text.strip()
        
        # Replace newlines and semicolons with commas
        raw_text = raw_text.replace('\n', ',').replace(';', ',').replace('  ', ' ')
        
        # Split by comma
        ingredients = []
        for item in raw_text.split(','):
            # Basic cleaning
            cleaned = item.strip().lower()
            
            # Remove common prefixes/suffixes
            for prefix in ['ingredients:', 'ingredient:', '- ', '* ', '• ']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Remove only leading/trailing numbers and dots, not internal ones
            cleaned = cleaned.strip('0123456789.-*•() ')
            
            # Only add if it's a valid ingredient (at least 3 characters)
            if cleaned and len(cleaned) >= 3:
                ingredients.append(cleaned)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ingredients = []
        for ing in ingredients:
            if ing not in seen:
                seen.add(ing)
                unique_ingredients.append(ing)
        
        print(f"✅ Extracted {len(unique_ingredients)} ingredients (in English): {unique_ingredients[:5]}...")
        return unique_ingredients
        
    except Exception as e:
        print(f"❌ Gemini AI error: {str(e)}")
        raise

def comprehensive_ai_analysis(ingredients: list, user_preferences: dict) -> dict:
    """Comprehensive AI analysis based on user's dietary preferences and allergens."""
    ingredients_str = ", ".join(ingredients)
    
    # Build the allergen/dietary preferences list
    preferences_list = []
    if user_preferences.get('gluten'): preferences_list.append('gluten')
    if user_preferences.get('dairy'): preferences_list.append('dairy/milk')
    if user_preferences.get('nuts'): preferences_list.append('nuts (all types)')
    if user_preferences.get('soy'): preferences_list.append('soy')
    if user_preferences.get('eggs'): preferences_list.append('eggs')
    if user_preferences.get('shellfish'): preferences_list.append('shellfish')
    if user_preferences.get('peanuts'): preferences_list.append('peanuts')
    if user_preferences.get('treeNuts'): preferences_list.append('tree nuts')
    
    dietary_restrictions = []
    if user_preferences.get('vegan'): dietary_restrictions.append('vegan')
    if user_preferences.get('vegetarian'): dietary_restrictions.append('vegetarian')
    if user_preferences.get('pescatarian'): dietary_restrictions.append('pescatarian')
    if user_preferences.get('kosher'): dietary_restrictions.append('kosher')
    if user_preferences.get('halal'): dietary_restrictions.append('halal')
    if user_preferences.get('jain'): dietary_restrictions.append('jain')
    
    allergen_check = f"Check ONLY for these allergens: {', '.join(preferences_list)}" if preferences_list else "No specific allergens to check"
    dietary_check = f"Check compatibility with: {', '.join(dietary_restrictions)}" if dietary_restrictions else "No dietary restrictions specified"
    
    prompt = f"""
    You are a comprehensive food analysis expert covering environmental impact, allergens, nutrition, and health.
    
    Analyze these ingredients: {ingredients_str}
    
    USER'S PREFERENCES:
    - {allergen_check}
    - {dietary_check}
    
    Provide a complete analysis including:
    
    1. ENVIRONMENTAL IMPACT:
       - Total CO2 emissions (kg CO2 per 100g)
       - Water usage (liters per 100g)
       - Animal impact score (Low/Medium/High - based on animal products like meat, dairy, eggs)
       - Overall sustainability rating (Low/Medium/High)
       - Top 5 highest-impact ingredients with their individual CO2 values and percentages
    
    2. ALLERGEN ANALYSIS (ONLY check for user's selected allergens):
       IMPORTANT - Use THREE severity levels:
       
       a) DEFINITE violations (severity: "Severe"):
          - Allergen is DIRECTLY LISTED in ingredients
          - Example: "milk" is in ingredients and user avoids dairy
       
       b) POSSIBLE violations (severity: "Caution"):
          - "May contain" warnings
          - Cross-contamination risks
          - Processed in facility with allergens
          - Uncertain ingredients that MIGHT contain allergen
          - Example: "natural flavors" might contain dairy
       
       c) SAFE:
          - No allergen present, no cross-contamination warnings
       
       For each allergen the user selected:
       - If DEFINITELY present → severity: "Severe", source: ingredient name
       - If POSSIBLY present → severity: "Caution", source: reason for caution
       - If safe → don't include in violations array
       
       Return separate arrays for "definiteViolations" and "cautionWarnings"
    
    3. DIETARY COMPATIBILITY:
       - Check if product meets user's dietary restrictions
       - List any violations (e.g., "Contains milk - not vegan")
       - Overall compatibility score (Compatible/Partially Compatible/Not Compatible)
    
    4. HEALTH ANALYSIS:
       - Nutritional concerns (high sodium, saturated fats, added sugars, artificial additives)
       - Health benefits (fiber, vitamins, minerals, antioxidants)
       - Overall health score (1-10, where 10 is healthiest)
    
    5. RECOMMENDATIONS:
       - Environmental alternatives (lower CO2/water usage)
       - Healthier alternatives
       - Allergen-free alternatives if violations found
       - General insights
    
    IMPORTANT: For "animalImpact", consider:
    - Low: Plant-based, no animal products
    - Medium: Contains dairy or eggs but no meat
    - High: Contains meat, fish, or multiple animal products
    
    Return ONLY valid JSON in this EXACT format (no extra text):
    {{
      "environmental": {{
        "totalCO2": <number>,
        "waterUsage": <number>,
        "animalImpact": "<Low/Medium/High>",
        "rating": "<Low/Medium/High>",
        "breakdown": [
          {{"ingredient": "<name>", "co2": <number>, "percentage": <number>}}
        ]
      }},
      "allergens": {{
        "definiteViolations": [
          {{"allergen": "<name>", "severity": "Severe", "source": "<ingredient>", "warning": "<message>"}}
        ],
        "cautionWarnings": [
          {{"allergen": "<name>", "severity": "Caution", "source": "<reason>", "warning": "<message>"}}
        ],
        "safe": <true/false>
      }},
      "dietary": {{
        "compatible": "<Compatible/Partially Compatible/Not Compatible>",
        "violations": ["<violation1>", "<violation2>"],
        "tags": ["<tag1>", "<tag2>"]
      }},
      "health": {{
        "score": <1-10>,
        "concerns": ["<concern1>", "<concern2>"],
        "benefits": ["<benefit1>", "<benefit2>"]
      }},
      "recommendations": {{
        "environmental": ["<alt1>", "<alt2>"],
        "health": ["<alt1>", "<alt2>"],
        "allergenFree": ["<alt1>", "<alt2>"],
        "insights": ["<insight1>", "<insight2>"]
      }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            return {"error": "No response from AI"}
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            print(f"✅ Comprehensive Analysis Complete")
            return result
        else:
            return {
                "error": "Unable to parse AI response",
                "rawResponse": response.text
            }
        
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {str(e)}")
        return {
            "error": str(e),
            "environmental": {
                "totalCO2": 0,
                "waterUsage": 0,
                "animalImpact": "Unknown",
                "rating": "Unknown",
                "breakdown": []
            },
            "allergens": {
                "definiteViolations": [],
                "cautionWarnings": [],
                "safe": True
            },
            "dietary": {"compatible": "Unknown", "violations": [], "tags": []},
            "health": {"score": 0, "concerns": [], "benefits": []},
            "recommendations": {"environmental": [], "health": [], "allergenFree": [], "insights": []}
        }

def calculate_co2_impact_with_ai(ingredients: list) -> dict:
    """Calculate CO2 impact of ingredients using Gemini AI."""
    ingredients_str = ", ".join(ingredients)
    
    prompt = f"""
    You are an environmental impact expert specializing in food production carbon footprints.
    
    Analyze the CO2 emissions for these food ingredients: {ingredients_str}
    
    Based on scientific research and lifecycle assessments, provide:
    1. Total estimated CO2 emissions in kg CO2 for a typical serving (100g) of a product with these ingredients
    2. Per-ingredient CO2 impact breakdown (only for major contributors)
    3. Overall sustainability rating (Low/Medium/High impact)
    4. Key environmental concerns
    5. Suggestions for lower-impact alternatives
    
    Consider:
    - Agricultural production emissions
    - Processing and manufacturing
    - Transportation (assume average supply chain)
    - Packaging materials
    
    Return ONLY valid JSON in this exact format:
    {{
      "totalCO2": <number in kg>,
      "rating": "<Low/Medium/High>",
      "breakdown": [
        {{"ingredient": "<name>", "co2": <kg>, "percentage": <number>}}
      ],
      "concerns": ["<concern1>", "<concern2>"],
      "alternatives": ["<alt1>", "<alt2>"]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            return {"error": "No response from AI"}
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            print(f"✅ CO2 Analysis: {result.get('totalCO2', 0)} kg CO2, Rating: {result.get('rating', 'Unknown')}")
            return result
        else:
            return {
                "totalCO2": 0,
                "rating": "Unknown",
                "breakdown": [],
                "concerns": ["Unable to parse AI response"],
                "alternatives": []
            }
        
    except Exception as e:
        print(f"❌ CO2 calculation error: {str(e)}")
        return {
            "totalCO2": 0,
            "rating": "Error",
            "breakdown": [],
            "concerns": [str(e)],
            "alternatives": []
        }

# ----------------- API ENDPOINTS -----------------
@app.route("/extract-ingredients", methods=["POST"])
def extract_ingredients():
    """Extract ingredients from food package image."""
    data = request.get_json()
    image_base64 = data.get("imageBase64")
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
        
    try:
        # Step 1: Extract text with OCR
        text = extract_text_from_image(image_base64)
        if not text.strip():
            return jsonify({"error": "No text found in image"}), 400
            
        # Step 2: Process with Gemini (auto-translates to English)
        ingredients = process_ingredients_with_gemini(text)
        if not ingredients:
            return jsonify({"error": "No ingredients identified"}), 400
            
        return jsonify({"ingredients": ingredients})
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/comprehensive-analysis", methods=["POST"])
def comprehensive_analysis():
    """Comprehensive AI analysis with user preferences."""
    data = request.get_json()
    ingredients = data.get("ingredients", [])
    user_preferences = data.get("userPreferences", {})
    
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    
    try:
        analysis = comprehensive_ai_analysis(ingredients, user_preferences)
        return jsonify(analysis)
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-co2", methods=["POST"])
def analyze_co2():
    """Calculate CO2 impact of ingredients using AI."""
    data = request.get_json()
    ingredients = data.get("ingredients", [])
    
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    
    try:
        co2_analysis = calculate_co2_impact_with_ai(ingredients)
        return jsonify(co2_analysis)
    except Exception as e:
        print(f"❌ CO2 analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/ocr", methods=["POST"])
def ocr_only():
    """Extract raw text from image."""
    data = request.get_json()
    image_base64 = data.get("imageBase64")
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
        
    try:
        text = extract_text_from_image(image_base64)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate():
    """Translate text using LibreTranslate."""
    data = request.get_json()
    text = data.get("text")
    target_lang = data.get("targetLang")
    
    if not text or not target_lang:
        return jsonify({"error": "Missing text or target language"}), 400
        
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            json={
                "q": text,
                "source": "auto",
                "target": target_lang,
                "format": "text"
            },
            headers={"Content-Type": "application/json"}
        )
        translated = response.json().get("translatedText")
        return jsonify({"translatedText": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test-cors", methods=["GET"])
def test_cors():
    """Test CORS configuration."""
    return jsonify({"message": "CORS is working!"})

# ----------------- START SERVER -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)