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

# Load environment variables
load_dotenv()
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
project_id = os.getenv('PROJECT_ID', 'ethica-vision-2024')

# Make credentials path absolute if it's relative
if credentials_path and not os.path.isabs(credentials_path):
    credentials_path = os.path.join(os.path.dirname(__file__), credentials_path)
    if os.path.exists(credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Clear any API key that might be set
if 'GOOGLE_API_KEY' in os.environ:
    del os.environ['GOOGLE_API_KEY']

print("📝 Loading configuration...")
print(f"Credentials path: {credentials_path}")
print(f"Project ID: {project_id}")

# ----------------- INITIALIZE APP -----------------
app = Flask(__name__)

# **CORS - ALLOW ALL ORIGINS**
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": False
     }})

# ----------------- GLOBAL CLIENTS (LAZY LOADED) -----------------
vision_client = None
model = None
_credentials = None

def get_credentials():
    global _credentials
    if _credentials is None:
        from google.oauth2 import service_account
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'ethica-service-account.json'),
            './ethica-service-account.json',
            'ethica-service-account.json',
            credentials_path if credentials_path else None
        ]
        
        creds_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                creds_path = path
                break
        
        if creds_path:
            _credentials = service_account.Credentials.from_service_account_file(creds_path)
            print(f"✅ Loaded credentials from: {creds_path}")
        else:
            # Fallback to default credentials (Cloud Run service account)
            import google.auth
            _credentials, _ = google.auth.default()
            print("✅ Using default Cloud Run credentials")
    return _credentials

def get_vision_client():
    global vision_client
    if vision_client is None:
        creds = get_credentials()
        vision_client = vision.ImageAnnotatorClient(credentials=creds)
        print("✅ Vision API client initialized")
    return vision_client

def get_gemini_model():
    global model
    if model is None:
        creds = get_credentials()
        genai.configure(credentials=creds)
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        print("✅ Gemini AI client initialized")
    return model


# ----------------- VALIDATION FUNCTIONS -----------------
def validate_ai_response(response: dict) -> dict:
    """Validate that AI response has all required fields and proper structure."""
    errors = []
    required_fields = {
        "environmental": ["totalCO2", "waterUsage", "animalImpact", "rating", "breakdown"],
        "allergens": ["definiteViolations", "cautionWarnings", "safe"],
        "dietary": ["compatible", "violations", "tags"],
        "health": ["score", "concerns", "benefits"],
        "recommendations": ["environmental", "health", "allergenFree", "alternatives", "insights"]
    }
    
    # Check top-level sections
    for section in required_fields.keys():
        if section not in response:
            errors.append(f"Missing required section: {section}")
            continue
            
        # Check fields within each section
        for field in required_fields[section]:
            if field not in response[section]:
                errors.append(f"Missing field: {section}.{field}")
    
    # Check confidence field (optional but recommended)
    if "confidence" not in response:
        errors.append("Missing recommended field: confidence")
    
    # Validate data types
    if "environmental" in response:
        env = response["environmental"]
        if "totalCO2" in env and not isinstance(env["totalCO2"], (int, float)):
            errors.append("environmental.totalCO2 must be a number")
        if "waterUsage" in env and not isinstance(env["waterUsage"], (int, float)):
            errors.append("environmental.waterUsage must be a number")
        if "animalImpact" in env and env["animalImpact"] not in ["Low", "Medium", "High"]:
            errors.append("environmental.animalImpact must be Low, Medium, or High")
    
    if "health" in response and "score" in response["health"]:
        score = response["health"]["score"]
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            errors.append("health.score must be a number between 1-10")
    
    if "allergens" in response and "safe" in response["allergens"]:
        if not isinstance(response["allergens"]["safe"], bool):
            errors.append("allergens.safe must be a boolean")
    
    if "confidence" in response:
        conf = response["confidence"]
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 100:
            errors.append("confidence must be a number between 0-100")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_ingredients(ingredients: list) -> dict:
    """Validate ingredients against common food items to catch obvious errors."""
    # Common food ingredients database (subset for quick validation)
    common_ingredients = {
        # Grains & Flours
        'wheat', 'flour', 'rice', 'corn', 'oat', 'barley', 'rye', 'quinoa', 'millet',
        'wheat flour', 'enriched wheat flour', 'whole wheat flour', 'corn flour', 'rice flour',
        'oat flour', 'corn starch', 'corn meal', 'semolina',
        
        # Sugars & Sweeteners  
        'sugar', 'glucose', 'fructose', 'sucrose', 'maltose', 'dextrose', 'lactose',
        'corn syrup', 'high fructose corn syrup', 'maple syrup', 'honey', 'molasses',
        'cane sugar', 'brown sugar', 'maltodextrin', 'aspartame', 'sucralose', 'stevia',
        
        # Fats & Oils
        'oil', 'butter', 'margarine', 'lard', 'shortening', 'palm oil', 'coconut oil',
        'olive oil', 'canola oil', 'soybean oil', 'sunflower oil', 'vegetable oil',
        'safflower oil', 'corn oil', 'peanut oil', 'sesame oil',
        
        # Dairy
        'milk', 'cream', 'cheese', 'yogurt', 'whey', 'casein', 'lactose', 'butter',
        'cheddar', 'mozzarella', 'parmesan', 'milk powder', 'skim milk', 'whole milk',
        'buttermilk', 'sour cream', 'cream cheese',
        
        # Proteins
        'egg', 'chicken', 'beef', 'pork', 'fish', 'shrimp', 'gelatin', 'albumin',
        'soy protein', 'whey protein', 'collagen', 'turkey', 'bacon', 'ham',
        
        # Vegetables & Fruits
        'tomato', 'onion', 'garlic', 'potato', 'carrot', 'celery', 'pepper', 'corn',
        'peas', 'beans', 'lentils', 'chickpeas', 'spinach', 'broccoli', 'cabbage',
        'apple', 'orange', 'lemon', 'lime', 'grape', 'raisin', 'date', 'fig',
        
        # Nuts & Seeds
        'peanut', 'almond', 'cashew', 'walnut', 'pecan', 'hazelnut', 'pistachio',
        'sesame', 'sunflower seed', 'flax seed', 'chia seed', 'pumpkin seed',
        'peanut butter', 'almond butter',
        
        # Seasonings & Additives
        'salt', 'pepper', 'paprika', 'cumin', 'cinnamon', 'vanilla', 'cocoa',
        'chocolate', 'yeast', 'baking soda', 'baking powder', 'vinegar',
        'citric acid', 'ascorbic acid', 'sodium benzoate', 'potassium sorbate',
        'xanthan gum', 'guar gum', 'lecithin', 'mono and diglycerides',
        'natural flavor', 'artificial flavor', 'natural flavoring', 'natural flavors',
        
        # Vitamins & Minerals
        'niacin', 'riboflavin', 'thiamine', 'folic acid', 'vitamin a', 'vitamin c',
        'vitamin d', 'vitamin e', 'calcium', 'iron', 'zinc', 'magnesium',
        'thiamine mononitrate', 'reduced iron', 'ferrous sulfate',
        
        # Soy products
        'soy', 'soybean', 'tofu', 'tempeh', 'soy sauce', 'soy lecithin', 'soy protein',
        
        # Preservatives & Emulsifiers
        'preservative', 'emulsifier', 'stabilizer', 'thickener', 'anticaking agent',
        'msg', 'monosodium glutamate', 'tbhq', 'bht', 'bha',
        
        # Other common
        'water', 'tapioca', 'arrowroot', 'modified food starch', 'food starch',
        'caramel color', 'annatto', 'turmeric', 'beetroot', 'cellulose'
    }
    
    suspicious_ingredients = []
    warnings = []
    
    for ingredient in ingredients:
        # Check if ingredient or any part of it matches common ingredients
        ingredient_lower = ingredient.lower()
        found = False
        
        # Direct match
        if ingredient_lower in common_ingredients:
            found = True
        else:
            # Check if any common ingredient is a substring
            for common in common_ingredients:
                if common in ingredient_lower or ingredient_lower in common:
                    found = True
                    break
        
        if not found:
            # Could be a compound ingredient or specific brand name
            # Check if it contains common words
            words = ingredient_lower.split()
            has_common_word = any(word in common_ingredients for word in words)
            
            if not has_common_word and len(ingredient) > 3:
                suspicious_ingredients.append(ingredient)
    
    # Generate warnings
    if len(suspicious_ingredients) > 0 and len(suspicious_ingredients) < len(ingredients) * 0.3:
        # If less than 30% are suspicious, it's probably fine
        warnings.append(f"Found {len(suspicious_ingredients)} uncommon ingredient(s): {', '.join(suspicious_ingredients[:3])}")
    elif len(suspicious_ingredients) >= len(ingredients) * 0.3:
        # If 30%+ are suspicious, OCR might have failed
        warnings.append(f"Warning: {len(suspicious_ingredients)} of {len(ingredients)} ingredients seem unusual - OCR may have errors")
    
    return {
        "valid": len(suspicious_ingredients) < len(ingredients) * 0.5,  # Valid if less than 50% suspicious
        "suspicious_count": len(suspicious_ingredients),
        "suspicious_ingredients": suspicious_ingredients[:5],  # Return max 5 examples
        "warnings": warnings,
        "confidence_adjustment": -10 if len(suspicious_ingredients) > 3 else 0
    }


# ----------------- HELPER FUNCTIONS -----------------
def extract_text_from_image(base64_image: str) -> str:
    """Extract text from image using Google Cloud Vision."""
    try:
        client = get_vision_client()
        content = base64.b64decode(base64_image)
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        
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
        gemini = get_gemini_model()
        response = gemini.generate_content(prompt)
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

def comprehensive_ai_analysis(ingredients: list, user_preferences: dict, dietary_preferences: list = None) -> dict:
    """Comprehensive AI analysis based on user's dietary preferences and allergens."""
    
    # Validate ingredients first
    ingredient_validation = validate_ingredients(ingredients)
    
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
    
    # Add custom allergens (anything not in the standard list)
    standard_allergens = ['gluten', 'dairy', 'nuts', 'soy', 'eggs', 'shellfish', 'peanuts', 'treeNuts']
    custom_allergens = []
    for key, value in user_preferences.items():
        if value and key not in standard_allergens and key not in ['vegan', 'vegetarian', 'pescatarian', 'kosher', 'halal', 'jain']:
            custom_allergens.append(key)
            preferences_list.append(key)
    
    # Use the new dietary_preferences array from toggles
    dietary_restrictions = dietary_preferences if dietary_preferences and 'none' not in dietary_preferences else []
    
    # Fallback to old preferences if no toggle selection
    if not dietary_restrictions:
        if user_preferences.get('vegan'): dietary_restrictions.append('vegan')
        if user_preferences.get('vegetarian'): dietary_restrictions.append('vegetarian')
        if user_preferences.get('pescatarian'): dietary_restrictions.append('pescatarian')
        if user_preferences.get('kosher'): dietary_restrictions.append('kosher')
        if user_preferences.get('halal'): dietary_restrictions.append('halal')
        if user_preferences.get('jain'): dietary_restrictions.append('jain')
    
    allergen_check = f"Check ONLY for these allergens: {', '.join(preferences_list)}" if preferences_list else "No specific allergens to check"
    dietary_check = f"Check compatibility with ALL of these diets (product must be compatible with ALL): {', '.join(dietary_restrictions)}" if dietary_restrictions else "No dietary restrictions specified"
    
    # Add ingredient quality warning to prompt if validation found issues
    ingredient_quality_note = ""
    if ingredient_validation['warnings']:
        ingredient_quality_note = f"\n\nWARNING: {ingredient_validation['warnings'][0]} - Please be extra careful in your analysis."
    
    # Build custom allergen note with derivative checking instructions
    custom_allergen_note = ""
    if custom_allergens:
        custom_allergen_note = f"\n\nCRITICAL - CUSTOM ALLERGENS: The user wants to avoid {', '.join(custom_allergens)}. You MUST:"
        custom_allergen_note += "\n- Check for EXACT matches (e.g., 'natural flavors' if user avoids 'natural flavoring')"
        custom_allergen_note += "\n- Check for DERIVATIVES and RELATED ingredients:"
        custom_allergen_note += "\n  * If user avoids 'potato': flag potato starch, potato flour, potato protein (but NOT tapioca - different plant)"
        custom_allergen_note += "\n  * If user avoids 'natural flavoring': flag natural flavors, natural flavor, natural flavoring"
        custom_allergen_note += "\n  * If user avoids 'corn': flag corn syrup, corn starch, maltodextrin, dextrose, corn oil"
        custom_allergen_note += "\n  * If user avoids 'sugar': flag cane sugar, organic cane sugar, glucose, sucrose, fructose"
        custom_allergen_note += "\n  * If user avoids 'milk': flag whey, casein, lactose, dairy derivatives"
        custom_allergen_note += "\n- Treat these as SEVERE violations if found (even if derivative/related)"
        custom_allergen_note += f"\n- Be STRICT: even trace amounts or 'may contain' warnings count as violations"
    
    # Note about custom diets
    custom_diet_note = ""
    if dietary_restrictions:
        known_diets = ['vegan', 'vegetarian', 'jain', 'halal', 'kosher', 'pescatarian', 'gluten-free', 'lactose-free', 'nut-free']
        custom = [d for d in dietary_restrictions if d.lower() not in known_diets]
        if custom:
            custom_diet_note = f"\n\nCUSTOM DIETARY RESTRICTIONS: User follows {', '.join(custom)}."
            custom_diet_note += "\n\nIMPORTANT - You MUST research and apply the scientifically accurate rules for each custom diet:"
            custom_diet_note += "\n\nCOMMON DIETS YOU SHOULD KNOW:"
            custom_diet_note += "\n- KETO/KETOGENIC: Very low carb (<20-50g/day). PROHIBITS: sugar, grains, bread, pasta, rice, potatoes, high-carb fruits, beans, most processed foods with carbs"
            custom_diet_note += "\n- PALEO: PROHIBITS: grains, legumes, dairy, refined sugar, processed foods, vegetable oils. ALLOWS: meat, fish, eggs, vegetables, fruits, nuts, seeds"
            custom_diet_note += "\n- LOW-FODMAP: PROHIBITS: onion, garlic, wheat, beans, certain fruits (apples, pears), lactose, high-fructose corn syrup"
            custom_diet_note += "\n- WHOLE30: PROHIBITS: sugar, alcohol, grains, legumes, dairy, processed foods, additives. Very strict 30-day elimination diet"
            custom_diet_note += "\n- CARNIVORE: ONLY allows animal products. PROHIBITS: all plant foods, vegetables, fruits, grains, nuts"
            custom_diet_note += "\n- MEDITERRANEAN: Emphasizes fish, olive oil, vegetables. Limits red meat and processed foods"
            custom_diet_note += "\n- DIABETIC/LOW-SUGAR: PROHIBITS/LIMITS: added sugars, high glycemic foods, refined carbs"
            custom_diet_note += "\n- RAW VEGAN: Only raw, uncooked plant foods. PROHIBITS: cooked foods, animal products"
            custom_diet_note += "\n- MACROBIOTIC: Whole grains, vegetables, beans. PROHIBITS: processed foods, nightshades, tropical fruits"
            custom_diet_note += "\n\nFor ANY custom diet not listed above:"
            custom_diet_note += "\n1. Use your knowledge to determine what the diet prohibits"
            custom_diet_note += "\n2. Check EVERY ingredient against those prohibitions"
            custom_diet_note += "\n3. Include derivatives (e.g., wheat flour counts as grain for Paleo)"
            custom_diet_note += "\n4. Be as strict as you would be for Vegan or Jain diets"
            custom_diet_note += "\n5. If the diet name is unclear or you're uncertain, research common restrictions for similar diets"
            custom_diet_note += "\n6. NEVER say 'Compatible' unless you're certain ALL ingredients are allowed"
    
    prompt = f"""
    You are a comprehensive food analysis expert covering environmental impact, allergens, nutrition, and health.
    You have deep knowledge of dietary restrictions worldwide and their specific rules.
    You are an expert in ALL popular diets including Keto, Paleo, Low-FODMAP, Whole30, Carnivore, Mediterranean, and more.
    
    Analyze these ingredients: {ingredients_str}
    
    USER'S PREFERENCES:
    - {allergen_check}
    - {dietary_check}{custom_allergen_note}{custom_diet_note}
    
    CRITICAL: Be extremely thorough in checking dietary compatibility. Research each diet's restrictions carefully.
    For custom diets, apply your knowledge of that specific diet's rules with the same rigor as standard diets.
    
    Provide a complete analysis including:
    
    1. ENVIRONMENTAL IMPACT:
       - Total CO2 emissions (kg CO2 per 100g) - Use realistic food industry averages:
         * Plant-based: 0.1-0.5 kg CO2
         * Dairy products: 0.8-2.0 kg CO2
         * Meat products: 2.0-10.0 kg CO2
       - Water usage (liters per 100g) - Use realistic averages:
         * Plant-based: 50-200 L
         * Dairy products: 300-600 L
         * Meat products: 1000-2000 L
         * DO NOT return 0 - always estimate a realistic value
       - Animal impact score (Low/Medium/High):
         * Low: NO animal products (100% plant-based)
         * Medium: Contains dairy OR eggs (but NO meat/fish)
         * High: Contains meat, fish, gelatin, or multiple animal products
       - Overall sustainability rating (Low/Medium/High)
       - Top 5 highest-impact ingredients with their individual CO2 values and percentages
    
    2. ALLERGEN ANALYSIS (ONLY check for user's selected allergens):
       IMPORTANT - Use THREE severity levels:
       
       a) DEFINITE violations (severity: "Severe"):
          - Allergen is DIRECTLY LISTED in ingredients (EXACT or DERIVATIVE match)
          - Example: "milk" is in ingredients and user avoids dairy
          - Example: "natural flavors" found and user avoids "natural flavoring"
          - Example: "potato starch" found and user avoids "potato"
          - FOR CUSTOM ALLERGENS: Use fuzzy matching - "natural flavors" matches "natural flavoring"
       
       b) POSSIBLE violations (severity: "Caution"):
          - "May contain" warnings
          - Cross-contamination risks
          - Processed in facility with allergens
          - Uncertain ingredients that MIGHT contain allergen
          - Ingredients that could be derived from the allergen
       
       c) SAFE:
          - No allergen present, no derivatives, no cross-contamination warnings
       
       CRITICAL RULES FOR CUSTOM ALLERGENS:
       - "natural flavoring" / "natural flavors" / "natural flavor" are ALL THE SAME - treat as exact match
       - "potato" includes: potato starch, potato flour, potato protein (NOT tapioca - different plant)
       - "corn" includes: corn syrup, corn starch, maltodextrin, dextrose, corn oil
       - "sugar" includes: cane sugar, organic cane sugar, glucose, sucrose, fructose
       - "milk" includes: whey, casein, lactose, dairy derivatives
       - Match ingredient names loosely (ignore plural/singular, "natural flavors" = "natural flavoring")
       
       For each allergen the user selected:
       - If DEFINITELY present (exact or derivative) → severity: "Severe", source: ingredient name
       - If POSSIBLY present → severity: "Caution", source: reason for caution
       - If safe → don't include in violations array
       
       Return separate arrays for "definiteViolations" and "cautionWarnings"
    
    3. DIETARY COMPATIBILITY:
       - Check if product meets user's dietary restrictions
       - List any violations (e.g., "Contains milk - not vegan")
       - Overall compatibility score (Compatible/Partially Compatible/Not Compatible)
       
       CRITICAL DIETARY RULES (MUST BE STRICTLY FOLLOWED):
       
       * Jain: 
         - ALLOWS: dairy (milk, cheese, yogurt, butter), eggs
         - PROHIBITS: ALL root vegetables including:
           * Onion, garlic, potato, sweet potato
           * Carrots, radish, turnip, beets
           * Ginger, turmeric (root form)
           * TAPIOCA (cassava root), arrowroot
           * Any ingredient derived from roots (potato starch, tapioca starch, etc.)
         - PROHIBITS: meat, fish, honey
         - If product contains ANY root vegetable or derivative → "Not Compatible"
       
       * Vegan: 
         - PROHIBITS: ALL animal products (meat, dairy, eggs, honey, gelatin, whey, casein, etc.)
         - If product contains ANY animal product → "Not Compatible"
       
       * Vegetarian: 
         - ALLOWS: dairy and eggs
         - PROHIBITS: meat, fish, seafood, gelatin
         - If product contains meat/fish → "Not Compatible"
       
       * Pescatarian: 
         - ALLOWS: dairy, eggs, fish, seafood
         - PROHIBITS: meat (beef, pork, chicken, etc.)
         - If product contains meat → "Not Compatible"
       
       * Halal:
         - PROHIBITS: pork, alcohol, non-halal meat
         - If product contains pork/alcohol → "Not Compatible"
       
       * Kosher:
         - PROHIBITS: pork, shellfish, mixing meat and dairy
         - If product contains prohibited items → "Not Compatible"
       
       * Gluten-Free:
         - PROHIBITS: wheat, barley, rye, malt, and derivatives
         - If product contains gluten → "Not Compatible"
       
       * Lactose-Free / Dairy-Free:
         - PROHIBITS: milk, cheese, butter, cream, whey, casein, lactose
         - If product contains dairy → "Not Compatible"
       
       * Nut-Free:
         - PROHIBITS: all tree nuts, peanuts, nut oils, nut derivatives
         - If product contains nuts → "Not Compatible"
       
       FOR CUSTOM DIETS: Use your knowledge to determine restrictions and check thoroughly.
       BE STRICT: Even trace amounts or derivatives count as violations.
    
    4. HEALTH ANALYSIS:
       - Nutritional concerns (high sodium, saturated fats, added sugars, artificial additives)
       - Health benefits (fiber, vitamins, minerals, antioxidants)
       - Overall health score (1-10, where 10 is healthiest)
    
    5. RECOMMENDATIONS:
       - Environmental alternatives (lower CO2/water usage)
       - Healthier alternatives
       - Allergen-free alternatives if violations found
       - General insights
    
    CRITICAL RULES FOR animalImpact:
    - Low: ONLY if 100% plant-based (NO dairy, eggs, meat, fish, gelatin, whey, casein, etc.)
    - Medium: Contains dairy (milk, cheese, butter, whey, casein) OR eggs (but NO meat/fish)
    - High: Contains ANY meat, poultry, fish, seafood, gelatin, or multiple animal products
    
    EXAMPLE:
    - "wheat flour, cheddar cheese, salt" = Medium (has dairy)
    - "wheat flour, salt, oil" = Low (100% plant-based)
    - "wheat flour, chicken flavor, cheese" = High (has meat ingredients)
    
    Return ONLY valid JSON in this EXACT format (no extra text):
    {{
      "confidence": <0-100>,
      "confidenceFactors": ["<factor1>", "<factor2>"],
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
        "alternatives": [
          {{"name": "<product name>", "reason": "<why it's better>", "brand": "<optional brand>"}}
        ],
        "insights": ["<insight1>", "<insight2>"]
      }}
    }}
    
    ALTERNATIVE PRODUCTS:
    - If product is NOT compatible with user's dietary restrictions, suggest 2-3 specific alternative products
    - Focus on popular, widely-available brands
    - Explain why the alternative is better (e.g., "Daiya Cheddar - 100% vegan, no dairy")
    - Consider the same product category (e.g., if analyzing cheese, suggest vegan cheese alternatives)
    
    CONFIDENCE ASSESSMENT:
    - Add a "confidence" field to your response with a score from 0-100
    - Base confidence on:
      * Clarity of ingredient list (100 if crystal clear, 70 if some ambiguity, 40 if very vague)
      * Completeness of nutritional data available
      * Certainty about dietary compatibility
      * Quality of OCR text extraction
    - Include "confidenceFactors" array explaining what affected the score
    - Example: {{"confidence": 85, "confidenceFactors": ["Clear ingredient list", "Minor ambiguity about natural flavors source"]}}{ingredient_quality_note}
    """
    
    try:
        gemini = get_gemini_model()
        response = gemini.generate_content(prompt)
        if not response.text:
            return {"error": "No response from AI"}
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Adjust confidence based on ingredient validation
            if 'confidence' in result and ingredient_validation['confidence_adjustment'] != 0:
                original_confidence = result['confidence']
                result['confidence'] = max(0, min(100, original_confidence + ingredient_validation['confidence_adjustment']))
                
                if 'confidenceFactors' not in result:
                    result['confidenceFactors'] = []
                
                if ingredient_validation['warnings']:
                    result['confidenceFactors'].append(f"Ingredient validation: {ingredient_validation['warnings'][0]}")
            
            # Validate response structure
            validation_result = validate_ai_response(result)
            if not validation_result["valid"]:
                print(f"⚠️ Invalid AI response: {validation_result['errors']}")
                return {
                    "error": "AI returned incomplete response",
                    "details": validation_result['errors'],
                    "partialData": result
                }
            
            print(f"✅ Comprehensive Analysis Complete (Confidence: {result.get('confidence', 'N/A')}%)")
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
        gemini = get_gemini_model()
        response = gemini.generate_content(prompt)
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
    dietary_preferences = data.get("dietaryPreferences", [])
    
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    
    try:
        analysis = comprehensive_ai_analysis(ingredients, user_preferences, dietary_preferences)
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