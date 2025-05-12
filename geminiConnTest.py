import google.generativeai as genai

def test_gemini_connection():
    try:
        # Read API key from file
        with open('gemini_key.txt', 'r') as file:
            api_key = file.read().strip()
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say hello!")
        if hasattr(response, "text") and "hello" in response.text.lower():
            print("Gemini API connection successful!")
            return True
        else:
            print("Connected, but unexpected response from Gemini API.")
            return False
    except FileNotFoundError:
        print("Error: gemini_key.txt file not found. Please create the file with your API key.")
        return False
    except Exception as e:
        print(f"Gemini API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini_connection()