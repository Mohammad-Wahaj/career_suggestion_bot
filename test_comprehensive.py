import requests
import json

def test_comprehensive_examples():
    url = "http://localhost:5000/career_advice"
    
    test_cases = [
        {
            "name": "Medicine Interest (Roman Urdu)",
            "input": "mujhe medicine me interest hai, hardworking hun aur logo ki madad krna acha lagta hai"
        },
        {
            "name": "Computer Interest (Roman Urdu)", 
            "input": "mujhe computer aur programming mein interest hai, coding karna pasand hai"
        },
        {
            "name": "Science Interest (Roman Urdu)",
            "input": "mujhe science aur research mein interest hai, laboratory work karna pasand hai"
        },
        {
            "name": "Law Interest (Roman Urdu)",
            "input": "mujhe law aur justice mein interest hai, qanoon padhna pasand hai"
        },
        {
            "name": "Agriculture Interest (Roman Urdu)",
            "input": "mujhe farming aur agriculture mein interest hai, kheti karna pasand hai"
        },
        {
            "name": "Sports Interest (Roman Urdu)",
            "input": "mujhe sports aur fitness mein interest hai, gym karna pasand hai"
        },
        {
            "name": "Business Interest (English)",
            "input": "I love business and marketing, want to make money and help companies grow"
        },
        {
            "name": "Engineering Interest (English)",
            "input": "I love engineering and building things, want to work with machines and technology"
        },
        {
            "name": "Media Interest (English)",
            "input": "I love journalism and communication, want to work in media and news"
        },
        {
            "name": "Hospitality Interest (English)",
            "input": "I love hospitality and customer service, want to work in hotels and restaurants"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print("-" * 80)
        
        try:
            response = requests.post(url, json={"user_input": test_case['input']})
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Language: {result['language_detected']}")
                print(f"✅ Response: {result['message']}")
                print(f"✅ Top 3 Careers: {', '.join(result['career_recommendations'][:3])}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_comprehensive_examples()
