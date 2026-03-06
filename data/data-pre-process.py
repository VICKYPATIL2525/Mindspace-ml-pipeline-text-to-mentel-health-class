import re
from collections import Counter


# -------------------------------
# Tokenization Function
# -------------------------------

def tokenize_text(text: str):
    """
    Tokenizes English, Hindi, and Marathi text.
    Removes punctuation and splits on whitespace.
    """

    # normalize case
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # split into tokens
    tokens = text.split()

    return tokens


# -------------------------------
# Moving Average TTR
# -------------------------------

def moving_average_ttr(tokens, window_size=50):
    """
    Calculates Moving Average Type Token Ratio (MATTR)
    """

    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0

    ttr_values = []

    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttr = len(set(window)) / window_size
        ttr_values.append(ttr)

    return sum(ttr_values) / len(ttr_values)


# -------------------------------
# Feature Extraction Function
# -------------------------------

def extract_lexical_features(text: str):
    """
    Extracts lexical diversity features from transcript text
    """

    tokens = tokenize_text(text)

    total_word_count = len(tokens)
    unique_word_count = len(set(tokens))

    # Type Token Ratio
    ttr = unique_word_count / total_word_count if total_word_count > 0 else 0

    # Moving Average TTR
    mattr = moving_average_ttr(tokens)

    # Hapax Legomena
    token_counts = Counter(tokens)
    hapax_count = sum(1 for word, count in token_counts.items() if count == 1)

    hapax_ratio = hapax_count / total_word_count if total_word_count > 0 else 0

    return {
        "total_word_count": total_word_count,
        "unique_word_count": unique_word_count,
        "type_token_ratio": ttr,
        "moving_average_ttr": mattr,
        "hapax_legomena_ratio": hapax_ratio
    }


# -------------------------------
# Example Usage
# -------------------------------

if __name__ == "__main__":

    text = """
     **Dr:** नमस्ते, आइए बैठिए। कैसा महसूस हो रहा है आज आपको?\n\n**Client:** नमस्ते... हाँ, ठीक हूँ। मतलब... ठीक तो नहीं हूँ, इसीलिए आया हूँ। बस थोड़ा... नहीं पता, सब कुछ उलझा-उलझा सा लग रहा है इन दिनों।\n\n---\n\n**Dr:** उलझा-उलझा... यह सुनकर लगा कि काफी कुछ चल रहा है मन में। क्या आप थोड़ा और बता सकते हैं — कब से यह महसूस हो रहा है?\n\n**Client:** शायद... दो-तीन महीने से। पहले मैंने सोचा कि काम का बोझ है, थोड़ा आराम मिलेगा तो ठीक हो जाएगा। लेकिन वो आराम आया ही नहीं। और अब तो... रात को भी नींद नहीं आती ठीक से। लेटता हूँ तो दिमाग बंद नहीं होता।\n\n---\n\n**Dr:** रात को नींद नहीं आती — यह बहुत थका देने वाला होता है। जब आप लेटते हैं और दिमाग बंद नहीं होता, तो उस वक्त क्या चलता रहता है मन में?\n\n**Client:** बस... काम की चिंता। यह प्रोजेक्ट पूरा होगा कि नहीं, मेरा मैनेजर क्या सोचता होगा मेरे बारे में, अगले महीने का टारगेट... और फिर उसके बाद घर की बातें आ जाती हैं। पापा की तबियत, छोटी बहन की शादी, EMI... एक के बाद एक। जैसे एक लंबी queue हो और उसका कोई अंत नहीं।\n\n---\n\n**Dr:** एक लंबी queue जिसका अंत नहीं — यह बहुत सटीक कहा आपने। इतनी सारी जिम्मेदारियाँ एक साथ... आप यह सब किससे शेयर करते हैं? कोई है जिससे बात होती है?\n\n**Client:** *(थोड़ी देर चुप रहते हैं)* ...नहीं, ज़्यादा नहीं। दोस्त हैं, लेकिन सबकी अपनी-अपनी life है। और घर में बताऊँगा तो माँ परेशान हो जाएंगी। पापा वैसे ही बीमार हैं। तो लगता है... बस खुद ही handle करो। लेकिन अब खुद भी नहीं हो रहा।\n\n---\n\n**Dr:** यह सुनकर मुझे लगा कि आप काफी अकेले carry कर रहे हैं यह सब। जब आप कहते हैं \"अब खुद भी नहीं हो रहा\" — तो वो कैसा लगता है? उस moment में आप क्या feel करते हैं?\n\n**Client:** एक तरह की... थकान। लेकिन नींद वाली नहीं। अंदर से कुछ खाली-खाली सा। पिछले हफ्ते ऑफिस में था, सब हँस रहे थे किसी बात पर, और मैं वहाँ बैठा था लेकिन जैसे... था ही नहीं। शरीर था, लेकिन मैं कहीं और था। यह डरा गया मुझे।\n\n---\n\n**Dr:** वो पल जब आप वहाँ थे लेकिन \"थे नहीं\" — इसने आपको डराया, यह जानकर अच्छा लगा कि आपने उसे notice किया। क्या ऐसा अक्सर होता है, यह disconnected महसूस करना?\n\n**Client:** हाँ... हाँ, होता है। काम के बीच में अचानक लगता है जैसे सब बेकार है। मेहनत करो, टारगेट पूरा करो, फिर अगला टारगेट। इसका क्या मतलब है? यह सवाल बहुत आता है। और फिर मैं खुद को डाँटता हूँ — \"अरे, इतने लोग इससे भी मुश्किल हालात में हैं, तू क्यों रो रहा है।\"\n\n---\n\n**Dr:** आप खुद को डाँटते हैं — यह बहुत ज़रूरी बात है जो आपने कही। क्या आपको लगता है कि आप जो महसूस कर रहे हैं, उसके लिए आपको खुद को justify करना पड़ता है?\n\n**Client:** *(धीरे से)* हाँ। हमेशा। जैसे मेरी तकलीफ \"काफी बड़ी\" नहीं है। कोई बड़ी accident नहीं हुई, कोई बड़ा नुकसान नहीं हुआ। तो फिर यह क्यों है यह सब? इसीलिए किसी से बोला नहीं। सोचा लोग क्या सोचेंगे — \"इतनी सी बात में रो रहा है।\"\n\n---\n\n**Dr:** आपकी तकलीफ को किसी बड़े कारण की ज़रूरत नहीं होती — वो real है, बस इसलिए कि आप उसे महसूस कर रहे हैं। यह बात आज यहाँ रखना — इसमें हिम्मत लगी। आगे की बात करें — भविष्य के बारे में, आने वाले समय के बारे में आप कैसा सोचते हैं? कोई उम्मीद दिखती है?\n\n**Client:** *(लंबी साँस लेते हैं)* कभी-कभी दिखती है। जैसे... सोचता हूँ कि अगर यह नौकरी छोड़ दूँ तो शायद ठीक होऊँगा। फिर सोचता हूँ — छोड़ूँगा तो घर कैसे चलेगा। तो एक loop है। बाहर निकलने का रास्ता दिखता ही नहीं। बस... चलते रहो, यही समझ आता है। लेकिन कब तक चलूँगा, यह नहीं पता।\n\n---\n\n**Dr:** \"कब तक चलूँगा\" — यह सवाल बहुत भारी है। और आज आपने इसे यहाँ लाने का फैसला किया — यह छोटा कदम नहीं है। अगले कुछ सत्रों में हम इस loop को साथ मिलकर समझने की कोशिश करेंगे — धीरे-धीरे। आपको अकेले नहीं करना यह।\n\n**Client:** *(आँखें थोड़ी नम)* ठीक है... शुक्रिया। बहुत दिनों बाद किसी से बोला हूँ यह सब।\n\n---\n\n*— सत्र समाप्त —*",

    """

    features = extract_lexical_features(text)

    print("\nLexical Features:")
    for key, value in features.items():
        print(f"{key}: {value}")