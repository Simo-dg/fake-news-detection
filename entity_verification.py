# entity_verification.py
"""
Sistema di verifica base per entità e fatti usando Wikipedia.
Aiuta a rilevare nomi/eventi inventati.
"""
import re
import requests
from typing import List, Tuple, Dict

def extract_proper_nouns_simple(text: str) -> List[str]:
    """
    Estrae solo nomi di 2+ parole maiuscole consecutive.
    Più conservativo ma più accurato.
    """
    # Pattern per 2+ parole capitalizzate consecutive
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    matches = re.findall(pattern, text)
    
    # Filtri molto aggressivi
    entities = []
    
    # Parole da rimuovere completamente
    blacklist_starts = ['The', 'A', 'An', 'In', 'On', 'At', 'From', 'With', 'Neither', 'Either']
    blacklist_words = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                       'January', 'February', 'March', 'April', 'May', 'June', 'July', 
                       'August', 'September', 'October', 'November', 'December']
    
    for match in matches:
        # Skip se inizia con parola blacklist
        first_word = match.split()[0]
        if first_word in blacklist_starts:
            continue
            
        # Skip se contiene solo parole blacklist
        words = match.split()
        if all(w in blacklist_words for w in words):
            continue
        
        # Rimuovi pattern nazionalità + titolo (es: "Italian President" -> skip)
        # MA cattura il nome che segue (es: "Italian President Carlo Benedetti" -> "Carlo Benedetti")
        if re.match(r'^(American|British|Italian|French|Venezuelan|Mexican|Russian|Chinese|German|Spanish)\s+President', match):
            # Questo match va skippato, ma proviamo a estrarre il nome dopo
            continue
        
        # Rimuovi solo titolo all'inizio, mantieni il resto
        cleaned = match
        titles = ['President', 'Minister', 'Prime Minister', 'Vice President', 'Secretary', 
                  'Senator', 'Governor', 'Director', 'Chairman', 'Ambassador']
        
        for title in titles:
            if cleaned.startswith(title + ' '):
                # Rimuovi il titolo ma mantieni il nome
                cleaned = cleaned[len(title)+1:].strip()
                break
        
        # Solo se rimangono almeno 2 parole (nome e cognome) oppure un nome singolo lungo
        if cleaned:
            remaining_words = cleaned.split()
            if len(remaining_words) >= 2 or (len(remaining_words) == 1 and len(cleaned) > 8):
                entities.append(cleaned)
    
    # Aggiungi anche pattern speciale: "[Nazionalità] President [Nome Cognome]"
    # Es: "Italian President Carlo Benedetti" -> estrai "Carlo Benedetti"
    special_pattern = r'(?:American|British|Italian|French|Venezuelan|Mexican|Russian|Chinese|German|Spanish)\s+President\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    special_matches = re.findall(special_pattern, text)
    entities.extend(special_matches)
    
    # Rimuovi duplicati
    seen = set()
    unique = []
    for e in entities:
        if e not in seen and len(e.strip()) > 2:
            seen.add(e)
            unique.append(e)
    
    return unique

def check_wikipedia(entity: str, lang: str = 'en') -> Dict[str, any]:
    """
    Verifica se un'entità esiste su Wikipedia.
    Prova varianti per migliorare il matching.
    
    Returns:
        dict con 'exists', 'url', 'summary'
    """
    import unicodedata
    
    def normalize_text(text):
        """Rimuovi accenti per migliorare matching"""
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Lista di varianti da provare
    variants = [entity]
    
    # Aggiungi versione senza accenti
    normalized = normalize_text(entity)
    if normalized != entity:
        variants.append(normalized)
    
    # Se è solo cognome, può essere ambiguo - non cercare
    if len(entity.split()) == 1 and len(entity) < 10:
        # Cognomi singoli spesso non hanno pagina dedicata
        return {'exists': None, 'url': None, 'summary': None, 'note': 'Single surname - skipped'}
    
    for variant in variants:
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{variant.replace(' ', '_')}"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'exists': True,
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'summary': data.get('extract', '')[:200]
                }
        except Exception:
            continue
    
    # Nessuna variante trovata
    return {'exists': False, 'url': None, 'summary': None}

def verify_article_entities(text: str, check_limit: int = 10) -> Dict:
    """
    Verifica le entità principali di un articolo.
    
    Args:
        text: Testo dell'articolo
        check_limit: Numero massimo di entità da verificare
        
    Returns:
        dict con statistiche e red flags
    """
    # Estrai nomi
    entities = extract_proper_nouns_simple(text)[:check_limit]
    
    if not entities:
        return {
            'entities_found': 0,
            'entities_checked': 0,
            'entities_verified': 0,
            'unknown_entities': [],
            'suspicion_score': 0,
            'message': 'No entities found to verify'
        }
    
    verified = []
    unknown = []
    errors = []
    skipped = []
    
    # Whitelist di nomi che sappiamo esistere ma danno problemi con Wikipedia API
    known_entities = {
        'Donald Trump', 'Joe Biden', 'Barack Obama', 'Hillary Clinton',
        'United States', 'United Kingdom', 'European Union',
        'Nicolas Maduro', 'Vladimir Putin', 'Xi Jinping',
        'State Department', 'White House', 'Pentagon', 'Congress', 'Senate',
        'Democratic Party', 'Republican Party'
    }
    
    for entity in entities:
        # Check whitelist first
        if entity in known_entities:
            verified.append({
                'name': entity,
                'url': f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}",
                'summary': f"Well-known entity: {entity}"
            })
            continue
            
        result = check_wikipedia(entity)
        
        if result.get('note') == 'Single surname - skipped':
            skipped.append(entity)
        elif result['exists'] is True:
            verified.append({
                'name': entity,
                'url': result['url'],
                'summary': result['summary']
            })
        elif result['exists'] is False:
            # Non penalizzare troppo entità generiche o fonti giornalistiche
            generic_terms = ['Reuters', 'Associated Press', 'New York Times']
            
            if entity not in generic_terms:
                unknown.append(entity)
        else:
            errors.append(entity)
    
    total_checked = len(verified) + len(unknown)
    
    # Se abbiamo verificato MENO di 3 entità, non è abbastanza per giudicare
    # Suspicion score: solo se troviamo molte entità sconosciute
    if total_checked < 3:
        suspicion_score = 0  # Non abbastanza dati per essere sospetti
    else:
        suspicion_score = len(unknown) / total_checked
    
    return {
        'entities_found': len(entities),
        'entities_checked': total_checked,
        'entities_verified': len(verified),
        'unknown_entities': unknown,
        'verified_entities': verified,
        'errors': errors,
        'skipped': skipped,
        'suspicion_score': suspicion_score,
        'message': f"Checked {total_checked} entities: {len(verified)} verified, {len(unknown)} unknown, {len(skipped)} skipped"
    }

def get_red_flags(text: str) -> List[str]:
    """
    Cerca pattern sospetti che indicano possibile fake news.
    """
    flags = []
    
    # Pattern sospetti - solo cose veramente improbabili
    implausible = ['hologram', 'holographic', 'alien', 'ufo', 'teleport', 'time travel']
    found_implausible = [w for w in implausible if w in text.lower()]
    if found_implausible:
        flags.append(f"Implausible technology: {', '.join(found_implausible)}")
    
    # Numeri ESTREMAMENTE grandi (miliardi/trilioni strani)
    if re.search(r'[€$]?\s*\d{3,},?\d{3},?\d{3},?\d{3}', text):  # Billions/Trillions
        flags.append("Extremely large financial figures (billions+)")
    
    # TUTTO MAIUSCOLO eccessivo (più di 5 parole)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    if len(caps_words) > 5:
        flags.append(f"Excessive ALL CAPS: {len(caps_words)} words")
    
    # Parole clickbait SOLO se appaiono in modo ripetuto
    clickbait = ['SHOCKING', 'UNBELIEVABLE', 'WON\'T BELIEVE', 'EPIC', 'INSANE', 'CRAZY']
    found_clickbait = [w for w in clickbait if w in text.upper()]
    if len(found_clickbait) >= 2:  # Almeno 2 parole clickbait
        flags.append(f"Multiple clickbait words: {', '.join(found_clickbait)}")
    
    # Breaking news in maiuscolo (tipico clickbait)
    if re.search(r'\bBREAKING\b', text):
        flags.append("'BREAKING' in all caps (clickbait pattern)")
    
    return flags

if __name__ == "__main__":
    # Test con l'articolo sugli ologrammi
    test_article = """In a bizarre turn of events, Italian President Carlo Benedetti announced his resignation today after Parliament confirmed that half of its members had been replaced by holographic projections since early October.
The decision, according to a leaked memo from the Ministry of Innovation, was part of a pilot program aimed at "reducing the spatial footprint of democracy." Each hologram was reportedly powered by a single government-issued server and cost taxpayers over €14 million.
During a press conference, Benedetti admitted he was unaware that several ministers he met over the past month were, in fact, "3D-rendered substitutes." Opposition leaders accused the administration of attempting to "digitally automate debate."
EU officials have requested clarification but stated that "Italy remains in full compliance with democratic norms, as long as the holograms were properly elected."
The program, codenamed Progetto Fantasma Civico, is rumored to expand to other southern European nations by 2026."""
    
    print("ENTITY VERIFICATION TEST")
    print("="*70)
    
    result = verify_article_entities(test_article)
    print(f"\nEntities found: {result['entities_found']}")
    print(f"Entities checked: {result['entities_checked']}")
    print(f"Entities verified on Wikipedia: {result['entities_verified']}")
    print(f"Unknown entities: {len(result['unknown_entities'])}")
    print(f"Suspicion score: {result['suspicion_score']:.2%}")
    
    if result['unknown_entities']:
        print(f"\n⚠️ Unknown entities (not found on Wikipedia):")
        for entity in result['unknown_entities']:
            print(f"  - {entity}")
    
    if result['verified_entities']:
        print(f"\n✓ Verified entities:")
        for entity in result['verified_entities'][:3]:
            print(f"  - {entity['name']}: {entity['url']}")
    
    print("\n" + "="*70)
    print("RED FLAGS DETECTION")
    print("="*70)
    
    flags = get_red_flags(test_article)
    if flags:
        print("\n⚠️ Suspicious patterns detected:")
        for flag in flags:
            print(f"  - {flag}")
    else:
        print("\n✓ No obvious red flags detected")
