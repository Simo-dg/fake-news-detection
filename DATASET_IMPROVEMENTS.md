# Come migliorare la detection di Fake News

## Problema attuale

I modelli attuali (BERT e TF-IDF) hanno imparato a riconoscere **STILE**, non **VERITÀ**:

- ✅ Stile Reuters (formale) → classificato come REAL
- ✅ Stile clickbait (MAIUSCOLE, sensazionalismo) → classificato come FAKE
- ❌ Non verificano i fatti

**Esempio**: "Parlamento italiano sostituito da ologrammi" scritto in stile Reuters → classificato REAL al 100%

## Soluzioni

### 1. Dataset con fact-checking verificabile

**Dataset migliori**:
- **LIAR** (Wang 2017): 12.8K notizie con fact-checking da PolitiFact
- **FakeNewsNet** (Shu et al.): Include social context e verifica esterna
- **FEVER** (Fact Extraction and VERification): Claims con evidenza da Wikipedia
- **MultiFC**: Multi-domain fact-checked claims

**Caratteristiche necessarie**:
- Claims verificabili oggettivamente
- Ground truth da fact-checkers professionisti
- Varietà di stili (non solo Reuters vs clickbait)

### 2. Feature engineering avanzate

**Aggiungere feature oltre al testo**:

```python
# Feature aggiuntive
- Named Entity Recognition (persone, luoghi, organizzazioni reali)
- Knowledge Graph verification (controllo con DBpedia, Wikidata)
- Source credibility score
- Writing complexity metrics
- Temporal consistency (eventi cronologicamente plausibili)
```

### 3. External knowledge integration

**Usare knowledge bases**:
- Wikipedia API per verificare nomi/eventi
- DBpedia per entity verification
- Fact-checking APIs (Snopes, PolitiFact)

```python
def verify_entities(text):
    # Extract entities
    entities = extract_named_entities(text)
    
    # Check against Wikipedia
    for entity in entities:
        if not exists_in_wikipedia(entity):
            suspicion_score += 1
    
    return suspicion_score
```

### 4. Ensemble multi-modale

**Combinare segnali diversi**:
1. **Testo**: Stile e contenuto
2. **Metadata**: Source, author, publication date
3. **Network**: Sharing patterns, social media spread
4. **Visual**: Analisi immagini (manipolazioni, stock photos)

### 5. Adversarial training

**Generare esempi difficili durante training**:
```python
# Adversarial examples
- Fake news scritte in stile professionale
- Real news scritte in modo sensazionalistico
- Mixed: fatti reali + interpretazioni false
```

### 6. Claim extraction + verification

**Pipeline a 2 stadi**:

```
Stage 1: Estrarre claims verificabili
  "Italian President Carlo Benedetti resigned"
  → "Carlo Benedetti" non esiste come presidente italiano
  
Stage 2: Verificare ogni claim
  - Check against knowledge base
  - Cross-reference con fonti attendibili
  - Temporal/logical consistency
```

## Implementazione pratica per il tuo progetto

### Opzione A: Quick fix con rule-based filters

```python
def advanced_verification(text, model_prediction):
    suspicion_flags = []
    
    # Check 1: Unknown entities
    entities = extract_entities(text)
    for entity in entities:
        if entity_type == "PERSON" and not in_wikipedia(entity):
            suspicion_flags.append(f"Unknown person: {entity}")
    
    # Check 2: Implausible claims
    if contains_absurd_technology(text):  # hologram parliament
        suspicion_flags.append("Implausible technology claim")
    
    # Check 3: Numerical anomalies
    if extreme_numbers(text):  # €14 million per hologram
        suspicion_flags.append("Unusual financial figures")
    
    # Override prediction if high suspicion
    if len(suspicion_flags) > 2:
        return "UNCERTAIN", suspicion_flags
    
    return model_prediction, suspicion_flags
```

### Opzione B: Transfer learning con dataset migliore

1. Scarica dataset LIAR o FakeNewsNet
2. Fine-tune BERT su quel dataset
3. Applica al tuo caso d'uso

### Opzione C: Ensemble con verifiche esterne

```python
def ensemble_prediction(text):
    # 1. Style-based (current models)
    bert_pred = bert_model.predict(text)
    tfidf_pred = tfidf_model.predict(text)
    
    # 2. Entity verification
    entity_score = verify_entities_wikipedia(text)
    
    # 3. Source credibility (if available)
    source_score = check_source_credibility(text)
    
    # 4. Weighted ensemble
    final_score = (
        0.3 * bert_pred +
        0.2 * tfidf_pred +
        0.3 * entity_score +
        0.2 * source_score
    )
    
    return final_score, confidence_level
```

## Cosa fare ORA per il tuo progetto

### 1. Aggiungi disclaimer nell'app

Comunica chiaramente le limitazioni:

```
⚠️ LIMITAZIONI:
- Questi modelli rilevano STILE, non verificano FATTI
- Addestrati su Reuters (real) vs clickbait (fake)
- Non fanno fact-checking
- Possono essere ingannati da fake news ben scritte
```

### 2. Aggiungi "Uncertainty detection"

```python
if prediction_confidence < 0.6:
    return "UNCERTAIN - Manual fact-checking recommended"
```

### 3. Suggerisci verifiche manuali

Mostra nell'app:
- Link a fact-checking sites (Snopes, PolitiFact)
- Entità estratte da verificare
- Red flags trovati

### 4. (MIGLIORE) Aggiungi NER + Wikipedia check

Implementa verifica base delle entità:
- Estrai persone, luoghi, organizzazioni
- Controlla esistenza su Wikipedia
- Segnala entità sconosciute

## Conclusione

**Per un sistema robusto serve**:
1. Dataset con fact-checking reale (non solo style-based)
2. External knowledge verification
3. Multi-modal signals (testo + metadata + network)
4. Clear communication delle limitazioni

**Per il tuo progetto accademico**:
- Documenta chiaramente i limiti
- Spiega che è style-based detection
- Suggerisci miglioramenti futuri
- Eventualmente aggiungi un layer di NER + Wikipedia verification come proof-of-concept
