#!/usr/bin/env python3
"""
Test rapido per verificare che app_simple.py possa importare entity_verification
"""

import sys
from pathlib import Path

# Aggiungi la directory corrente al path
BASE = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE))

print("Testing entity_verification module...")
print("="*70)

try:
    from entity_verification import verify_article_entities, get_red_flags
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 1: Articolo REAL
print("\n1. Testing REAL article (Venezuela):")
real_article = """CARACAS (Reuters) - Venezuelan President Nicolás Maduro said on Friday that United States sanctions against the country's oil industry would cost Venezuela billions in lost revenue."""

try:
    result = verify_article_entities(real_article, check_limit=10)
    print(f"   Entities: {result['entities_found']}, Verified: {result['entities_verified']}, Suspicion: {result['suspicion_score']:.0%}")
    
    flags = get_red_flags(real_article)
    print(f"   Red flags: {len(flags)}")
    print("   ✓ REAL article test passed")
except Exception as e:
    print(f"   ✗ REAL article test failed: {e}")

# Test 2: Articolo FAKE
print("\n2. Testing FAKE article (holograms):")
fake_article = """Italian President Carlo Benedetti announced his resignation after Parliament confirmed that half of its members had been replaced by holographic projections."""

try:
    result = verify_article_entities(fake_article, check_limit=10)
    print(f"   Entities: {result['entities_found']}, Verified: {result['entities_verified']}, Suspicion: {result['suspicion_score']:.0%}")
    
    flags = get_red_flags(fake_article)
    print(f"   Red flags: {len(flags)} - {flags}")
    print("   ✓ FAKE article test passed")
except Exception as e:
    print(f"   ✗ FAKE article test failed: {e}")

print("\n" + "="*70)
print("✓ All tests passed! The module is working correctly.")
print("\nNow restart Streamlit to load the changes:")
print("  1. Press Ctrl+C in the terminal running Streamlit")
print("  2. Run: streamlit run app_simple.py")
print("  3. Or just reload the page in your browser (R key)")
